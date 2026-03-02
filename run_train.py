from collections import defaultdict
from functools import partial
from itertools import islice
from pathlib import Path
import pickle
import signal

import numpy as np
import jax
from jax import jit, value_and_grad, pmap, numpy as jnp
from jax.tree_util import tree_map
from jax.experimental import multihost_utils
from flax.core.frozen_dict import freeze, unfreeze
import flax.jax_utils
import optax
import jmp
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb

from args import parser
from models import VFF, Hypernetwork
from utils import (MEAN, VAR, add_batch_dims, make_grid, interpolate_grid, seed_all, RandomRotate,
                   split, inf_iter, save_checkpoint, dprint)
from data import ContinuousWrapper, Adobe240VideoFolder, DataShard
from train_state import TrainState


received_sigterm = False


def handle_sigterm(*_args, **_kwargs):
    global received_sigterm
    dprint("SIGTERM received, setting global terminate flag")
    received_sigterm = True


def prepare_batch(batch):
    batch = {k: jnp.asarray(v.numpy() if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch['source'] = jax.nn.standardize(batch['source'], mean=MEAN, variance=VAR)
    return batch


@jit
def get_metrics(out, target):
    """
    Returns: tuple(loss, dict of metrics)
    """
    mse = (lambda x, y: jnp.mean((x - y) ** 2))(out, target)
    mae = (lambda x, y: jnp.mean(jnp.abs(x - y)))(out, target)
    psnr = -10 * jnp.log10(mse)
    return {'MSE': mse, 'MAE': mae, 'PSNR': psnr}


def forward(apply_fn, field_apply_fn, variables, source, target_coords, target_coords_z,
            target_scale, key, train=False):
    mutable = ['batch_stats'] if train else False
    res = apply_fn(variables, source, target_coords,
                   training=train, mutable=mutable, rngs={'dropout': key})
    phi_params = res[0] if isinstance(res, tuple) else res

    # create local coordinate systems
    source_grid = jnp.asarray(make_grid(args.patch_size))
    source_coords = jnp.tile(source_grid, (*source.shape[:2], 1, 1, 1)).astype(target_coords.dtype)
    interp_coords = interpolate_grid(target_coords, source_coords)
    rel_coords = (target_coords - interp_coords) * source.shape[-2]

    # append z (temporal) coordinates
    target_coords_z = jnp.tile(target_coords_z[:, :, None, None, None],
                               (1, 1, *target_coords.shape[-3:-1], 1))
    rel_coords = jnp.concatenate([rel_coords, target_coords_z], axis=-1)

    # vectorizing map over params and inputs, appending (N, H, W) as batch dims
    in_axes = (0, 0, 0, None, None)
    apply_phi_batched = add_batch_dims(field_apply_fn, 4, in_axes)  # don't map k

    target_scale = jnp.tile(target_scale[:, None, None, None], (1, *target_coords.shape[-4:-1]))
    k = variables['params']['k']
    out = apply_phi_batched(phi_params, rel_coords, target_scale, k, variables['params']['freqs'])
    out = out * np.sqrt(VAR).astype(out.dtype) + MEAN.astype(out.dtype)

    if args.tv_weight > 0.:
        raise NotImplementedError

    return (out, *res[1:]) if isinstance(res, tuple) else out


@partial(pmap, axis_name='num_devices')
def train_step(batch, key, state: TrainState):
    def get_loss_and_metrics(params):
        params_c, batch_c = state.mp_policy.cast_to_compute((params, batch))
        out, new_model_state = forward(
            state.apply_fn, state.field_apply_fn,
            {'params': params_c, 'batch_stats': state.batch_stats},
            batch_c['source'], batch_c['target_coords'], batch_c['target_coords_z'],
            batch_c['target_scale'], key, train=True)
        out = out + batch_c['source_nearest']
        out = state.mp_policy.cast_to_output(out)
        metrics = get_metrics(out, batch['target'])
        loss = metrics[args.loss]
        if state.mp_policy.compute_dtype == jnp.float16:
            loss = state.loss_scale.scale(loss)
        return loss, (metrics, new_model_state)

    (_, (metrics, new_model_state)), grads = value_and_grad(
        get_loss_and_metrics, has_aux=True)(state.params)
    if state.mp_policy.compute_dtype == jnp.float16:
        grads = state.loss_scale.unscale(grads)
    assert jax.tree.leaves(grads)[0].dtype == jnp.float32

    # combine gradients and metrics from all devices
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    metrics = jax.lax.pmean(metrics, axis_name='num_devices')
    # compute optimizer update in the same precision as params
    # grads = policy.cast_to_param(grads)
    assert jax.tree.leaves(grads)[0].dtype == jnp.float32

    grads_finite = jmp.all_finite(grads)
    metrics['grads_finite'] = grads_finite

    # parameter updates happen on each device individually
    # updates, new_opt_state = optimizer.update(grads, opt_state, params)
    # new_params = optax.apply_updates(params, updates)
    new_state = state.apply_gradients(
        grads=grads,
        batch_stats=jax.lax.pmean(new_model_state, axis_name='num_devices')['batch_stats'],
        loss_scale=state.loss_scale.adjust(grads_finite)
    )

    return metrics, new_state


def train(train_loader, val_loader, state, args, i_start):
    # register SIGTERM handler
    signal.signal(signal.SIGTERM, handle_sigterm)

    # replicate state accross devices
    state = flax.jax_utils.replicate(state)

    train_iter = inf_iter(train_loader)
    train_metrics = defaultdict(list)

    for i in (pbar := tqdm(range(i_start, args.n_iter), total=args.n_iter, initial=i_start,
                           disable=jax.process_index() != 0)):
        inner_steps_done = 0
        while inner_steps_done < args.accu_steps:
            batch = prepare_batch(next(train_iter))
            batch = tree_map(partial(split, n_devices=1), batch)
            keys = jnp.stack(jax.random.split(
                jax.random.PRNGKey(i * int(1e9) + inner_steps_done), 1))

            # this helped to prevent timeouts with collective operations
            multihost_utils.sync_global_devices('before_step')

            batch_metrics, state = train_step(batch, keys, state)

            if not batch_metrics['grads_finite'][0]:
                dprint(f'WARN: Grads not all finite in step {i}, repeating')
                continue

            for k, v in batch_metrics.items():
                train_metrics[k].append(v[0].item())
            inner_steps_done += 1

        if received_sigterm:
            dprint('Saving checkpoint and exiting training loop.')
            if jax.process_index() == 0:
                save_checkpoint(args.checkpoint_path, state)
            exit(0)

        if i in (args.freeze_flow_first, args.freeze_encoder_first):
            # replace optimizer by one that doesn't freeze selected param sets
            state = state.replace(tx=make_optimizer(
                state.params, args, freeze_flow=i < args.freeze_flow_first,
                freeze_encoder=i < args.freeze_encoder_first))

        if (i % args.val_every == 0 and i > 0) or i == args.n_iter - 1:
            val_metrics = defaultdict(list)
            variables_c = {'params': state.mp_policy.cast_to_compute(state.params),
                           'batch_stats': state.batch_stats}  # always keep BN stats in fp32
            for batch in islice(inf_iter(val_loader), args.val_samples // jax.device_count() // args.local_batch_size):
                batch = prepare_batch(batch)
                batch = tree_map(partial(split, n_devices=1), batch)
                batch_c = state.mp_policy.cast_to_compute(batch)
                out = pmap(forward, static_broadcasted_argnums=(0, 1))(
                    state.apply_fn, state.field_apply_fn, variables_c,
                    batch_c['source'], batch_c['target_coords'], batch_c['target_coords_z'],
                    batch_c['target_scale'], keys
                )
                out = out + batch_c['source_nearest']
                out = state.mp_policy.cast_to_output(out)
                batch_metrics = get_metrics(out, batch['target'])
                # average metrics over processes
                batch_metrics = multihost_utils.process_allgather(batch_metrics)
                batch_metrics = jax.tree.map(lambda x: x.mean(axis=0), batch_metrics)
                for k, v in batch_metrics.items():
                    val_metrics[k].append(v.item())

            train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
            val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

            if jax.process_index() == 0:
                if not args.no_wandb:
                    wandb.log({k + '/train': v for k, v in train_metrics.items()}, i)
                    wandb.log({k + '/val': v for k, v in val_metrics.items()}, i)

                # save latest checkpoint
                save_checkpoint(args.checkpoint_path, state)

                # save recurrent checkpoint
                if i % 100_000 == 0 and i > 0:
                    rec_name = args.checkpoint_path.name.replace('latest', str(i), 1)
                    save_checkpoint(args.checkpoint_path.parent / rec_name, state)

            pbar.set_postfix_str(f'Val {args.loss}: {val_metrics[args.loss].round(3)}')
            train_metrics = defaultdict(list)


def build_models(key, sample_input, args):
    phi = VFF(3)

    key0, key1, key2, key3, key4, key5 = jax.random.split(key, num=6)

    # use sample parameter set to infer sizes of phi's parameters
    sample_params = phi.init(key0,
        np.ones((3,)), 1., 1., jax.random.normal(key1, shape=(1, 1, 3, args.thera_dim)))
    sample_params_flat, tree_def = jax.tree_util.tree_flatten(sample_params)
    param_sizes = [p.shape for p in sample_params_flat]

    hyper_net = Hypernetwork(param_sizes, tree_def, args.embed_dims, args.num_blocks, args.depths,
        args.attention_heads, args.deformable_groups, args.output_dims, args.use_remat, args.raft_size)
    variables = hyper_net.init(key2, *sample_input)

    # Add frequencies to parameters dict
    variables = unfreeze(variables)
    variables['params']['k'] = jnp.array(args.k)
    shape = (1, 1, 1, args.num_basis)
    norm = np.pi * args.init_scale * (jax.random.uniform(key3, shape=shape) ** .5)
    theta = 2 * np.pi * jax.random.uniform(key4, shape=shape)
    iota = 2 * np.pi * jax.random.uniform(key5, shape=shape)
    x, y, z = norm * jnp.cos(theta), norm * jnp.sin(theta), args.t_init_scale * norm * jnp.cos(iota)
    variables['params']['freqs'] = jnp.concatenate([x, y, z], axis=2)

    # load pretrained encoder checkpoint
    if args.pretrained_raft is not None:
        from flax.serialization import from_bytes
        print(f'Loading pretrained RAFT from {args.pretrained_raft}')
        raft_variables = {'params': variables['params']['encoder']['flow_model']['model']}
        if 'batch_stats' in variables:
            raft_variables['batch_stats'] = variables['batch_stats']['encoder']['flow_model']['model']
        with open(args.pretrained_raft, 'rb') as f:
            raft_variables = from_bytes(raft_variables, f.read())
        variables['params']['encoder']['flow_model']['model'] = raft_variables['params']
        if 'batch_stats' in variables:
            variables['batch_stats']['encoder']['flow_model']['model'] = raft_variables['batch_stats']

    dprint(f'# params: {sum(p.size for p in jax.tree.leaves(variables["params"]))}')

    variables = freeze(variables)
    return hyper_net, variables, phi


def make_optimizer(params, args, freeze_flow=False, freeze_encoder=False):
    schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=args.n_iter)
    optimizer = optax.adamw(schedule)
    if args.accu_steps > 1:
        optimizer = optax.MultiSteps(optimizer, args.accu_steps)
    # in mixed precision training, it can happen that some gradients become (-)inf due to loss
    # scaling. The scaling is automatically adapted based on this, but still we want to ignore
    # these steps.
    optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=args.n_iter)
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optimizer)

    is_flow_model = jax.tree.map_with_path(
        lambda p, _: 'flow_model' in '.'.join(k.key for k in p), params)
    is_encoder = jax.tree.map_with_path(
        lambda p, _: ('encoder' in '.'.join(k.key for k in p)) or ('refine' in '.'.join(k.key for k in p)), params)

    # we append the freeze tx either way, so that states remain compatible
    optimizer = optax.chain(
        optimizer,
        optax.masked(optax.scale(args.encoder_grad_multiplier), is_encoder),
        optax.masked(optax.scale(args.flow_grad_multiplier), is_flow_model),
        optax.transforms.freeze(is_encoder if freeze_encoder else False),
        optax.transforms.freeze(is_flow_model if freeze_flow else False),
    )
    return optimizer


def make_data_loaders(args):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRotate([0, 90, 180, 270]),
    ])

    shard = DataShard(jax.process_index(), jax.device_count())
    data_sets = [
        Adobe240VideoFolder(Path(args.data_dir) / args.train_set, args.seq_len, shard,
                            every=args.every_frame),
        Adobe240VideoFolder(Path(args.data_dir) / args.val_set, args.seq_len, shard,
                            every=args.every_frame)
    ]
    print(f'Rank {jax.process_index()}: Read train set of length {len(data_sets[0])} and val set '
          f'of length {len(data_sets[1])}')

    data_sets = [ContinuousWrapper(
        ds,
        args.patch_size,
        args.seq_len,
        scale_range=args.scale_range,
        augment_scale_range=args.augment_scale_range,
        augment_scale_prob=args.augment_scale_prob,
        transforms=transform
    ) for ds in data_sets]

    data_loaders = [DataLoader(
        ds,
        batch_size=args.local_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0
    ) for ds in data_sets]
    return data_loaders


def main(args):
    jax.distributed.initialize()

    dprint(f'global device count: {jax.device_count()}')

    seed_all(args.seed, jax.process_index())

    data_loaders = make_data_loaders(args)

    sample_batch = prepare_batch(next(iter(data_loaders[0])))
    sample_input = [sample_batch[k] for k in ('source', 'target_coords')]
    # same key for all processes
    hyper_model, variables, phi = build_models(jax.random.PRNGKey(args.seed), sample_input, args)

    # init encoder and convnextblock from checkpoint if requested for post-training
    if args.pretrained_encoder is not None:
        with open(args.pretrained_encoder, 'rb') as fh:
            c = pickle.load(fh)
            variables = unfreeze(variables)
            variables['params']['encoder'] = c['model']['params']['encoder']
            variables['batch_stats']['encoder'] = c['model']['batch_stats']['encoder']
            variables['params']['refine'] = c['model']['params']['refine']
            variables['params']['k'] = c['model']['params']['k']
            variables = freeze(variables)

    optimizer = make_optimizer(
        variables['params'],
        args,
        freeze_encoder=args.freeze_encoder_first != 0,
        freeze_flow=args.freeze_flow_first != 0,
    )

    state = TrainState.create(
        apply_fn=hyper_model.apply,
        field_apply_fn=phi.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'] if 'batch_stats' in variables else {},
        tx=optimizer,
        mp_policy=jmp.get_policy(args.mp_policy),
        loss_scale=jmp.DynamicLossScale(jnp.asarray(2. ** 15)),
        wandb_id=None,
    )

    i_start = 0
    wandb_id = None
    if args.checkpoint_path.exists():
        with open(args.checkpoint_path, 'rb') as fh:
            checkpoint = pickle.load(fh)
            state = state.replace(
                params=checkpoint['model']['params'],
                batch_stats=checkpoint['model']['batch_stats'],
                opt_state=checkpoint['optimizer'],
            )
            if 'loss_scale' in checkpoint:
                state = state.replace(loss_scale=checkpoint['loss_scale'])
            i_start = int(checkpoint['optimizer'][0][1][3][2].count) + 1
            state = state.replace(tx=make_optimizer(
                state.params,
                args,
                freeze_flow=i_start < args.freeze_flow_first,
                freeze_encoder=i_start < args.freeze_encoder_first)
            )
            wandb_id = checkpoint.get('wandb_id', None)
            dprint(f'Resuming from checkpoint {args.checkpoint_path}')
            dprint(f'[i_start={i_start}, wandb_id={wandb_id}]')

    if jax.process_index() == 0 and not args.no_wandb:
        wandb.init(project=args.wandb_project, dir=args.wandb_dir, id=wandb_id, resume='allow')
        wandb.config.update(args, allow_val_change=True)
        state = state.replace(wandb_id=wandb.run.id)

    train(*data_loaders, state, args, i_start)


if __name__ == '__main__':
    args = parser.parse_args()
    # append checkpoint path to config values
    args.checkpoint_path = Path(args.wandb_dir) / \
        (f'params_latest' + (f'-{args.tag}.pkl' if args.tag else '.pkl'))
    main(args)
