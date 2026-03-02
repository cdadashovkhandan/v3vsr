import pickle
from pathlib import Path
from functools import partial
import math

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from jax.image import resize
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from args import parser
from data import HFEvalVideoFolder, DataShard, make_z_coords
from models import VFF, Hypernetwork
from utils import make_grid, interpolate_grid, seed_all, add_batch_dims
from matlab_bicubic import imresize as matlab_imresize

MEAN = np.array([0.485, 0.456, 0.406])
VAR = np.array([0.229, 0.224, 0.225]) ** 2

SEQ_LEN_ENC = 14
PATCH_SIZE_ENC = 96
OVERLAP_ENC_TIME = 2
OVERLAP_ENC_SPACE = 4
PATCH_SIZE_DEC = 32


def nearest_interp1d(x, coords):
    indices = np.clip(coords.round().astype(np.int32), 0, x.shape[0] - 1)
    return x[indices]


def preprocess_video(target, scale, time_scale):
    target = np.asarray(target)
    target = target.transpose((0, 1, 3, 4, 2))

    source_h, source_w = int(target.shape[-3] / scale), int(target.shape[-2] / scale)
    target = target[..., :source_h * scale, :source_w * scale, :]
    target_coords = jnp.array(make_grid(PATCH_SIZE_DEC * scale))

    source = np.stack([matlab_imresize(t.astype(np.float32) / 255., output_shape=(source_h, source_w))
                       for t in target[0, ::time_scale]])[None]
    source_up = np.asarray(resize(source, source.shape[:2] + target.shape[2:], 'nearest'),
                           dtype=np.float16)
    source = np.asarray(jax.nn.standardize(source, mean=MEAN, variance=VAR)).astype(np.float16)

    target_z = jnp.asarray(make_z_coords(every=time_scale, seq_len=source.shape[1]).numpy())
    target = target[:, :len(target_z)].astype(np.float16) / 255.

    return source, source_up, target_coords, target_z, target


def make_idc(start, shape, size, overlap):
    start = min(start, shape - size)
    end = start + size
    start_out = start if start == 0 else start + overlap
    end_out = end if end == shape else end - overlap
    return start, end, start_out, end_out


def do_inference(val_loader, hyper_model, phi, scale, time_scale, save_dir: Path):
    hyper_model, params = hyper_model.unbind()

    def forward_encoder(params, source):
        apply_fn = jit(partial(hyper_model.apply, method=hyper_model.get_encoding))
        seq_len_enc = min(SEQ_LEN_ENC, source.shape[-4])
        patch_size_enc = min(PATCH_SIZE_ENC, *source.shape[-3:-1])

        stride_t = seq_len_enc - 2 * OVERLAP_ENC_TIME
        stride_s = patch_size_enc - 2 * OVERLAP_ENC_SPACE
        encoding = np.full((*source.shape[:-1], 256), np.nan, dtype=np.float16)

        for t0 in range(0, source.shape[-4], stride_t):
            t0, t1, t0_out, t1_out = make_idc(t0, source.shape[-4], seq_len_enc, OVERLAP_ENC_TIME)
            for h0 in range(0, source.shape[-3], stride_s):
                h0, h1, h0_out, h1_out = make_idc(h0, source.shape[-3], patch_size_enc, OVERLAP_ENC_SPACE)
                for w0 in range(0, source.shape[-2], stride_s):
                    w0, w1, w0_out, w1_out = make_idc(w0, source.shape[-2], patch_size_enc, OVERLAP_ENC_SPACE)
                    source_p = source[:, t0:t1, h0:h1, w0:w1]
                    encoding_p = apply_fn(params, source_p)
                    # shave off borders
                    encoding_p = encoding_p[:,
                        t0_out - t0:-(t1 - t1_out) or None,
                        h0_out - h0:-(h1 - h1_out) or None,
                        w0_out - w0:-(w1 - w1_out) or None]
                    encoding[:, t0_out:t1_out, h0_out:h1_out, w0_out:w1_out, :] = np.asarray(encoding_p)

        assert not np.isnan(encoding).any()
        return encoding

    @jit
    def forward_decoder(params, encoding, target_coords, target_z, scale):
        target_coords = jnp.tile(target_coords, (1, encoding.shape[1], 1, 1, 1))
        target_z = jnp.tile(target_z[None, :, None, None, None], (1, 1, *target_coords.shape[-3:-1], 1))

        phi_params = hyper_model.apply(params, encoding, target_coords,
                                       method=hyper_model.get_params_at_coords)

        # create local coordinate system
        source_coords = jnp.tile(make_grid(encoding.shape[-2]), (*encoding.shape[:2], 1, 1, 1))
        interp_coords = interpolate_grid(target_coords, source_coords)
        rel_coords = (target_coords - interp_coords) * encoding.shape[-2]
        # append z (temporal) coordinates
        rel_coords = jnp.concatenate([rel_coords, target_z], axis=-1)

        # vectorizing map over params and inputs, appending (N, H, W) as batch dims
        apply_phi_batched = add_batch_dims(phi.apply, 4, (0, 0, 0, None, None))
        out = apply_phi_batched(phi_params, rel_coords, jnp.tile(scale, target_coords.shape[:-1]),
                                params['params']['k'], params['params']['freqs'])
        out = out * np.sqrt(VAR) + MEAN
        return out

    for i_vid, target in enumerate(pbar := tqdm(val_loader)):
        source, source_up, target_coords, target_z, target = \
            preprocess_video(target, scale, time_scale)

        assert PATCH_SIZE_DEC <= min(source.shape[-3], source.shape[-2]), \
            f'PATCH_SIZE={PATCH_SIZE_DEC} too large for current image at scale={scale}'

        pbar.set_description(f'encoding...')
        encoding = forward_encoder(params, source)[0]
        assert encoding.shape[:-1] == source.shape[1:-1]

        seq_len = encoding.shape[0]
        hr_coords = np.linspace(-0.5, 0.5, (seq_len - 1) * time_scale + 1, dtype=np.float64)
        hr_coords_int = (hr_coords + 0.5) * (seq_len - 1)
        hr_coords_int -= 1e-5  # makes sure the center frame belongs to the previous input frame

        num_patches_h = math.ceil(source.shape[-3] / PATCH_SIZE_DEC)
        num_patches_w = math.ceil(source.shape[-2] / PATCH_SIZE_DEC)
        num_patches_t = math.ceil(target.shape[1] / PATCH_SIZE_DEC)

        out = np.full(target.shape, np.nan, dtype=np.float16)
        decode_step, total_decode_steps = 1, num_patches_h * num_patches_w * num_patches_t
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                for k in range(num_patches_t):
                    h_min = min(i * PATCH_SIZE_DEC, source.shape[-3] - PATCH_SIZE_DEC)
                    h_max = min((i + 1) * PATCH_SIZE_DEC, source.shape[-3])
                    w_min = min(j * PATCH_SIZE_DEC, source.shape[-2] - PATCH_SIZE_DEC)
                    w_max = min((j + 1) * PATCH_SIZE_DEC, source.shape[-2])
                    t_min = min(k * PATCH_SIZE_DEC, target.shape[1] - PATCH_SIZE_DEC)
                    t_max = min((k + 1) * PATCH_SIZE_DEC, target.shape[1])
                    pbar.set_description(f"decoded {decode_step}/{total_decode_steps}")
                    decode_step += 1

                    hr_coords_p = hr_coords_int[t_min:t_max]
                    encoding_p = nearest_interp1d(encoding, hr_coords_p)[None][
                                 :, :, h_min:h_max, w_min:w_max, :]
                    source_up_p = nearest_interp1d(source_up[0], hr_coords_p)[None][
                                  :, :, scale * h_min:scale * h_max, scale * w_min:scale * w_max, :]
                    target_z_p = target_z[t_min:t_max]
                    out_p = forward_decoder(params, encoding_p, target_coords, target_z_p, np.float32(scale))
                    out_p += source_up_p
                    out[:, t_min:t_max, scale * h_min:scale * h_max,
                        scale * w_min:scale * w_max, :] = np.asarray(out_p, dtype=np.float16)

        del encoding, source_up
        assert not np.isnan(out).any()
        out = out.clip(0., 1., out=out)  # in-place

        if save_dir is not None:
            save_dir_ = save_dir / str(i_vid)
            if not save_dir_.exists():
                save_dir_.mkdir(parents=True, exist_ok=True)
            for i_img in range(out.shape[1]):
                Image.fromarray(np.rint(np.array(out[0, i_img] * 255)).astype(np.uint8))\
                    .save(save_dir_ / f'{i_img}.png')
        del out


def build_models(key, args):
    phi = VFF(3)
    key0, key1 = jax.random.split(key, num=2)

    # use sample parameter set to infer sizes of phi's parameters
    sample_params = phi.init(key0, np.ones((3,)), 1., 1., np.ones((1, 1, 3, args.num_basis)))
    sample_params_flat, tree_def = jax.tree_util.tree_flatten(sample_params)
    param_sizes = [p.shape for p in sample_params_flat]

    hyper_net = Hypernetwork(param_sizes, tree_def, args.embed_dims, args.num_blocks, args.depths,
                             args.attention_heads, args.deformable_groups, args.output_dims)
    with open(args.checkpoint_path, 'rb') as fh:
        unpickled = pickle.load(fh)
        params = unpickled['model'] if 'model' in unpickled else unpickled

    return hyper_net.bind(params), phi


def main(args):
    seed_all(args.seed)
    key = jax.random.PRNGKey(args.seed)

    assert args.save_dir is not None

    data_sets = [HFEvalVideoFolder(Path(args.data_dir) / s, DataShard(0, 1)) for s in args.eval_sets]
    data_loaders = [DataLoader(s, batch_size=1, num_workers=1, shuffle=False) for s in data_sets]

    hyper_model, phi = build_models(key, args)

    for eval_set, data_loader in zip(args.eval_sets, data_loaders):
        save_dir = (Path(args.save_dir) / eval_set / f'x{args.space_scale}')
        do_inference(data_loader, hyper_model, phi, args.space_scale, args.time_scale, save_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
