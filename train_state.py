from typing import Callable, Any

from flax import core, struct
from flax.training import train_state
import jmp


class TrainState(train_state.TrainState):
    """Extension of `flax.training.train_state.TrainState` to include additional fields"""

    # original fields:
    # step: int | jax.Array
    # apply_fn: Callable = struct.field(pytree_node=False)
    # params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    # tx: optax.GradientTransformation = struct.field(pytree_node=False)
    # opt_state: optax.OptState = struct.field(pytree_node=True)

    field_apply_fn: Callable = struct.field(pytree_node=False)
    batch_stats: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    mp_policy: jmp.Policy = struct.field(pytree_node=False)
    loss_scale: jmp.DynamicLossScale = struct.field(pytree_node=True)
    wandb_id: str = struct.field(pytree_node=False)

