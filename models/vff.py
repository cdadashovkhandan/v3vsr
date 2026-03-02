import math
from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, ArrayLike

W0_SCALE = 30.


def uniform_between(a: float, b: float, dtype=jnp.float32) -> Callable:
    def init(key, shape, dtype=dtype) -> Array:
        return jax.random.uniform(key, shape, dtype=dtype, minval=a, maxval=b)
    return init


class VFF(nn.Module):
    """
    A single Video Fourier Field
    """

    dim_out: int
    w0: float = 1.
    c: float = 6.

    @nn.compact
    def __call__(self, x: ArrayLike, scale: ArrayLike, k: ArrayLike, freqs: ArrayLike) -> Array:
        x = jnp.reshape(x, (1, 1, 1, -1)) @ freqs
        phase = self.param('phase', nn.initializers.uniform(0.5), (1, 1, 1, x.shape[-1]))
        x = jnp.sin(x + phase)

        spatial_norm = jnp.linalg.norm(freqs[:, :, :-1], axis=-2)
        x = x * jnp.exp(- spatial_norm**2 * k * scale**-2)

        dim_in = x.shape[-1]
        w_std = math.sqrt(self.c / dim_in) / self.w0
        init_fn = uniform_between(-w_std, w_std)
        kernel = self.param('Conv_0', init_fn, (1, 1, dim_in, self.dim_out))
        x = jnp.matmul(x, kernel)  # e.g., [1, 1, 1, 384] x [1, 1, 384, 3] -> [1, 1, 1, 3]

        return x.squeeze()
