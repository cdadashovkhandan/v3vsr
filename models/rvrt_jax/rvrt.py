import math
from functools import partial, reduce
from operator import mul
from typing import Callable, Iterable, Optional

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

from utils import MEAN, VAR
from .deform_attn import deform_attn
from ..raft_jax.raft_jax import raft_small, raft_large


class LayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        x = nn.LayerNorm(epsilon=1e-05)(x)
        x = x.astype(orig_dtype)
        return x


def flow_warp(x, flow, padding_mode='constant'):
    """Warp an image or feature map with optical flow.

    Args:
        x: jnp.ndarray of shape (n, h, w, c)
        flow: jnp.ndarray of shape (n, h, w, 2)
    Returns:
        Warped image or feature map (n, h, w, c)
    """
    n, h, w, c = x.shape

    # Create base grid
    grid_y, grid_x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
    base_grid = jnp.stack([grid_x, grid_y], axis=-1)  # (h, w, 2)
    base_grid = base_grid[None, ...]  # (1, h, w, 2)

    # Add flow
    grid = base_grid + flow  # (n, h, w, 2)
    grid = jnp.flip(grid, -1)
    grid = grid.transpose((0, 3, 1, 2))  # (n, h, w, 2) -> (n, 2, h, w)

    # Interpolate
    # Note: this does the same as align_corners=True in torch grid_sample
    map_fn = partial(jax.scipy.ndimage.map_coordinates, order=1, mode=padding_mode)
    return jax.vmap(jax.vmap(map_fn, in_axes=(2, None), out_axes=2))(x, grid)


def resize_with_aligned_corners(
    image: jax.Array,
    shape: tuple[int, ...],
    method: str | jax.image.ResizeMethod,
    antialias: bool = True,
):
    """Alternative to jax.image.resize(), which emulates align_corners=True in PyTorch's
    interpolation functions.
    https://github.com/jax-ml/jax/issues/11206#issuecomment-1423140760"""
    assert method == 'bilinear'
    spatial_dims = tuple(
        i for i in range(len(shape))
        if image.shape[i] != shape[i]
    )
    scale = jnp.array([(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims])
    translation = -(scale / 2.0 - 0.5)
    return jax.image.scale_and_translate(
        image,
        shape,
        method=method,
        scale=scale,
        spatial_dims=spatial_dims,
        translation=translation,
        antialias=antialias,
    )


def make_layer(block: Callable, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(layers)


class Layers(nn.Module):
    block: Callable
    num_blocks: int
    kwarg: dict

    def setup(self):
        layers = []
        for _ in range(self.num_blocks):
            layers.append(self.block(**self.kwarg))
        self.layers = layers

    def __call__(self, x, training):
        for l in self.layers:
            x = l(x, training)
        return x


class RAFTWrapper(nn.Module):
    """Wrapper around RAFT handling normalization & rescaling"""
    size: str
    raft_iters: int
    min_size: int = 128

    def setup(self):
        assert self.size in ('small', 'large')
        self.model = raft_small() if self.size == 'small' else raft_large()

    def preprocess(self, image):
        b, _, _, c = image.shape
        image = image * np.sqrt(VAR).astype(image.dtype) + MEAN.astype(image.dtype)  # -> [0, 1]
        image = jax.image.resize(image, (b, self.min_size, self.min_size, c), "bicubic")
        image = image.clip(0, 1)
        return image * 2. - 1.  # -> [-1, 1]

    def postprocess(self, flow, orig_size):
        flow = jax.image.resize(flow, (flow.shape[0], orig_size, orig_size, 2), "bicubic")
        return flow * (orig_size / self.min_size)

    def __call__(self, image1, image2, train):
        assert image1.shape[-2] == image1.shape[-3]
        orig_size = image1.shape[-2]
        assert orig_size <= self.min_size
        image1, image2 = self.preprocess(image1), self.preprocess(image2)
        flow = self.model(image1, image2, train, num_flow_updates=self.raft_iters)[-1]
        return self.postprocess(flow, orig_size)


class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """
    in_features: int
    hidden_features: int = None
    out_features: int = None
    act_layer: Callable = partial(nn.gelu, approximate=True)

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_features or self.in_features)(x)
        x = self.act_layer(x)
        x = nn.Dense(self.out_features or self.in_features)(x)
        return x


class GuidedDeformAttnPack(nn.Module):
    """Guided deformable attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """
    in_channels: int
    out_channels: int
    attention_window: tuple[int, int] = (3, 3)
    deformable_groups: int = 12
    attention_heads: int = 12
    clip_size: int = 1
    max_residue_magnitude: int = 10

    def setup(self):
        self.kernel_h = self.attention_window[0]
        self.kernel_w = self.attention_window[1]
        self.attn_size = self.kernel_h * self.kernel_w
        self.stride = 1
        self.padding = self.kernel_h // 2
        self.dilation = 1

        self.conv_offset = nn.Sequential([
            nn.Conv(64, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            partial(nn.leaky_relu, negative_slope=0.1),
            nn.Conv(64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            partial(nn.leaky_relu, negative_slope=0.1),
            nn.Conv(64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            partial(nn.leaky_relu, negative_slope=0.1),
            nn.Conv(64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            partial(nn.leaky_relu, negative_slope=0.1),
            nn.Conv(64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            partial(nn.leaky_relu, negative_slope=0.1),
            # weight and bias of this conv should be initialized to 0
            nn.Conv(self.clip_size * self.deformable_groups * self.attn_size * 2,
                    kernel_size=(1, 1, 1), padding=(0, 0, 0),
                    kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros),
        ])

        # proj to a higher dimension can slightly improve the performance
        self.proj_channels = int(self.in_channels * 2)
        self.proj_q = nn.Dense(self.proj_channels)
        self.proj_k = nn.Dense(self.proj_channels)
        self.proj_v = nn.Dense(self.proj_channels)
        self.proj = nn.Dense(self.in_channels)
        self.mlp = Mlp(self.in_channels, self.in_channels * 2, self.in_channels)

    def __call__(self, q, k, v, v_prop_warped, flows, return_updateflow):
        offset = self.conv_offset(jnp.concatenate([q] + v_prop_warped + flows, -1))
        offset1, offset2 = jnp.split(self.max_residue_magnitude * jnp.tanh(offset), 2, axis=-1)

        offset1 = offset1 + jnp.tile(jnp.flip(flows[0], -1), (1, 1, 1, 1, offset1.shape[-1] // 2))
        offset2 = offset2 + jnp.tile(jnp.flip(flows[1], -1), (1, 1, 1, 1, offset2.shape[-1] // 2))
        offset = jnp.concatenate([offset1, offset2], axis=-1)
        offset = offset.reshape((-1, *offset.shape[2:]))  # flatten(0, 1)

        b, t, h, w, c = offset1.shape
        q = self.proj_q(q).reshape((b * t, 1, h, w, self.proj_channels))
        kv = jnp.concatenate([self.proj_k(k), self.proj_v(v)], -1)
        v = deform_attn(
            q.transpose((0, 1, 4, 2, 3)),  # deform_attn is channels-first currently
            kv.transpose((0, 1, 4, 2, 3)),
            offset.transpose((0, 3, 1, 2)),
            kernel_h=self.kernel_h,
            kernel_w=self.kernel_w,
            stride_h=self.stride,
            stride_w=self.stride,
            pad_h=self.padding,
            pad_w=self.padding,
            dilation_h=self.dilation,
            dilation_w=self.dilation,
            attn_head=self.attention_heads,
            deform_group=self.deformable_groups,
            clip_size=self.clip_size,
        )
        v = v.transpose((0, 1, 3, 4, 2))  # back to channels-last
        v = v.reshape(b, t, h, w, self.proj_channels)
        v = self.proj(v)
        v = v + self.mlp(v)

        if return_updateflow:
            return (v,
                    jnp.flip(offset1.reshape(b, t, h, w, c // 2, 2).mean(-2), -1),
                    jnp.flip(offset2.reshape(b, t, h, w, c // 2, 2).mean(-2), -1))
        else:
            return v


def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.reshape((B, D // window_size[0], window_size[0], H // window_size[1], window_size[1],
                   W // window_size[2], window_size[2], C))
    windows = x.transpose((0, 1, 3, 5, 2, 4, 6, 7)).reshape((-1, reduce(mul, window_size), C))
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.reshape((B, D // window_size[0], H // window_size[1], W // window_size[2],
                         window_size[0], window_size[1], window_size[2], -1))
    x = x.transpose((0, 1, 4, 2, 5, 3, 6, 7)).reshape((B, D, H, W, -1))

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    dim: int
    window_size: Iterable[int]
    num_heads: int
    qkv_bias: bool = False
    qk_scale: Optional[float] = None

    def make_rel_pos_index(self):
        d_indices = np.arange(0, self.window_size[0])
        h_indices = np.arange(0, self.window_size[1])
        w_indices = np.arange(0, self.window_size[2])
        indices = np.stack(np.meshgrid(d_indices, h_indices, w_indices, indexing="ij"))  # 3, Wd, Wh, Ww
        flatten_indices = np.reshape(indices, (3, -1))  # 3, Wd*Wh*Ww
        relative_indices = flatten_indices[:, :, None] - flatten_indices[:, None, :]
        relative_indices = np.transpose(relative_indices, (1, 2, 0))
        relative_indices[:, :, 0] += self.window_size[0] - 1
        relative_indices[:, :, 1] += self.window_size[1] - 1
        relative_indices[:, :, 2] += self.window_size[2] - 1

        relative_indices[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_indices[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_pos_index = np.sum(relative_indices, -1)
        return relative_pos_index

    @nn.compact
    def __call__(self, inputs, mask, training):
        rpbt = self.param(
            "relative_position_bias_table",
            nn.initializers.zeros,
            ((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) *
             (2 * self.window_size[2] - 1), self.num_heads)
        )

        batch, n, channels = inputs.shape
        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv")(inputs)
        qkv = qkv.reshape(batch, n, 3, self.num_heads, channels // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.qk_scale or (self.dim // self.num_heads) ** -0.5
        q = q * scale
        # q, k, v values can get pretty big such that the following multiplication can already
        # overflow with float16. We therefore run this code path in float32.
        orig_dtype = q.dtype
        att = q.astype(jnp.float32) @ jnp.swapaxes(k, -2, -1)

        #rel_pos_bias_old = jnp.reshape(
        #    rpbt[np.reshape(self.make_rel_pos_index()[:n, :n], (-1))],
        #    (n, n, -1)
        #)
        rel_pos_bias = jnp.reshape(
            rpbt.take(np.reshape(self.make_rel_pos_index()[:n, :n], (-1))),
            (n, n, -1)
        )
        rel_pos_bias = jnp.transpose(rel_pos_bias, (2, 0, 1))
        att += jnp.expand_dims(rel_pos_bias, 0)

        if mask is not None:
            att = jnp.reshape(
                att, (batch // mask.shape[0], mask.shape[0], self.num_heads, n, n)
            )
            att = att + jnp.expand_dims(jnp.expand_dims(mask[:, :n, :n], 1), 0)
            att = jnp.reshape(att, (-1, self.num_heads, n, n))

        att = jax.nn.softmax(att).astype(orig_dtype)
        x = jnp.reshape(jnp.swapaxes(att @ v, 1, 2), (batch, n, channels))

        # projection
        x = nn.Dense(self.dim, name="proj")(x)

        return x


class STL(nn.Module):
    dim: int
    input_resolution: tuple[int]
    num_heads: int
    window_size: tuple = (2, 8, 8)
    shift_size: tuple = (0, 0, 0)
    mlp_ratio: float = 2.
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    act_layer: Callable = partial(nn.gelu, approximate=True)
    norm_layer: Callable = LayerNorm

    @nn.compact
    def __call__(self, x, mask, training):
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        res1 = x
        x = self.norm_layer()(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = jnp.pad(x, ((0, 0), (pad_d0, pad_d1), (pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = jnp.roll(x, (-shift_size[0], -shift_size[1], -shift_size[2]), axis=(1, 2, 3))
            attn_mask = mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        attn = WindowAttention(self.dim, self.window_size, self.num_heads, self.qkv_bias, self.qk_scale)
        attn_windows = attn(x_windows, attn_mask, training)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape((-1, *(window_size + (C,))))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = jnp.roll(shifted_x, (shift_size[0], shift_size[1], shift_size[2]), axis=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        x = res1 + x
        res2 = x

        x = self.norm_layer()(x)
        x = Mlp(in_features=self.dim, hidden_features=int(self.dim * self.mlp_ratio),
                  act_layer=self.act_layer)(x)

        return res2 + x


class STG(nn.Module):
    dim: int
    input_resolution: tuple
    depth: int
    num_heads: int
    window_size: tuple = (2, 8, 8)
    shift_size: tuple = None
    mlp_ratio: float = 2.
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    norm_layer: Callable = LayerNorm

    @staticmethod
    def make_att_mask(D, H, W, window_size, shift_size, dtype=np.float32):
        img_mask = np.zeros((1, D, H, W, 1), dtype=dtype)  # 1 Dp Hp Wp 1
        cnt = 0
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
        mask_windows = mask_windows[..., 0]  # nW, ws[0]*ws[1]*ws[2]
        attn_mask = mask_windows[:, None] - mask_windows[:, :, None]
        attn_mask = np.where(attn_mask != 0., -100., 0.).astype(dtype)
        return attn_mask

    @nn.compact
    def __call__(self, x, training):
        shift_size = list(i // 2 for i in self.window_size) \
            if self.shift_size is None else self.shift_size

        # calculate attention mask for attention
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, shift_size)
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = self.make_att_mask(Dp, Hp, Wp, window_size, shift_size, dtype=x.dtype)

        for i in range(self.depth):
            x = STL(
                dim=self.dim,
                input_resolution=self.input_resolution,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=[0, 0, 0] if i % 2 == 0 else shift_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                norm_layer=self.norm_layer,
            )(x, attn_mask, training)

        x = x.reshape((B, D, H, W, -1))

        return x


class RSTB(nn.Module):

    dim: int
    input_resolution: tuple
    depth: int
    num_heads: int
    window_size: tuple = (2, 8, 8)
    shift_size: tuple = None
    mlp_ratio: float = 2.
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    norm_layer: Callable = LayerNorm

    @nn.compact
    def __call__(self, x, training):
        group = STG(
            dim=self.dim,
            input_resolution=self.input_resolution,
            depth=self.depth,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            norm_layer=self.norm_layer)

        return x + nn.Dense(self.dim)(group(x, training))


class RSTBWithInputConv(nn.Module):

    kernel_size: tuple = (1, 3, 3)
    stride: tuple = (1, 1, 1)
    groups: int = 1
    num_blocks: int = 2
    rstb_kwargs: dict = None

    @nn.compact
    def __call__(self, x, training):
        x = nn.Conv(
            self.rstb_kwargs['dim'],
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=((self.kernel_size[0] // 2,) * 2, (self.kernel_size[1] // 2,) * 2, (self.kernel_size[2] // 2,) * 2),
            feature_group_count=self.groups,
        )(x)

        x = LayerNorm()(x)

        # rstb = make_layer(RSTB, self.num_blocks, **self.rstb_kwargs)
        rstb = Layers(RSTB, self.num_blocks, self.rstb_kwargs)
        x = rstb(x, training)

        x = LayerNorm()(x)

        return x


class RVRT(nn.Module):
    """ Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT).
        A JAX port of: `Recurrent Video Restoration Transformer with Guided Deformable Attention`  -
          https://arxiv.org/pdf/2205.00000
          https://github.com/JingyunLiang/RVRT

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        clip_size (int): Size of clip in recurrent restoration transformer.
        img_size (int | tuple(int)): Size of input video. Default: [2, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (2,8,8).
        num_blocks (list[int]): Number of RSTB blocks in each stage.
        depths (list[int]): Depths of each RSTB.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        inputconv_groups (int): Group of the first convolution layer in RSTBWithInputConv. Default: [1,1,1,1,1,1]
        deformable_groups (int): Number of deformable groups in deformable attention. Default: 12.
        attention_heads (int): Number of attention heads in deformable attention. Default: 12.
        attention_window (list[int]): Attention window size in aeformable attention. Default: [3, 3].
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
        no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
        no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
        cpu_cache_length: (int): Maximum video length without cpu caching. Default: 100.
    """

    clip_size: int = 2
    img_size: tuple = (2, 64, 64)
    window_size: tuple = (2, 8, 8)
    num_blocks: tuple = (1, 2, 1)
    depths: tuple = (2, 2, 2)
    embed_dims: tuple = (144, 144, 144)
    num_heads: tuple = (6, 6, 6)
    mlp_ratio: float = 2.
    qkv_bias: bool = True
    qk_scale: float = None
    norm_layer: Callable = LayerNorm
    inputconv_groups: tuple = (1, 1, 1, 1, 1, 1)
    max_residue_magnitude: int = 10
    deformable_groups: int = 12
    attention_heads: int = 12
    attention_window: tuple = (3, 3)
    output_dims: int = 64
    raft_size: str = 'large'
    raft_iters: int = 24
    use_remat: bool = False

    def setup(self):
        self.flow_model = RAFTWrapper(self.raft_size, self.raft_iters)

        self.feat_extract = RSTBWithInputConv(
            kernel_size=(1, 3, 3),
            groups=self.inputconv_groups[0],
            num_blocks=self.num_blocks[0],
            rstb_kwargs=dict(
                dim=self.embed_dims[0],
                input_resolution=[1, self.img_size[1], self.img_size[2]],
                depth=self.depths[0],
                num_heads=self.num_heads[0],
                window_size=[1, self.window_size[1], self.window_size[2]],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                norm_layer=self.norm_layer,
            )
        )

        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        deform_align, backbone = {}, {}

        for i, module in enumerate(modules):
            # deformable attention
            cls_ = nn.remat(GuidedDeformAttnPack, static_argnums=6) if self.use_remat else GuidedDeformAttnPack
            deform_align[module] = cls_(
                self.embed_dims[1],
                self.embed_dims[1],
                attention_window=self.attention_window,
                attention_heads=self.attention_heads,
                deformable_groups=self.deformable_groups,
                clip_size=self.clip_size,
                max_residue_magnitude=self.max_residue_magnitude
            )

            # feature propagation
            cls_ = nn.remat(RSTBWithInputConv, static_argnums=2) if self.use_remat else RSTBWithInputConv
            backbone[module] = cls_(
                kernel_size=(1, 3, 3),
                groups=self.inputconv_groups[i + 1],
                num_blocks=self.num_blocks[1],
                rstb_kwargs=dict(
                    dim=self.embed_dims[1],
                    input_resolution=self.img_size,
                    depth=self.depths[1],
                    num_heads=self.num_heads[1],
                    window_size=self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_scale=self.qk_scale,
                    norm_layer=self.norm_layer,
                )
            )

        self.deform_align = deform_align
        self.backbone = backbone

        # reconstruction
        self.reconstruction = RSTBWithInputConv(
            kernel_size=(1, 3, 3),
            groups=self.inputconv_groups[5],
            num_blocks=self.num_blocks[2],
            rstb_kwargs=dict(
                dim=self.embed_dims[2],
                input_resolution=[1, self.img_size[1], self.img_size[2]],
                depth=self.depths[2],
                num_heads=self.num_heads[2],
                window_size=[1, self.window_size[1], self.window_size[2]],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                norm_layer=self.norm_layer,
            )
        )
        self.conv_before_upsampler = nn.Sequential([
            nn.Conv(self.output_dims, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            partial(nn.leaky_relu, negative_slope=0.1),
        ])

    # @staticmethod
    # def is_mirror_extended(lqs):
    #     if lqs.shape[1] % 2 == 0:
    #         lqs_1, lqs_2 = jnp.split(lqs, 2, axis=1)
    #         if jnp.linalg.norm(lqs_1 - jnp.flip(lqs_2, 1)) == 0:
    #             return True
    #     return False

    def compute_flow(self, lqs, is_mirror_extended: bool, train: bool):
        """Compute optical flow for feature alignment.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        n, t, h, w, c = lqs.shape
        lqs_1 = lqs[:, :-1, :, :, :].reshape((-1, h, w, c))
        lqs_2 = lqs[:, 1:, :, :, :].reshape((-1, h, w, c))

        flows_backward = self.flow_model(lqs_1, lqs_2, train).reshape((n, t - 1, h, w, 2))

        if is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.flow_model(lqs_2, lqs_1, train).reshape((n, t - 1, h, w, 2))

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name, updated_flows=None, training=True):
        n, t, h, w, _ = flows.shape

        if 'backward' in module_name:
            flow_idx = range(0, t + 1)[::-1]
            clip_idx = range(0, (t + 1) // self.clip_size)[::-1]
        else:
            flow_idx = range(-1, t)
            clip_idx = range(0, (t + 1) // self.clip_size)

        if '_1' in module_name:
            updated_flows[f'{module_name}_n1'] = []
            updated_flows[f'{module_name}_n2'] = []

        feat_prop = jnp.zeros_like(feats['shallow'][0])

        last_key = list(feats)[-2]
        for i in range(0, len(clip_idx)):
            idx_c = clip_idx[i]
            if i > 0:
                if '_1' in module_name:
                    flow_n01 = flows[:, flow_idx[self.clip_size * i - 1], :, :, :]
                    flow_n12 = flows[:, flow_idx[self.clip_size * i], :, :, :]
                    flow_n23 = flows[:, flow_idx[self.clip_size * i + 1], :, :, :]
                    flow_n02 = flow_n12 + flow_warp(flow_n01, flow_n12)
                    flow_n13 = flow_n23 + flow_warp(flow_n12, flow_n23)
                    flow_n03 = flow_n23 + flow_warp(flow_n02, flow_n23)
                    flow_n1 = jnp.stack([flow_n02, flow_n13], 1)
                    flow_n2 = jnp.stack([flow_n12, flow_n03], 1)
                else:
                    module_name_old = module_name.replace('_2', '_1')
                    flow_n1 = updated_flows[f'{module_name_old}_n1'][i - 1]
                    flow_n2 = updated_flows[f'{module_name_old}_n2'][i - 1]

                if 'backward' in module_name:
                    feat_q = jnp.flip(feats[last_key][idx_c], 1)
                    feat_k = jnp.flip(feats[last_key][clip_idx[i - 1]], 1)
                else:
                    feat_q = feats[last_key][idx_c]
                    feat_k = feats[last_key][clip_idx[i - 1]]

                feat_prop_warped1 = flow_warp(
                    feat_prop.reshape((-1, *feat_prop.shape[2:])),
                    flow_n1.reshape((-1, *flow_n1.shape[2:]))
                )
                feat_prop_warped1 = feat_prop_warped1.reshape((n, feat_prop.shape[1], h, w, feat_prop.shape[-1]))

                feat_prop_flipped = jnp.flip(feat_prop, 1)
                feat_prop_warped2 = flow_warp(
                    feat_prop_flipped.reshape((-1, *feat_prop_flipped.shape[2:])),
                    flow_n2.reshape((-1, *flow_n2.shape[2:]))
                )
                feat_prop_warped2 = feat_prop_warped2.reshape((n, feat_prop.shape[1], h, w, feat_prop.shape[-1]))

                if '_1' in module_name:
                    feat_prop, flow_n1, flow_n2 = self.deform_align[module_name](
                        feat_q, feat_k, feat_prop, [feat_prop_warped1, feat_prop_warped2],
                        [flow_n1, flow_n2], True)
                    updated_flows[f'{module_name}_n1'].append(flow_n1)
                    updated_flows[f'{module_name}_n2'].append(flow_n2)
                else:
                    feat_prop = self.deform_align[module_name](
                        feat_q, feat_k, feat_prop, [feat_prop_warped1, feat_prop_warped2],
                        [flow_n1, flow_n2], False)

            if 'backward' in module_name:
                feat = [jnp.flip(feats[k][idx_c], 1) for k in feats if k not in [module_name]] + [feat_prop]
            else:
                feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]

            feat_prop = feat_prop + self.backbone[module_name](jnp.concatenate(feat, axis=-1), training)
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
            feats[module_name] = [jnp.flip(f, 1) for f in feats[module_name]]

        return feats

    def upsample(self, lqs, feats, training):
        feats['shallow'] = jnp.concatenate(feats['shallow'], 1)
        feats['backward_1'] = jnp.concatenate(feats['backward_1'], 1)
        feats['forward_1'] = jnp.concatenate(feats['forward_1'], 1)
        feats['backward_2'] = jnp.concatenate(feats['backward_2'], 1)
        feats['forward_2'] = jnp.concatenate(feats['forward_2'], 1)

        hr = jnp.concatenate([feats[k] for k in feats], -1)
        hr = self.reconstruction(hr, training)
        hr = self.conv_before_upsampler(hr)

        return hr

    def __call__(self, lqs, training):
        """Forward function for RVRT.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, h, w, c).

        Returns:
            Tensor: Output HR sequence with shape (n, t, 4h, 4w, c).
        """

        n, t, h, w, _ = lqs.shape
        # is_mirror_extended = self.is_mirror_extended(lqs)

        # shallow feature extraction
        feats = dict()
        feats['shallow'] = list(jnp.split(self.feat_extract(lqs, training), t // self.clip_size, axis=1))
        flows_forward, flows_backward = self.compute_flow(lqs, is_mirror_extended=False, train=training)

        # recurrent feature refinement
        updated_flows = {}
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward if flows_forward is not None else flows_backward.flip(1)

                module_name = f'{direction}_{iter_}'
                feats[module_name] = []
                feats = self.propagate(feats, flows, module_name, updated_flows, training=training)

        # reconstruction
        return self.upsample(lqs[:, :, :, :, :3], feats, training)
