"""
This is a JAX/Flax port of the RAFT model (<https://arxiv.org/abs/2003.12039>) based on the PyTorch
version from https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py.
"""

from typing import Union, Tuple, Optional, Callable, Sequence, Any
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

__all__ = (
    "RAFT",
    "raft_large",
    "raft_small",
)


def grid_sample(img, grid):
    assert img.shape[-3] > 1
    assert grid.ndim == 4 and grid.shape[-1] == 2

    grid = jnp.flip(grid, -1)  # (x, y) -> (y, x) for map_coordinates
    grid = grid.transpose((0, 3, 1, 2))  # (n, h, w, 2) -> (n, 2, h, w) for map_coordinates

    # Note: this does the same as torch grid_sample(align_corners=True, mode='bilinear',
    # padding_mode='zeros')
    map_fn = partial(jax.scipy.ndimage.map_coordinates, order=1, mode='constant')
    return jax.vmap(jax.vmap(map_fn, in_axes=(2, None), out_axes=2))(img, grid)


def make_coords_grid(batch_size: int, h: int, w: int):
    coords = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    coords = jnp.stack(coords[::-1], axis=-1).astype(jnp.float32)
    return jnp.tile(coords[None], (batch_size, 1, 1, 1))


def resize_with_aligned_corners(
    image: jax.Array,
    shape: tuple[int, ...],
    method: str | jax.image.ResizeMethod,
    antialias: bool = True,
):
    """Alternative to jax.image.resize(), which emulates align_corners=True in PyTorch's
    interpolation functions. https://github.com/jax-ml/jax/issues/11206#issuecomment-1423140760"""
    assert method == 'bilinear', 'currently only bilinear interpolation is supported'
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


def upsample_flow(flow, up_mask=None, factor=8):
    """Upsample flow by the input factor (default 8).

    If up_mask is None we just interpolate.
    If up_mask is specified, we upsample using a convex combination of its weights. See paper page 8 and appendix B.
    Note that in appendix B the picture assumes a downsample factor of 4 instead of 8.
    """
    batch_size, h, w, num_channels = flow.shape
    new_h, new_w = h * factor, w * factor

    if up_mask is None:
        # this should be equivalent to torch.nn.functional.interpolate(align_corners=True)
        upsampled_flow = resize_with_aligned_corners(flow, (batch_size, new_h, new_w, num_channels),
            method='bilinear', antialias=False)
        return factor * upsampled_flow

    assert up_mask.shape == (batch_size, h, w, 9 * factor * factor)
    up_mask = up_mask.reshape(batch_size, h, w, 1, 9, factor, factor)
    up_mask = nn.softmax(up_mask, axis=4)  # "convex" == weights sum to 1

    # there's no direct JAX equivalent to torch.nn.functional.unfold, but according to
    # https://github.com/jax-ml/jax/discussions/5968 we can use conv_general_dilated_patches.
    upsampled_flow = jax.lax.conv_general_dilated_patches(factor * flow, filter_shape=(3, 3),
        window_strides=(1, 1), padding='SAME', dimension_numbers=('NHWC', 'IOHW', 'NHWC'))
    upsampled_flow = upsampled_flow.reshape(batch_size, h, w, num_channels, 9, 1, 1)
    upsampled_flow = jnp.sum(up_mask * upsampled_flow, axis=4)
    assert upsampled_flow.shape == (batch_size, h, w, num_channels, factor, factor)

    return (upsampled_flow.transpose((0, 1, 4, 2, 5, 3))
            .reshape(batch_size, new_h, new_w, num_channels))


def Conv(*args, **kwargs) -> Callable:
    """Convenience function that returns an nn.Conv with kaiming normal kernel initialization"""
    init = nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal")
    return nn.Conv(*args, kernel_init=init, **kwargs)


def BatchNorm(*args, **kwargs) -> Callable:
    """Full precision BatchNorm"""
    return nn.BatchNorm(*args, **kwargs, dtype=jnp.float32, param_dtype=jnp.float32)


class Sequential(nn.Module):
    """Custom nn.Sequential which registers in the module tree, and also accepts a keyword arguments
    passed to all layers."""

    layers: Sequence[Callable[..., Any]]

    @nn.compact
    def __call__(self, x, **kwargs):
        for l in self.layers:
            x = l(x, **kwargs)
        return x


class ConvNormActivation(nn.Module):
    out_channels: int
    kernel_size: Tuple[int, ...] = (3, 3)
    stride: Tuple[int, ...] = (1, 1)
    padding: Optional[Union[int, Tuple[int, ...], str]] = None
    groups: int = 1
    norm_layer: Optional[Callable] = BatchNorm
    activation_fn: Optional[Callable] = nn.relu
    use_bias: Optional[bool] = None

    def setup(self) -> None:
        use_bias = self.use_bias
        if use_bias is None:
            use_bias = self.norm_layer is None

        assert len(self.kernel_size) == len(self.stride) == 2

        padding = self.padding
        if padding is None:
            # original padding logic (corresponds to JAXs SAME_LOWER):
            _conv_dim = len(self.kernel_size)
            padding = tuple((self.kernel_size[i] - 1) // 2 for i in range(_conv_dim))

        layers = [Conv(self.out_channels, self.kernel_size, self.stride, padding,
            feature_group_count=self.groups, use_bias=use_bias)]

        if self.norm_layer is not None:
            layers.append(self.norm_layer())

        if self.activation_fn is not None:
            layers.append(self.activation_fn)

        self.layers = layers

    @nn.compact
    def __call__(self, x, train: bool):
        for l in self.layers:
            kwargs = {'use_running_average': not train} if isinstance(l, nn.BatchNorm) else {}
            x = l(x, **kwargs)
        return x


class ResidualBlock(nn.Module):
    """Slightly modified Residual block with extra relu and biases."""

    out_channels: int
    norm_layer: Callable
    stride: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, train: bool):
        y = x
        y = ConvNormActivation(
            self.out_channels, norm_layer=self.norm_layer, kernel_size=(3, 3),
            stride=self.stride, use_bias=True, name='convnormrelu1')(y, train)
        y = ConvNormActivation(
            self.out_channels, norm_layer=self.norm_layer, kernel_size=(3, 3),
            use_bias=True, name='convnormrelu2')(y, train)

        if self.stride != (1, 1):
            x = ConvNormActivation(
                self.out_channels, norm_layer=self.norm_layer, kernel_size=(1, 1),
                stride=self.stride, use_bias=True, activation_fn=None, name='downsample')(x, train)

        return nn.relu(x + y)


class BottleneckBlock(nn.Module):
    """Slightly modified BottleNeck block (extra relu and biases)"""

    out_channels: int
    norm_layer: Callable
    stride: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, train: bool):
        y = x
        y = ConvNormActivation(
            self.out_channels // 4, norm_layer=self.norm_layer, kernel_size=(1, 1), use_bias=True,
            name='convnormrelu1'
        )(y, train)
        y = ConvNormActivation(
            self.out_channels // 4, norm_layer=self.norm_layer, kernel_size=(3, 3), stride=self.stride,
            use_bias=True, name='convnormrelu2'
        )(y, train)
        y = ConvNormActivation(
            self.out_channels, norm_layer=self.norm_layer, kernel_size=(1, 1), use_bias=True,
            name='convnormrelu3'
        )(y, train)

        if self.stride != (1, 1):
            x = ConvNormActivation(
                self.out_channels, norm_layer=self.norm_layer, kernel_size=(1, 1), stride=self.stride,
                use_bias=True, activation_fn=None, name='downsample'
            )(x, train)

        return nn.relu(x + y)


class FeatureEncoder(nn.Module):
    """The feature encoder, used both as the actual feature encoder, and as the context encoder.

    It must downsample its input by 8.
    """

    block: nn.Module = ResidualBlock
    layers: Tuple[int] = (64, 64, 96, 128, 256)
    strides: Tuple[Tuple[int]] = ((2, 2), (1, 1), (2, 2), (2, 2))
    norm_layer: Callable = BatchNorm

    def _make_2_blocks(self, out_channels, first_stride, name=None):
        block1 = self.block(out_channels, norm_layer=self.norm_layer, stride=first_stride)
        block2 = self.block(out_channels, norm_layer=self.norm_layer, stride=(1, 1))
        return Sequential([block1, block2], name=name)

    def setup(self) -> None:
        assert len(self.layers) == 5

        self.convnormrelu = ConvNormActivation(
            self.layers[0], norm_layer=self.norm_layer, kernel_size=(7, 7),
            stride=self.strides[0], use_bias=True)

        self.layer1 = self._make_2_blocks(self.layers[1], self.strides[1])
        self.layer2 = self._make_2_blocks(self.layers[2], self.strides[2])
        self.layer3 = self._make_2_blocks(self.layers[3], self.strides[3])

        self.conv = Conv(self.layers[4], kernel_size=(1, 1))

    def __call__(self, x, train: bool):
        x = self.convnormrelu(x, train)

        x = self.layer1(x, train=train)
        x = self.layer2(x, train=train)
        x = self.layer3(x, train=train)

        x = self.conv(x)

        return x


class MotionEncoder(nn.Module):
    """The motion encoder, part of the update block.

    Takes the current predicted flow and the correlation features as input and returns an encoded version of these.
    """

    corr_layers: Tuple[int] = (256, 192)
    flow_layers: Tuple[int] = (128, 64)
    out_channels: int = 128

    @nn.compact
    def __call__(self, flow, corr_features, train: bool):
        assert len(self.flow_layers) == 2
        assert len(self.corr_layers) in (1, 2)

        corr = ConvNormActivation(self.corr_layers[0], norm_layer=None, kernel_size=(1, 1),
            name='convcorr1')(corr_features, train)
        if len(self.corr_layers) == 2:
            corr = ConvNormActivation(self.corr_layers[1], norm_layer=None, kernel_size=(3, 3),
                name='convcorr2')(corr, train)

        flow_orig = flow
        flow = ConvNormActivation(
            self.flow_layers[0], norm_layer=None, kernel_size=(7, 7), name='convflow1')(flow, train)
        flow = ConvNormActivation(
            self.flow_layers[1], norm_layer=None, kernel_size=(3, 3), name='convflow2')(flow, train)

        corr_flow = jnp.concatenate([corr, flow], axis=-1)
        corr_flow = ConvNormActivation(
            self.out_channels - 2, norm_layer=None, kernel_size=(3, 3), name='conv')(corr_flow, train)
        return jnp.concatenate([corr_flow, flow_orig], axis=-1)


class ConvGRU(nn.Module):
    """Convolutional Gru unit."""

    hidden_size: int
    kernel_size: Tuple[int, int]
    padding: Tuple[int, int]

    @nn.compact
    def __call__(self, h, x):
        assert len(self.kernel_size) == len(self.padding) == 2
        hx = jnp.concatenate([h, x], axis=-1)
        z = nn.sigmoid(nn.Conv(
            self.hidden_size, kernel_size=self.kernel_size, padding=self.padding, name='convz')(hx))
        r = nn.sigmoid(nn.Conv(
            self.hidden_size, kernel_size=self.kernel_size, padding=self.padding, name='convr')(hx))
        q_in = jnp.concatenate([r * h, x], axis=-1)
        q = nn.tanh(nn.Conv(
            self.hidden_size, kernel_size=self.kernel_size, padding=self.padding, name='convq')(q_in))
        h = (1 - z) * h + z * q
        return h


class RecurrentBlock(nn.Module):
    """Recurrent block, part of the update block.

    Takes the current hidden state and the concatenation of (motion encoder output, context) as input.
    Returns an updated hidden state.
    """

    hidden_size: int
    kernel_size: Tuple = ((1, 5), (5, 1))
    padding: Tuple = ((0, 2), (2, 0))

    @nn.compact
    def __call__(self, h, x):
        assert len(self.kernel_size) == len(self.padding)
        assert len(self.kernel_size) in (1, 2)

        h = ConvGRU(self.hidden_size, self.kernel_size[0], self.padding[0], name='convgru1')(h, x)
        if len(self.kernel_size) == 2:
            h = ConvGRU(self.hidden_size, self.kernel_size[1], self.padding[1], name='convgru2')(h, x)
        return h


class FlowHead(nn.Module):
    """Flow head, part of the update block.

    Takes the hidden state of the recurrent unit as input, and outputs the predicted "delta flow".
    """

    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.hidden_size, (3, 3), padding=1, name='conv1')(x)
        x = nn.relu(x)
        x = nn.Conv(2, (3, 3), padding=1, name='conv2')(x)
        return x


class UpdateBlock(nn.Module):
    """The update block which contains the motion encoder, the recurrent block, and the flow head.

    It must expose a ``hidden_state_size`` attribute which is the hidden state size of its recurrent block.
    """

    motion_encoder: MotionEncoder
    recurrent_block: RecurrentBlock
    flow_head: FlowHead

    @nn.compact
    def __call__(self, hidden_state, context, corr_features, flow, train: bool):
        motion_features = self.motion_encoder(flow, corr_features, train)
        x = jnp.concatenate([context, motion_features], axis=-1)

        hidden_state = self.recurrent_block(hidden_state, x)
        delta_flow = self.flow_head(hidden_state)
        return hidden_state, delta_flow

    @property
    def hidden_state_size(self):
        return self.recurrent_block.hidden_size


class MaskPredictor(nn.Module):
    """Mask predictor to be used when upsampling the predicted flow.

    It takes the hidden state of the recurrent unit as input and outputs the mask.
    This is not used in the raft-small model.
    """

    hidden_size: int
    multiplier: float = 0.25

    @nn.compact
    def __call__(self, x, train: bool):
        x = ConvNormActivation(
            self.hidden_size, norm_layer=None, kernel_size=(3, 3), name='convrelu')(x, train)
        # 8 * 8 * 9 because the predicted flow is downsampled by 8, from the downsampling of the
        # initial FeatureEncoder and we interpolate with all 9 surrounding neighbors.
        # See paper and appendix B.
        x = nn.Conv(8 * 8 * 9, kernel_size=(1, 1), padding=0, name='conv')(x)

        # In the original code, they use a factor of 0.25 to "downweight the gradients" of that
        # branch. See e.g. https://github.com/princeton-vl/RAFT/issues/119#issuecomment-953950419
        # or https://github.com/princeton-vl/RAFT/issues/24.
        # It doesn't seem to affect epe significantly and can likely be set to 1.
        return self.multiplier * x


class CorrBlock:
    """The correlation block.

    Creates a correlation pyramid with ``num_levels`` levels from the outputs of the feature encoder,
    and then indexes from this pyramid to create correlation features.
    The "indexing" of a given centroid pixel x' is done by concatenating its surrounding neighbors that
    are within a ``radius``, according to the infinity norm (see paper section 3.2).
    Note: typo in the paper, it should be infinity norm, not 1-norm.
    """

    def __init__(self, num_levels: int = 4, radius: int = 4):
        self.num_levels = num_levels
        self.radius = radius
        self.out_channels = num_levels * (2 * radius + 1) ** 2

    def build_pyramid(self, fmap1, fmap2) -> jnp.ndarray:
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2)
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """

        assert fmap1.shape == fmap2.shape, "Input feature maps should have the same shapes"

        # Explaining min_fmap_size below: the fmaps are down-sampled (num_levels - 1) times by a factor of 2.
        # The last corr_volume most have at least 2 values (hence the 2* factor), otherwise grid_sample() would
        # produce nans in its output.
        min_fmap_size = 2 * (2 ** (self.num_levels - 1))
        assert not any(fmap_size < min_fmap_size for fmap_size in fmap1.shape[-3:-1]), \
            "Feature maps are too small to be down-sampled by the correlation pyramid. " \
            f"H and W of feature maps should be at least {min_fmap_size}; got: {fmap1.shape[-3:-1]}. " \
            "Remember that input images to the model are downsampled by 8, so that means their " \
            f"dimensions should be at least 8 * {min_fmap_size} = {8 * min_fmap_size}."

        corr_volume = self._compute_corr_volume(fmap1, fmap2)

        batch_size, h, w, _, _, num_channels = corr_volume.shape  # _, _ = h, w
        corr_volume = corr_volume.reshape(batch_size * h * w, h, w, num_channels)
        corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = nn.avg_pool(corr_volume, (2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
            corr_pyramid.append(corr_volume)
        return corr_pyramid

    def index_pyramid(self, corr_pyramid, centroids_coords):
        """Return correlation features by indexing from the pyramid"""

        neighborhood_side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
        di = jnp.linspace(-self.radius, self.radius, neighborhood_side_len)
        dj = jnp.linspace(-self.radius, self.radius, neighborhood_side_len)
        delta = jnp.stack(jnp.meshgrid(di, dj, indexing="ij"), axis=-1)
        delta = delta.reshape((1, neighborhood_side_len, neighborhood_side_len, 2))

        assert centroids_coords.shape[-1] == 2
        batch_size, h, w, _ = centroids_coords.shape
        centroids_coords = centroids_coords.reshape(batch_size * h * w, 1, 1, 2)

        indexed_pyramid = []
        for corr_volume in corr_pyramid:
            sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, side_len, side_len, 2)
            indexed_corr_volume = grid_sample(corr_volume, sampling_coords).reshape((batch_size, h, w, -1))
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords = centroids_coords / 2

        corr_features = jnp.concatenate(indexed_pyramid, axis=-1)
        assert corr_features.shape == (batch_size, h, w, self.out_channels)
        return corr_features

    @staticmethod
    def _compute_corr_volume(fmap1, fmap2):
        batch_size, h, w, num_channels = fmap1.shape
        fmap1 = fmap1.reshape(batch_size, h * w, num_channels)
        fmap2 = fmap2.reshape(batch_size, h * w, num_channels)

        corr = jnp.matmul(fmap1, jnp.swapaxes(fmap2, 1, 2))
        assert corr.shape == (batch_size, h * w, h * w)
        corr = corr.reshape(batch_size, h, w, h, w, 1)
        return corr / np.sqrt(num_channels)


class RAFT(nn.Module):
    """RAFT model from
    `RAFT: Recurrent All Pairs Field Transforms for Optical Flow <https://arxiv.org/abs/2003.12039>`

     args:
        feature_encoder (nn.Module): The feature encoder. It must downsample the input by 8.
            Its input is the concatenation of ``image1`` and ``image2``.
        context_encoder (nn.Module): The context encoder. It must downsample the input by 8.
            Its input is ``image1``. As in the original implementation, its output will be split into 2 parts:

            - one part will be used as the actual "context", passed to the recurrent unit of the ``update_block``
            - one part will be used to initialize the hidden state of the of the recurrent unit of
              the ``update_block``

            These 2 parts are split according to the ``hidden_state_size`` of the ``update_block``, so the output
            of the ``context_encoder`` must be strictly greater than ``hidden_state_size``.

        corr_block (nn.Module): The correlation block, which creates a correlation pyramid from the output of the
            ``feature_encoder``, and then indexes from this pyramid to create correlation features. It must expose
            2 methods:

            - a ``build_pyramid`` method that takes ``feature_map_1`` and ``feature_map_2`` as input (these are the
              output of the ``feature_encoder``).
            - a ``index_pyramid`` method that takes the coordinates of the centroid pixels as input, and returns
              the correlation features. See paper section 3.2.

            It must expose an ``out_channels`` attribute.

        update_block (nn.Module): The update block, which contains the motion encoder, the recurrent unit, and the
            flow head. It takes as input the hidden state of its recurrent unit, the context, the correlation
            features, and the current predicted flow. It outputs an updated hidden state, and the ``delta_flow``
            prediction (see paper appendix A). It must expose a ``hidden_state_size`` attribute.
        mask_predictor (nn.Module, optional): Predicts the mask that will be used to upsample the predicted flow.
            The output channel must be 8 * 8 * 9 - see paper section 3.3, and Appendix B.
            If ``None`` (default), the flow is upsampled using interpolation.
    """

    feature_encoder: nn.Module
    context_encoder: nn.Module
    corr_block: nn.Module
    update_block: nn.Module
    mask_predictor: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, image1, image2, train: bool, num_flow_updates: int = 24):
        batch_size, h, w, _ = image1.shape
        assert (h, w) == image2.shape[-3:-1], "input images should have the same shape"
        assert (h % 8 == 0) and (w % 8 == 0), "input image H and W should be divisible by 8"

        fmaps = self.feature_encoder(jnp.concatenate([image1, image2], axis=0), train)
        fmap1, fmap2 = jnp.split(fmaps, 2, axis=0)
        assert fmap1.shape[-3:-1] == (h // 8, w // 8), \
                "The feature encoder should downsample H and W by 8"

        corr_pyramid = self.corr_block.build_pyramid(fmap1, fmap2)

        context_out = self.context_encoder(image1, train)
        assert context_out.shape[-3:-1] == (h // 8, w // 8), \
                "The context encoder should downsample H and W by 8"

        # As in the original paper, the actual output of the context encoder is split in 2 parts:
        # - one part is used to initialize the hidden state of the recurent units of the update block
        # - the rest is the "actual" context.
        hidden_state_size = self.update_block.hidden_state_size
        out_channels_context = context_out.shape[-1] - hidden_state_size
        assert out_channels_context > 0, \
            f"The context encoder outputs {context_out.shape[1]} channels, but it should have at " \
            f"strictly more than hidden_state={hidden_state_size} channels"

        hidden_state, context = jnp.split(context_out, [hidden_state_size], axis=-1)
        hidden_state = jnp.tanh(hidden_state)
        context = nn.relu(context)

        coords0 = make_coords_grid(batch_size, h // 8, w // 8)
        coords1 = make_coords_grid(batch_size, h // 8, w // 8)

        flow_predictions = []
        for _ in range(num_flow_updates):
            # Don't backpropagate gradients through this branch, see paper
            coords1 = jax.lax.stop_gradient(coords1)
            corr_features = self.corr_block.index_pyramid(corr_pyramid, coords1)

            flow = coords1 - coords0
            hidden_state, delta_flow = self.update_block(
                hidden_state, context, corr_features, flow, train)

            coords1 = coords1 + delta_flow

            up_mask = None if self.mask_predictor is None else self.mask_predictor(hidden_state, train)
            upsampled_flow = upsample_flow(flow=(coords1 - coords0), up_mask=up_mask)
            flow_predictions.append(upsampled_flow)

        """
        def scan_fn(carry, _):
            coords1, hidden_state = carry
            # Don't backpropagate gradients through this branch, see paper
            coords1 = jax.lax.stop_gradient(coords1)
            corr_features = self.corr_block.index_pyramid(corr_pyramid, coords1)

            flow = coords1 - coords0
            hidden_state, delta_flow = self.update_block(
                hidden_state, context, corr_features, flow, train)

            coords1 = coords1 + delta_flow

            up_mask = None if self.mask_predictor is None else self.mask_predictor(hidden_state, train)
            upsampled_flow = upsample_flow(flow=(coords1 - coords0), up_mask=up_mask)
            return (coords1, hidden_state), upsampled_flow

        (c1_final, hs_final), flow_predictions = jax.lax.scan(
            scan_fn, (coords1, hidden_state), xs=None, length=num_flow_updates
        )
        """

        return flow_predictions


def _raft(
    *,
    # Feature encoder
    feature_encoder_layers,
    feature_encoder_block,
    feature_encoder_norm_layer,
    # Context encoder
    context_encoder_layers,
    context_encoder_block,
    context_encoder_norm_layer,
    # Correlation block
    corr_block_num_levels,
    corr_block_radius,
    # Motion encoder
    motion_encoder_corr_layers,
    motion_encoder_flow_layers,
    motion_encoder_out_channels,
    # Recurrent block
    recurrent_block_hidden_state_size,
    recurrent_block_kernel_size,
    recurrent_block_padding,
    # Flow Head
    flow_head_hidden_size,
    # Mask predictor
    use_mask_predictor,
    pretrained_arch=None,
    **kwargs,
):
    feature_encoder = kwargs.pop("feature_encoder", None) or FeatureEncoder(
        block=feature_encoder_block, layers=feature_encoder_layers, norm_layer=feature_encoder_norm_layer
    )
    context_encoder = kwargs.pop("context_encoder", None) or FeatureEncoder(
        block=context_encoder_block, layers=context_encoder_layers, norm_layer=context_encoder_norm_layer
    )

    corr_block = kwargs.pop("corr_block", None) or \
        CorrBlock(num_levels=corr_block_num_levels, radius=corr_block_radius)

    update_block = kwargs.pop("update_block", None)
    if update_block is None:
        motion_encoder = MotionEncoder(
            corr_layers=motion_encoder_corr_layers,
            flow_layers=motion_encoder_flow_layers,
            out_channels=motion_encoder_out_channels,
        )

        recurrent_block = RecurrentBlock(
            hidden_size=recurrent_block_hidden_state_size,
            kernel_size=recurrent_block_kernel_size,
            padding=recurrent_block_padding,
        )

        flow_head = FlowHead(hidden_size=flow_head_hidden_size)

        update_block = UpdateBlock(
            motion_encoder=motion_encoder, recurrent_block=recurrent_block, flow_head=flow_head)

    mask_predictor = kwargs.pop("mask_predictor", None)
    if mask_predictor is None and use_mask_predictor:
        mask_predictor = MaskPredictor(
            hidden_size=256,
            multiplier=0.25,  # See comment in MaskPredictor about this
        )

    model = RAFT(
        feature_encoder=feature_encoder,
        context_encoder=context_encoder,
        corr_block=corr_block,
        update_block=update_block,
        mask_predictor=mask_predictor,
        **kwargs,  # not really needed, all params should be consumed by now
    )

    return model


def raft_large(**kwargs):
    """RAFT model from
    `RAFT: Recurrent All Pairs Field Transforms for Optical Flow <https://arxiv.org/abs/2003.12039>.

    Returns:
        nn.Module: The model.
    """

    return _raft(
        # Feature encoder
        feature_encoder_layers=(64, 64, 96, 128, 256),
        feature_encoder_block=ResidualBlock,
        feature_encoder_norm_layer=partial(
            nn.InstanceNorm, epsilon=1e-5, use_bias=False, use_scale=False),
        # Context encoder
        context_encoder_layers=(64, 64, 96, 128, 256),
        context_encoder_block=ResidualBlock,
        context_encoder_norm_layer=BatchNorm,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=4,
        # Motion encoder
        motion_encoder_corr_layers=(256, 192),
        motion_encoder_flow_layers=(128, 64),
        motion_encoder_out_channels=128,
        # Recurrent block
        recurrent_block_hidden_state_size=128,
        recurrent_block_kernel_size=((1, 5), (5, 1)),
        recurrent_block_padding=((0, 2), (2, 0)),
        # Flow head
        flow_head_hidden_size=256,
        # Mask predictor
        use_mask_predictor=True,
        pretrained_arch=None,
        **kwargs,
    )


def raft_small(**kwargs):
    """RAFT "small" model from
    `RAFT: Recurrent All Pairs Field Transforms for Optical Flow <https://arxiv.org/abs/2003.12039>`.

    Returns:
        nn.Module: The model.
    """

    return _raft(
        # Feature encoder
        feature_encoder_layers=(32, 32, 64, 96, 128),
        feature_encoder_block=BottleneckBlock,
        feature_encoder_norm_layer=partial(
            nn.InstanceNorm, epsilon=1e-5, use_bias=False, use_scale=False),
        # Context encoder
        context_encoder_layers=(32, 32, 64, 96, 160),
        context_encoder_block=BottleneckBlock,
        context_encoder_norm_layer=None,
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=3,
        # Motion encoder
        motion_encoder_corr_layers=(96,),
        motion_encoder_flow_layers=(64, 32),
        motion_encoder_out_channels=82,
        # Recurrent block
        recurrent_block_hidden_state_size=96,
        recurrent_block_kernel_size=((3, 3),),
        recurrent_block_padding=((1, 1),),
        # Flow head
        flow_head_hidden_size=128,
        # Mask predictor
        use_mask_predictor=False,
        pretrained_arch=None,
        **kwargs,
    )
