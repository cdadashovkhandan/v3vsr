import ctypes
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp

path = next((Path(__file__).parent / 'dist').glob("*.so"))
lib = ctypes.cdll.LoadLibrary(path)
jax.ffi.register_ffi_target(
    "modulated_deformable_im2col_cuda",
    jax.ffi.pycapsule(lib.modulated_deformable_im2col_cuda),
    platform="CUDA"
)
jax.ffi.register_ffi_target(
    "modulated_deformable_col2im_cuda",
    jax.ffi.pycapsule(lib.modulated_deformable_col2im_cuda),
    platform="CUDA"
)
jax.ffi.register_ffi_target(
    "modulated_deformable_col2im_coord_cuda",
    jax.ffi.pycapsule(lib.modulated_deformable_col2im_coord_cuda),
    platform="CUDA"
)


def modulated_deformable_im2col_cuda(
        kv, offset, mask, batch_size, channels, height_im, width_im, height_col, width_col, kernel_h,
        kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deform_group):
    out_shape = (channels * kernel_h * kernel_w, kv.shape[-2] * kv.shape[-1])
    call = jax.ffi.ffi_call(
        "modulated_deformable_im2col_cuda",
        jax.ShapeDtypeStruct(out_shape, jnp.float32),
        vmap_method="broadcast_all",
    )
    orig_dtype = kv.dtype
    return call(
        kv.astype(jnp.float32),
        offset.astype(jnp.float32),
        mask.astype(jnp.float32),
        batch_size=batch_size,
        channels=channels,
        height_im=height_im,
        width_im=width_im,
        height_col=height_col,
        width_col=width_col,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        pad_h=pad_h,
        pad_w=pad_w,
        stride_h=stride_h,
        stride_w=stride_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        deform_group=deform_group
    ).astype(orig_dtype)


def modulated_deformable_col2im_cuda(
        cols, offset, mask, batch_size, channels, height_im, width_im, height_col, width_col, kernel_h,
        kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deform_group):
    out_shape = (channels, height_im, width_im)  # e.g. (576, 128, 128)
    call = jax.ffi.ffi_call(
        "modulated_deformable_col2im_cuda",
        jax.ShapeDtypeStruct(out_shape, jnp.float32),
        vmap_method="broadcast_all",
    )
    orig_dtype = cols.dtype
    return call(
        cols.astype(jnp.float32),
        offset.astype(jnp.float32),
        mask.astype(jnp.float32),
        batch_size=batch_size,
        channels=channels,
        height_im=height_im,
        width_im=width_im,
        height_col=height_col,
        width_col=width_col,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        pad_h=pad_h,
        pad_w=pad_w,
        stride_h=stride_h,
        stride_w=stride_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        deform_group=deform_group
    ).astype(orig_dtype)


def modulated_deformable_col2im_coord_cuda(
        cols, im, offset, mask, batch_size, channels, height_im, width_im, height_col, width_col, kernel_h,
        kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deform_group):
    call = jax.ffi.ffi_call(
        "modulated_deformable_col2im_coord_cuda",
        jax.ShapeDtypeStruct(offset.shape, jnp.float32),
        vmap_method="broadcast_all",
    )
    orig_dtype = cols.dtype
    return call(
        cols.astype(jnp.float32),
        im.astype(jnp.float32),
        offset.astype(jnp.float32),
        mask.astype(jnp.float32),
        batch_size=batch_size,
        channels=channels,
        height_im=height_im,
        width_im=width_im,
        height_col=height_col,
        width_col=width_col,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        pad_h=pad_h,
        pad_w=pad_w,
        stride_h=stride_h,
        stride_w=stride_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        deform_group=deform_group
    ).astype(orig_dtype)


#@partial(jax.jit, static_argnums=list(range(3, 14)))
def deform_attn_fwd_impl(q, kv, offset, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                    dilation_w, attn_head, deform_group, clip_size):
    batch = q.shape[0]
    _, _, kv_channels, height, width = kv.shape
    channels = kv_channels // 2
    area = height * width
    attn_dim = channels // attn_head
    attn_size = kernel_h * kernel_w
    attn_scale = attn_dim ** -0.5

    # resize inputs
    q = q.reshape(batch, 1, attn_head, attn_dim, area).transpose(0, 2, 4, 1, 3)
    q = q * attn_scale
    offset = offset.reshape(batch, clip_size, offset.shape[1] // clip_size, area)

    output = jnp.zeros((batch, attn_head, attn_dim, height, width), dtype=q.dtype)

    # resize temporary columns and attns
    columns = jnp.zeros((clip_size, kv_channels * attn_size, area), dtype=q.dtype)
    # attns = jnp.zeros((attn_head, area, 1, clip_size * attn_size), dtype=q.dtype)
    mask_ones = jnp.ones((deform_group * attn_size, area), dtype=q.dtype)

    # This will be unrolled during jitting, so it should be as fast as the original C++ loop
    for b in range(batch):
        for n in range(clip_size):
            columns = columns.at[n].set(modulated_deformable_im2col_cuda(
                kv[b // clip_size][(n + b) % clip_size],
                offset[b][n],
                mask_ones,
                1, kv_channels, height, width, height, width,
                kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, deform_group
            ))

        columns = columns.reshape(clip_size, 2, attn_head, attn_dim, attn_size, area)
        columns = jnp.transpose(columns, (1, 2, 5, 3, 0, 4))
        columns = columns.reshape(2, attn_head, area, attn_dim, -1)

        attns = jax.nn.softmax(jnp.matmul(q[b], columns[0]).astype(jnp.float32), axis=-1)
        attns = attns.astype(q.dtype)

        output = output.at[b].set(
            jnp.matmul(attns, columns[1].transpose(0, 1, 3, 2))
            .transpose(0, 3, 2, 1)
            .reshape(attn_head, attn_dim, height, width))

        columns = columns.reshape(2, attn_head, area, attn_dim, clip_size, attn_size)
        columns = columns.transpose(4, 0, 1, 3, 5, 2).reshape(  # clip_size x attn_head x attn_dim x attn_size x (height*width)
            clip_size, 2 * attn_head * attn_dim * attn_size, area)

    output = output.reshape(batch, 1, channels, height, width)
    return output


#@partial(jax.jit, static_argnums=list(range(4, 15)))
def deform_attn_bwd_impl(q, kv, offset, grad_output, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                    pad_w, dilation_h, dilation_w, attn_head, deform_group, clip_size):
    batch = q.shape[0]
    _, _, kv_channels, height, width = kv.shape
    channels = kv_channels // 2
    area = height * width
    attn_dim = channels // attn_head
    attn_size = kernel_h * kernel_w
    attn_scale = attn_dim ** -0.5

    # reshape inputs
    q = q.reshape(batch, 1, attn_head, attn_dim, area).transpose(0, 2, 4, 1, 3) * attn_scale
    offset = offset.reshape(batch, clip_size, offset.shape[1] // clip_size, area)

    grad_q = jnp.zeros_like(q)
    grad_offset = jnp.zeros_like(offset)
    grad_output = grad_output.reshape(batch, 1, attn_head, attn_dim, area).transpose(0, 2, 4, 1, 3)
    grad_kv = jnp.zeros_like(kv)

    columns = jnp.zeros((clip_size, kv_channels * attn_size, area), dtype=q.dtype)
    # attns = jnp.zeros((attn_head, area, 1, clip_size * attn_size), dtype=q.dtype)
    mask_ones = jnp.ones((deform_group * attn_size, area), dtype=q.dtype)
    # grad_attns = jnp.zeros_like(attns)
    # grad_mask_ones = jnp.zeros_like(mask_ones)

    for b in range(batch):
        # recompute columns via kernel
        for n in range(clip_size):
            columns = columns.at[n].set(modulated_deformable_im2col_cuda(
                kv[b // clip_size][(n + b) % clip_size],
                offset[b][n],
                mask_ones,
                1, kv_channels, height, width, height, width,
                kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, deform_group
            ))

        columns = columns.reshape(clip_size, 2, attn_head, attn_dim, attn_size, area)
        columns = jnp.transpose(columns, (1, 2, 5, 3, 0, 4))
        columns = columns.reshape(2, attn_head, area, attn_dim, -1)

        ### until here identical with forward, ###
        ### should be optimzed away by jit though ###

        attns, softmax_vjp_fn = jax.vjp(
            jax.nn.softmax, jnp.matmul(q[b], columns[0]).astype(jnp.float32))
        grad_attns = jnp.matmul(grad_output[b], columns[1]).astype(jnp.float32)
        grad_attns = softmax_vjp_fn(grad_attns)[0].astype(q.dtype)
        attns = attns.astype(q.dtype)
        columns = columns.at[1].set(jnp.matmul(grad_output[b].transpose(0, 1, 3, 2), attns))

        grad_q = grad_q.at[b].set(
            jnp.matmul(grad_attns, columns[0].transpose(0, 1, 3, 2)) * attn_scale)
        columns = columns.at[0].set(jnp.matmul(q[b].transpose(0, 1, 3, 2), grad_attns) * attn_scale)

        columns = columns.reshape(2, attn_head, area, attn_dim, clip_size, attn_size)
        columns = (columns.transpose(4, 0, 1, 3, 5, 2)
                   .reshape(clip_size, 2 * attn_head * attn_dim * attn_size, area))

        for n in range(clip_size):
            grad_offset = grad_offset.at[b, n].set(modulated_deformable_col2im_coord_cuda(
                columns[n],
                kv[b // clip_size][(n + b) % clip_size],
                offset[b][n],
                mask_ones,
                1, kv_channels, height, width, height, width,
                kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, deform_group
            ))

            grad_kv = grad_kv.at[b // clip_size, (n + b) % clip_size].add(
                modulated_deformable_col2im_cuda(
                    columns[n],
                    offset[b][n],
                    mask_ones,
                    1, kv_channels, height, width, height, width,
                    kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                    dilation_h, dilation_w, deform_group
                ))

    grad_q = grad_q.transpose(0, 1, 3, 4, 2).reshape(batch, channels, height, width)
    grad_q = grad_q[:, None]

    grad_offset = grad_offset.reshape(batch, -1, height, width)

    return grad_q, grad_kv, grad_offset


def deform_attn_fwd(q, kv, offset, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                    dilation_w, attn_head, deform_group, clip_size):
    res = (q, kv, offset)
    out = deform_attn_fwd_impl(*res, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                    dilation_w, attn_head, deform_group, clip_size)
    return out, res


def deform_attn_bwd(kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                    dilation_w, attn_head, deform_group, clip_size, res, grad_output):
    (q, kv, offset) = res
    grad_q, grad_kv, grad_offset = deform_attn_bwd_impl(q, kv, offset, grad_output,
                               kernel_h, kernel_w, stride_h, stride_w,
                               pad_h, pad_w, dilation_h, dilation_w,
                               attn_head, deform_group, clip_size)
    return grad_q, grad_kv, grad_offset


@partial(jax.custom_vjp, nondiff_argnums=list(range(3, 14)))
def deform_attn(q, kv, offset, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                dilation_w, attn_head, deform_group, clip_size):
    return deform_attn_fwd(q, kv, offset, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                dilation_w, attn_head, deform_group, clip_size)[0]


deform_attn.defvjp(deform_attn_fwd, deform_attn_bwd)
