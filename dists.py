# DISTS implementation for JAX

import jax
import jax.numpy as jnp
import flax.linen as nn
from torchvision import transforms
import scipy
import numpy as np

class L2Pooling(nn.Module):
    channels: int
    filter_size: int = 5
    stride: int = 2

    @nn.compact
    def __call__(self, x):
        a = jnp.hanning(self.filter_size)[1:-1]
        g = jnp.outer(a, a)
        g = g / jnp.sum(g)
        kernel = jnp.tile(g[:, :, None, None], (1, 1, 1, self.channels))
        padding = (self.filter_size - 2) // 2

        x = x**2
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(self.stride, self.stride),
            padding=((padding, padding), (padding, padding)),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.channels,
        )
        return jnp.sqrt(x + 1e-12)


class DISTS(nn.Module):

    def setup(self):
        key = jax.random.PRNGKey(0)
        self.mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        self.chns = [3, 64, 128, 256, 512, 512]
        self.dtype = 'float32'
        self.param_dict = scipy.io.loadmat('dists/weights/net_param.mat')

        weights = scipy.io.loadmat('dists/weights/alpha_beta.mat')
    
        # Create and initialize variables with reshaped weights
        alpha_data = jnp.array(weights['alpha'], dtype=self.dtype).reshape((1, 1, 1, -1))
        beta_data = jnp.array(weights['beta'], dtype=self.dtype).reshape((1, 1, 1, -1))
        
        self.alpha = self.variable('params', 'alpha', lambda: alpha_data)
        self.beta = self.variable('params', 'beta', lambda: beta_data)

    def _conv_block(self, x, features, num_layers, block_num, dtype='float32'):
        for l in range(num_layers):
            layer_name = f'conv{block_num}_{l + 1}'
            w = lambda *_ : jnp.array(self.param_dict[layer_name + "_weight"], dtype=self.dtype)
            b = lambda *_ : jnp.array(self.param_dict[layer_name + "_bias"], dtype=self.dtype)
            x = nn.Conv(features=features, kernel_size=(3, 3), kernel_init=w, bias_init=b, 
                       padding='same', name=layer_name, dtype=dtype)(x)
            x = nn.relu(x)
        return x

    @nn.compact
    def get_features(self, x):
        x0 = x
        x = (x - self.mean) / self.std

        x = self._conv_block(x, features=64, num_layers=2, block_num=1, dtype=self.dtype)
        h_relu1_2 = x
        x = L2Pooling(64)(x)

        x = self._conv_block(x, features=128, num_layers=2, block_num=2, dtype=self.dtype)
        h_relu2_2 = x
        x = L2Pooling(128)(x)

        x = self._conv_block(x, features=256, num_layers=3, block_num=3, dtype=self.dtype)
        h_relu3_3 = x
        x = L2Pooling(256)(x)

        x = self._conv_block(x, features=512, num_layers=3, block_num=4, dtype=self.dtype)
        h_relu4_3 = x
        x = L2Pooling(512)(x)

        x = self._conv_block(x, features=512, num_layers=3, block_num=5, dtype=self.dtype)
        h_relu5_3 = x

        return [x0, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    @nn.compact
    def __call__(self, x, y, require_grad=False, batch_average=False):
        feats0 = self.get_features(x)
        feats1 = self.get_features(y)

        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6

        w_sum = self.alpha.value.sum() + self.beta.value.sum()

        splits = tuple(np.cumsum(np.array(self.chns))[:-1])
        alpha_split = jnp.split(self.alpha.value / w_sum, splits, axis=3)
        beta_split = jnp.split(self.beta.value / w_sum, splits, axis=3)

        for k in range(len(self.chns)):
            x_mean = feats0[k].mean((1, 2), keepdims=True)
            y_mean = feats1[k].mean((1, 2), keepdims=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha_split[k] * S1).sum(3, keepdims=True)

            x_var = ((feats0[k] - x_mean)**2).mean((1, 2), keepdims=True)
            y_var = ((feats1[k] - y_mean)**2).mean((1, 2), keepdims=True)
            xy_cov = (feats0[k] * feats1[k]).mean((1, 2), keepdims=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta_split[k] * S2).sum(3, keepdims=True)

        score = 1 - (dist1 + dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score

def prepare_image(image, resize=True):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)

    image_jax = jnp.array(image)
    # CHW to HWC
    image_jax = jnp.transpose(image_jax, (1, 2, 0))

    return jnp.expand_dims(image_jax, 0)

if __name__ == '__main__':

    from PIL import Image
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='dists/images/r0.png')
    parser.add_argument('--dist', type=str, default='dists/images/r1.png')
    args = parser.parse_args()
    
    ref = prepare_image(Image.open(args.ref).convert("RGB"))
    dist = prepare_image(Image.open(args.dist).convert("RGB"))
    assert ref.shape == dist.shape

    available_backends = [str(d.platform) for d in jax.devices()]
    if 'gpu' in available_backends:
        device = jax.devices('gpu')[0]
    else:
        device = jax.devices('cpu')[0]    

    model = DISTS()
    ref = jax.device_put(ref, device)
    dist = jax.device_put(dist, device)
    key = jax.random.PRNGKey(0)
    
    with jax.default_device(device):
        init_key, dropout_key = jax.random.split(key)

        variables = model.init({"params": init_key}, ref, dist)
        score = model.apply(variables, ref, dist)
    
    print(score)
    # score: 0.3347
