# DISTS implementation for JAX

import jax
import jax.numpy as jnp
import os, sys
import flax.linen as nn
from flax import nnx
import flaxmodels as fm
from torchvision import transforms
import scipy
import utils
import h5py

class L2Pooling(nn.Module):
    channels: int
    filter_size: int = 5
    stride: int = 2
    @nn.compact
    def __call__(self, x):
        a = jnp.hanning(self.filter_size)[1:-1]
        g = jnp.outer(a, a)
        g = g / jnp.sum(g)
        kernel = jnp.tile(g[None, None, :, :], (self.channels, 1, 1, 1))
        padding = (filter_size - 2 ) // 2

        x = x**2
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(self.stride, self.stride),
            padding=((padding, padding), (padding, padding)),
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
            feature_group_count=self.channels,
        )
        return jnp.sqrt(x + 1e-12)



class DISTS(nn.Module):

    def setup(self, load_weights=True):

        key = jax.random.PRNGKey(0)

        self.mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        self.chns = [3,64,128,256,512,512]
        self.alpha = nnx.Param(jax.random.normal(key, (1, sum(self.chns),1,1)))
        self.beta = nnx.Param(jax.random.normal(key, (1, sum(self.chns),1,1)))
        # self.alpha.data.normal_(0.1,0.01)
        # self.beta.data.normal_(0.1,0.01)
        self.dtype = 'float32'
        if load_weights:
            # DISTS weights
            weights = scipy.io.loadmat('dists/weights/alpha_beta.mat')
            self.alpha.data = jnp.array(weights['alpha'])
            self.beta.data = jnp.array(weights['beta'])
            self.param_dict = None

            # VGG16 weights
            ckpt_file = utils.download('dists/weights', 'https://www.dropbox.com/s/ew3vhtlg5kks8mz/vgg16_weights.h5?dl=1')
            self.param_dict = h5py.File(ckpt_file, 'r')


    # Taken from https://github.com/matthias-wright/flaxmodels/blob/main/flaxmodels/vgg/vgg.py
    def _conv_block(self, x, features, num_layers, block_num, dtype='float32'):
        for l in range(num_layers):
            layer_name = f'conv{block_num}_{l + 1}'
            w = self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict[layer_name]['weight']) 
            b = self.bias_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict[layer_name]['bias']) 
            x = nn.Conv(features=features, kernel_size=(3, 3), kernel_init=w, bias_init=b, padding='same', name=layer_name, dtype=dtype)(x)
            act[layer_name] = x
            x = nn.relu(x)
        return x

    @nn.compact
    def get_features(self, x):
        x = (x-self.mean)/self.std

        x = self._conv_block(x, features=64, num_layers=2, block_num=1, dtype=self.dtype)
        x = L2pooling(64)(x)
        h_relu1_2 = x

        x = self._conv_block(x, features=128, num_layers=2, block_num=2, dtype=self.dtype)
        x = L2pooling(128)(x)
        h_relu2_2 = x

        x = self._conv_block(x, features=256, num_layers=3, block_num=3, dtype=self.dtype)
        x = L2pooling(256)(x)
        h_relu_3_3 = x

        x = self._conv_block(x, features=512, num_layers=3, block_num=4, dtype=self.dtype)
        x = L2pooling(512)(x)
        h_relu_4_3 = x

        x = self._conv_block(x, features=512, num_layers=3, block_num=5, dtype=self.dtype)
        h_relu_5_3 = x

        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
 

    @nn.compact
    def __call__(self, x, y, require_grad=False, batch_average=False):
        # if require_grad:
        #     feats0 = nnx.grad(self.get_features(x))
        #     feats1 = nnx.grad(self.get_features(y))
        # else:
        #     # with torch.no_grad()
        feats0 = self.get_features(x)
        feats1 = self.get_features(y)

        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = jnp.split(self.alpha/w_sum, self.chns, axis=1)
        beta = jnp.split(self.beta/w_sum, self.chns, axis=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score



# def prepare_image(image, resize=True):
#     if resize and min(image.size)>256:
#         image = transforms.functional.resize(image,256)
#     image = transforms.ToTensor()(image)
#     return image.unsqueeze(0)


def prepare_image(image, resize=True):
    from jax import image as jax_image
    
    # Convert PIL Image to numpy array
    image_array = jnp.array(image, dtype=jnp.float32) / 255.0
    
    # Resize if needed
    if resize and min(image.size) > 256:
        # image.size is (width, height) in PIL
        height, width = image_array.shape[:2]
        scale = 256 / min(width, height)
        new_height = int(height * scale)
        new_width = int(width * scale)
        image_array = jax_image.resize(image_array, (new_height, new_width, 3))
    
    # Convert to JAX array and normalize to [0, 1]
    image_jax = jnp.array(image_array)

    image_jax = jnp.transpose(image_jax, (2, 0, 1))
    
    # Add batch dimension
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
        score = model.apply(ref, dist)
    
    print(score.item())
    # score: 0.3347
