import torch
from PIL import Image
import numpy as np

from comfy.k_diffusion import sampling
from comfy.samplers import KSAMPLER
from tqdm.auto import trange

class LatentFFTAsImage:
    """Takes a latent as input , """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latent" : ("LATENT",),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "latentfft_as_image"
    CATEGORY = "latent/advanced"
    def latentfft_as_image(self, latent):
        latent = latent['samples'].permute((0,2,3,1))
        return (latent.sigmoid_(),)

def dump_image(tensor):
    tensor -= tensor.min()
    tensor /= tensor.max()
    n,h,w = tensor.shape
    tensor = tensor[0].unsqueeze(-1).expand(h,w,3)
    b = np.clip(tensor.cpu().numpy() * 255, 0, 255).astype(np.uint8)
    Image.fromarray(b).save("test.png")

@torch.no_grad()
def sample_dynamic(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n,c,h,w = x.shape
    kernel = torch.ones((4,1,h//4,w//4), dtype=x.dtype, device=x.device)
    hist = torch.zeros(n,h-h//4+1,w-w//4+1, dtype=x.dtype, device=x.device)
    base = x.clone()
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = sampling.to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        #Most change is happening where magnitude of dt is greatest -> blur, find peak
        summed = torch.nn.functional.conv2d(d, kernel, groups=4)
        _,_,ch,cw = summed.shape
        dist = (summed*summed).sum(1).sqrt()
        hist += dist
        focal = torch.unravel_index(hist.view((n,ch*cw)).argmax(1), (ch,cw))
        focal = torch.stack(focal)
        print(float(hist[0, *focal.transpose(0,1)[0]]))
        focal[0] += h//8
        focal[1] += w//8
        print(focal.transpose(0,1) * 8)

        # Euler method
        x = x + d * dt
    dif = x-base
    dif = (dif*dif).sum(1).sqrt()*0
    dif[:,h//8:7*h//8+1, w//8:7*w//8+1] -= hist
    dif = (-dif).clip(min=0)
    dump_image(dif)
    return x

class DynamicSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"
    def get_sampler(self):
        return (KSAMPLER(sample_dynamic),)

class LatentAsImage:
    """Converts a latent to an image with minimal processing. Provides a means of viewing
       latent channels visually with minimal overhead"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latent" : ("LATENT",),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "latent_as_image"
    CATEGORY = "latent/advanced"
    def latent_as_image(self, latent):
        latent = latent['samples'].permute((0,2,3,1))
        return (latent.sigmoid_(),)

NODE_CLASS_MAPPINGS = {
    "DynamicSampler": DynamicSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {}
