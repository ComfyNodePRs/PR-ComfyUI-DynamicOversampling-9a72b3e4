import torch
from PIL import Image
import numpy as np
from torch.nn.functional import interpolate
import math

from comfy.k_diffusion import sampling
from comfy.samplers import KSAMPLER
from tqdm.auto import trange

from comfy import latent_formats
latent_factors = torch.tensor(latent_formats.SD15().latent_rgb_factors)

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
    if (tensor == tensor.flatten()[0]).all():
        print("trivial dump")
        return
    tensor = tensor.to('cpu', copy=True)
    tensor -= tensor.min()
    tensor /= tensor.max()
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1).expand(-1,3,-1,-1)
    tensor = tensor.permute(0,2,3,1)[0]
    tensor @= latent_factors
    h,w,c = tensor.shape
    b = np.clip(tensor.numpy() * 255, 0, 255).astype(np.uint8)
    Image.fromarray(b).save("test.png")

@torch.no_grad()
def sample_dynamic(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n,c,h,w = x.shape
    hist = torch.zeros((n,h,w), device=x.device, dtype=x.dtype)
    kx, ky = torch.meshgrid(torch.linspace(-h/2,h/2,h),torch.linspace(-w/2,w/2,w), indexing="ij")
    base_mask = (kx*kx + ky*ky).sqrt().to(x.device)
    kx, ky = torch.meshgrid(torch.linspace(-h/4,h/4,h//2),torch.linspace(-w/4,w/4,w//2), indexing="ij")
    base_submask = (kx*kx + ky*ky).sqrt().to(x.device)
    cw = w-w//4+1
    ch = h-h//4+1
    kernel = torch.ones((1,1,h//4,w//4), dtype=x.dtype, device=x.device)
    prev_denoised = None
    early_terminate = False
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        #Most change is happening where magnitude of dt is greatest -> blur, find peak
        maskl = (i/len(sigmas)*base_mask.max()-base_mask).clip(max=1,min=0).unsqueeze(0).unsqueeze(0)
        maskh = ((i+1)/len(sigmas)*base_mask.max()-base_mask).clip(max=1,min=0).unsqueeze(0).unsqueeze(0)
        mask = torch.fft.ifftshift(maskh*(1-maskl))
        filt = torch.fft.ifft2(mask*torch.fft.fft2(denoised)).real
        if prev_denoised is not None:
            ext = prev_denoised+filt-denoised
            dist = (ext*ext).sum(1)/math.sqrt(-dt)
            hist+= dist
        prev_denoised = denoised
        d = sampling.to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
        if early_terminate:
            break
    summed = torch.nn.functional.conv2d(hist, kernel, groups=1)
    focal = torch.unravel_index(summed.view((n,ch*cw)).argmax(1), (ch,cw))
    focal = torch.stack(focal).transpose(0,1)
    sublocs = []
    subinds = []
    focalm = []
    for j,ind in enumerate(focal):
        focalm.append(summed[j, *ind])
        if summed[j, *ind] > 1000:
            subx = x[j,:,ind[0]:ind[0]+h//4,ind[1]:ind[1]+w//4]
            sublocs.append(subx)
            subinds.append(j)
    if len(sublocs) > 0:
        print("doing subrender")
        subx = torch.stack(sublocs)
        subx = interpolate(subx, size=(h//2,w//2), mode='bicubic')
        subnoise = model.noise[:,:,:h//2,:w//2]
        subnoise = model.inner_model.inner_model.model_sampling.noise_scaling(sigmas[len(sigmas)//2], subnoise, subx, 0)
        subres = sampling.sample_euler(model, subx, sigmas[len(sigmas)//2:], extra_args, callback, disable, s_churn, s_tmin, s_tmax, s_noise)
        #downscale
        subres = interpolate(subres, size=(h//4,w//4))
        #TODO: Does added noise need to be filtered out?
        for j,dn in zip(subinds,subres):
            subx = x[j,:,focal[j][0]:focal[j][0]+h//4,focal[j][1]:focal[j][1]+w//4]
            sub_weight = -1
            if sub_weight == -1:
                subx[:] = dn
            else:
                subx[:] += sub_weight*dn
                subx /= sub_weight+1
    focal[:,0] += h//8
    focal[:,1] += w//8
    print(focal*8)
    print(focalm)
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
