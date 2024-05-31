import torch
from PIL import Image
import numpy as np
from torch.nn.functional import interpolate
import math

from comfy.k_diffusion import sampling
from comfy.samplers import KSAMPLER
from tqdm.auto import trange

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
    h,w,c = tensor.shape
    b = np.clip(tensor.numpy() * 255, 0, 255).astype(np.uint8)
    Image.fromarray(b).save("test.png")

@torch.no_grad()
def sample_dynamic(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1., sub_weight=3, early_terminate=False, resolution_mult=1.25):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n,c,h,w = x.shape
    hist = torch.zeros((n,h,w), device=x.device, dtype=x.dtype)
    kx, ky = torch.meshgrid(torch.linspace(-h/2,h/2,h),torch.linspace(-w/2,w/2,w), indexing="ij")
    base_mask = (kx*kx + ky*ky).sqrt().to(x.device)
    cw = w-w//4+1
    ch = h-h//4+1
    kernel = torch.ones((1,1,h//4,w//4), dtype=x.dtype, device=x.device)
    prev_denoised = None
    oversample_performed = False
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        maskl = (i/len(sigmas)*base_mask.max()-base_mask).clip(max=1,min=0).unsqueeze(0).unsqueeze(0)
        maskh = ((i+1)/len(sigmas)*base_mask.max()-base_mask).clip(max=1,min=0).unsqueeze(0).unsqueeze(0)
        #A filter of where changes are expected
        mask = torch.fft.ifftshift(maskh*(1-maskl))
        #Just the expected changes
        filt = torch.fft.ifft2(mask*torch.fft.fft2(denoised)).real
        if prev_denoised is not None:
            #changes made outside the current sigma schedule
            ext = prev_denoised+filt-denoised
            #The distance of changes relative to the current step
            dist = (ext*ext).sum(1)/math.sqrt(-dt)
            hist+= dist
            #The sum of distances that would be modified by a subrender for each center point
            summed = torch.nn.functional.conv2d(dist, kernel, groups=1)
            #The most valuable of possible center points
            focal = torch.unravel_index(summed.view((n,ch*cw)).argmax(1), (ch,cw))
            focal = torch.stack(focal).transpose(0,1)
            sublocs = []
            subinds = []
            focalm = []
            for j,ind in enumerate(focal):
                focalm.append(summed[j, *ind])
                #TODO: allow for simultaneous selection of multiple zones
                #An arbitrary cutoff
                if summed[j, *ind] > 1000:
                    subx = x[j,:,ind[0]:ind[0]+h//4,ind[1]:ind[1]+w//4]
                    #scale+noise subx
                    sublocs.append(subx)
                    subinds.append(j)
            if len(sublocs) > 0:
                oversample_performed = True
                subx = torch.stack(sublocs)
                sh,sw = h//4,w//4
                subx = interpolate(subx, size=(int(sh*resolution_mult),
                                               int(sw*resolution_mult)), mode='bicubic')
                #The upscaled/interpolated pixels lack higher frequency noise,
                #so additional noise must be added
                subx += torch.randn_like(subx)*(1-math.sqrt(2))*sigma_hat*s_in
                subdn = model(subx, sigma_hat*s_in*math.sqrt(2), **extra_args)
                #downscale
                subdn = interpolate(subdn, size=(sh,sw), mode='bicubic')
                for j,dn in zip(subinds,subdn):
                    subx = denoised[j,:,focal[j][0]:focal[j][0]+h//4,focal[j][1]:focal[j][1]+w//4]
                    if sub_weight == -1:
                        subx[:] = dn
                    else:
                        subx[:] += sub_weight*dn
                        subx /= sub_weight+1
                focal[:,0] += h//8
                focal[:,1] += w//8
                print(focal*8, focalm)
        prev_denoised = denoised
        d = sampling.to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
        if early_terminate and oversample_performed:
            break
    #Debugging code to dump the measured distances
    #summed = torch.nn.functional.conv2d(hist, kernel, groups=1)
    #dump_image(hist)
    return x

class DynamicSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"sub_weight": ("FLOAT", {"default": 3, "min": -1, "step": 0.01}),
                             "resolution_mult": ("FLOAT", {"default": 1.25, "min": 1, "step": 0.01}),
                             "early_terminate": ("BOOLEAN", {"default": False}),}}
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"
    def get_sampler(self, sub_weight, early_terminate, resolution_mult):
        return (KSAMPLER(lambda *args, **kwargs: sample_dynamic(*args, sub_weight=sub_weight, early_terminate=early_terminate, resolution_mult=resolution_mult, **kwargs)),)

class MeasuredSampler(KSAMPLER):
    def __init__(self, sampler):
        self.sampler = sampler.sampler_function
        self.prev_denoised = None
        self.prev_sigma = 0
        super().__init__(self.wrapped_sample)
    def wrapped_sample(self, *args, **kwargs):
        original_callback = kwargs.get("callback", None)
        def callback(args):
            self.callback(args)
            if original_callback is not None:
                original_callback(args)
        self.sigmas = args[2]
        self.steps = len(args[2])
        kwargs["callback"] = callback
        x = args[1]
        n,c,h,w = x.shape
        self.hist = torch.zeros((n,h,w), device=x.device, dtype=torch.float32)
        kx, ky = torch.meshgrid(torch.linspace(-h/2,h/2,h),torch.linspace(-w/2,w/2,w), indexing="ij")
        self.base_mask = (kx*kx + ky*ky).sqrt().to(x.device)
        self.prev_denoised = None
        x =  self.sampler(*args, **kwargs)
        return x

    def callback(self, args):
        denoised = args["denoised"]
        i = args["i"]
        dt = self.sigmas[i+1] - args['sigma_hat']
        if self.prev_denoised is not None:
            maskl = (i/self.steps*self.base_mask.max()-self.base_mask).clip(max=1,min=0).unsqueeze(0).unsqueeze(0)
            maskh = ((i+1)/self.steps*self.base_mask.max()-self.base_mask).clip(max=1,min=0).unsqueeze(0).unsqueeze(0)
            #A filter of where changes are expected
            mask = torch.fft.ifftshift(maskh*(1-maskl))
            #Just the expected changes
            filt = torch.fft.ifft2(mask*torch.fft.fft2(denoised)).real
            #changes made outside the current sigma schedule
            ext = self.prev_denoised+filt-denoised
            #The distance of changes relative to the current step
            dist = (ext*ext).sum(1)/math.sqrt(-dt)
            self.hist+= dist
        self.prev_denoised = denoised

#Blur functions borrowed from comfy_extras/nodes_post_processing
#with slight modifications
def gaussian_blur(latents, sigma=1):
    radius = min(latents.shape[2:])//2
    size = radius*2+1
    maxl = size // 2
    x, y = torch.meshgrid(torch.linspace(-maxl, maxl, size), torch.linspace(-maxl, maxl, size), indexing="ij")
    d = (x * x + y * y) / (2 * sigma * sigma)
    mask =  torch.exp(-d) / (2 * torch.pi * sigma * sigma)
    kernel = (mask / mask.sum())[None, None].to(latents.device)
    padded_latents = torch.nn.functional.pad(latents, [radius]*4, 'reflect')
    blurred = torch.nn.functional.conv2d(padded_latents, kernel, padding=(radius*2+1) // 2, groups=1)
    return blurred[:, :, radius:-radius, radius:-radius]

class MeasuredSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"sampler": ("SAMPLER",),}}
    RETURN_TYPES = ("SAMPLER", "MASK_PROMISE")
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"
    def get_sampler(self, sampler):
        s = MeasuredSampler(sampler)
        return (s, s)

class ResolveMaskPromise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT",), "mask_promise": ("MASK_PROMISE",),
                             "upper_threshold": ("FLOAT", {"default": .8, "step": .01, "min": 0, "max": 1}),
                             "lower_threshold": ("FLOAT", {"default": .2, "step": .01, "min": 0, "max": 1}),
                             "blur_sigma": ("FLOAT", {"default": 0, "min": 0, "step": .01}),}}
    RETURN_TYPES = ("MASK",)

    FUNCTION = "get_mask"
    def get_mask(self, latent, mask_promise, lower_threshold, upper_threshold, blur_sigma):
        #NOTE: latent is only used to ensure this executes after sampling
        hist = mask_promise.hist
        if blur_sigma > 0:
            hist = gaussian_blur(hist.unsqueeze(1), blur_sigma).squeeze(1)
        sorted_hist = hist.flatten(start_dim=1).sort().values
        lower = sorted_hist[:,int((sorted_hist.size(1)-1)*lower_threshold)]
        upper = sorted_hist[:,int((sorted_hist.size(1)-1)*upper_threshold)]
        mask = ((hist-lower)/(upper-lower)).clip(max=1, min=0)
        return (mask.cpu(),)

NODE_CLASS_MAPPINGS = {
    "DynamicSampler": DynamicSampler,
    "MeasuredSampler": MeasuredSamplerNode,
    "ResolveMaskPromise": ResolveMaskPromise,
}
NODE_DISPLAY_NAME_MAPPINGS = {}
