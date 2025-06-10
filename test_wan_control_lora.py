import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    raise Exception("unable to initialize CUDA")

import os
import gc
import math
import datetime
import argparse
from tqdm import tqdm
from safetensors.torch import load_file
from torchvision.transforms import v2

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video

import decord
decord.bridge.set_bridge('torch')


def convert_sd(sd):
    converted_sd = {}
    for key in sd.keys():
        renamed_key = key
        
        if ".reshape_weight" in key:
            continue
        
        if "diffusion_model." in renamed_key:
            renamed_key = renamed_key.replace("diffusion_model.", "")
        
        converted_sd[renamed_key] = sd[key].to(torch.float32)
    
    return converted_sd


@torch.inference_mode()
def main(args):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_output_dir = os.path.join(args.output_dir, date_time)
    os.makedirs(real_output_dir, exist_ok=True)
    
    if args.prompt is not None or args.negative_prompt is not None:
        ckpt_dir = "./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"
        tk_dir = "./models/UMT5/google/umt5-xxl"
        umt5_model = T5EncoderModel(
            text_len = 512,
            dtype = torch.bfloat16,
            device = device,
            checkpoint_path = ckpt_dir,
            tokenizer_path = tk_dir,
        )
        
        gc.collect()
        torch.cuda.empty_cache()
        
        positive_context = umt5_model([args.prompt], umt5_model.device)[0].to(device=device, dtype=torch.bfloat16)
        negative_context = umt5_model([args.negative_prompt], umt5_model.device)[0].to(device=device, dtype=torch.bfloat16)
        
        del umt5_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        positive_context = [load_file(args.embedding)["context"].to(device=device, dtype=torch.bfloat16)]
        negative_context = [load_file(args.negative_embedding)["context"].to(device=device, dtype=torch.bfloat16)]
    
    vae = WanVAE(vae_pth="./models/vae/Wan2.1_VAE.pth", dtype=torch.bfloat16, device=device)
    diffusion_model = WanModel.from_pretrained("./models/" + args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    
    lora_sd = load_file(args.lora)
    lora_sd = convert_sd(lora_sd)
    
    if "patch_embedding.lora_A.weight" in lora_sd:
        in_cls = diffusion_model.patch_embedding.__class__ # nn.Conv3d
        old_in_dim = diffusion_model.in_dim # 16
        new_in_dim = lora_sd["patch_embedding.lora_A.weight"].shape[1]
        assert new_in_dim == 32
        
        new_in = in_cls(
            new_in_dim,
            diffusion_model.patch_embedding.out_channels,
            diffusion_model.patch_embedding.kernel_size,
            diffusion_model.patch_embedding.stride,
            diffusion_model.patch_embedding.padding,
        ).to(device=device, dtype=torch.bfloat16)
        
        new_in.weight.zero_()
        new_in.bias.zero_()
        
        new_in.weight[:, :old_in_dim].copy_(diffusion_model.patch_embedding.weight)
        new_in.bias.copy_(diffusion_model.patch_embedding.bias)
        
        diffusion_model.patch_embedding = new_in
        diffusion_model.register_to_config(in_dim=new_in_dim)
    
    diffusion_model.load_lora_adapter(lora_sd, adapter_name="default")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    latent_width = (args.width // 16) * 2
    latent_height = (args.height // 16) * 2
    latent_frames = (args.frames - 1) // 4 + 1
    
    width = latent_width * 8
    height = latent_height * 8
    frames = (latent_frames - 1) * 4 + 1
    
    seq_len = math.ceil((latent_height / 2) * (latent_width / 2) * latent_frames)
    
    torch.manual_seed(args.seed)
    latents = torch.randn(16, latent_frames, latent_height, latent_width).to(device)
    
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(args.steps, device=device, shift=args.shift)
    
    if args.control_video is not None:
        vr = decord.VideoReader(args.control_video)
        control_pixels = vr[:frames]
        control_pixels = control_pixels.movedim(3, 1).unsqueeze(0) # FHWC -> FCHW -> BFCHW
        
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(height // 4, width // 4)),
            v2.Resize(size=(height, width)),
            v2.GaussianBlur(kernel_size=15, sigma=4),
        ])
        
        control_pixels = transform(control_pixels) * 2 - 1
        control_pixels = torch.clamp(torch.nan_to_num(control_pixels), min=-1, max=1)
        control_pixels = control_pixels[0].movedim(0, 1) # BFCHW -> FCHW -> CFHW
        
        control_latents = vae.encode([control_pixels.to(dtype=torch.bfloat16, device=device)])[0].to(device)
        assert control_latents.shape == latents.shape, f"{control_latents.shape} {latents.shape}"
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for t in tqdm(scheduler.timesteps):
            latent_model_input = [torch.cat([latents, control_latents], dim=0)]
            timestep = torch.stack([t])
            
            noise_pred_cond = diffusion_model(
                latent_model_input,
                t = timestep,
                context = positive_context,
                seq_len = seq_len,
            )[0]
            
            noise_pred_uncond = diffusion_model(
                latent_model_input,
                t = timestep,
                context = negative_context,
                seq_len = seq_len,
            )[0]
            
            noise_pred = noise_pred_uncond + args.cfg * (noise_pred_cond - noise_pred_uncond)
            
            latents = scheduler.step(
                noise_pred.unsqueeze(0),
                timestep,
                latents.unsqueeze(0),
                return_dict=False,
            )[0].squeeze(0)
        
        decoded_video = vae.decode([latents])[0].to(device)
        control_pixels = control_pixels.to(device)
        
        if args.width > args.height:
            cat_dim = -2 # H
        else:
            cat_dim = -1 # W
        
        decoded_video = torch.cat([control_pixels, decoded_video], dim=cat_dim)
        
        cache_video(
            tensor = decoded_video[None],
            save_file = os.path.join(real_output_dir, "test.mp4"),
            fps = 16,
            nrow = 1,
            normalize = True,
            value_range = (-1, 1),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description = "HunyuanVideo training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B",
        help="Path to pretrained model or model identifier from huggingface",
    )
    parser.add_argument(
        "--lora",
        type = str,
        default = None,
        help = "LoRA checkpoint to load",
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./outputs/test",
        help = "Output directory for training results",
    )
    parser.add_argument(
        "--prompt",
        type = str,
        default = None,
        help = "Positive prompt to use instead of precalculated embedding",
    )
    parser.add_argument(
        "--negative_prompt",
        type = str,
        default = None,
        help = "Negative prompt to use instead of precalculated embedding",
    )
    parser.add_argument(
        "--embedding",
        type = str,
        default = "./embeddings/default_empty_wan.safetensors",
        help = "Precalculated positive embedding file to use if no prompt is provided",
    )
    parser.add_argument(
        "--negative_embedding",
        type = str,
        default = "./embeddings/default_video_negative_wan.safetensors",
        help = "Precalculated negative embedding file to use if no negative prompt is provided",
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducible training",
    )
    parser.add_argument(
        "--cfg",
        type = float,
        default = 6.0,
        help = "CFG scale",
    )
    parser.add_argument(
        "--steps",
        type = int,
        default = 30,
        help = "Total number of inference steps",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default = 8.0,
        help = "Noise scheduler shift",
    )
    parser.add_argument(
        "--width",
        type = int,
        default = 832,
        help = "Generated width",
    )
    parser.add_argument(
        "--height",
        type = int,
        default = 480,
        help = "Generated height",
    )
    parser.add_argument(
        "--frames",
        type = int,
        default = 33,
        help = "Generated frames",
    )
    parser.add_argument(
        "--control_video",
        type = str,
        default = None,
        help = "Control signal video to use if lora is a control lora",
    )
    parser.add_argument(
        "--control_preprocess",
        type = str,
        default = "tile",
        choices=["tile",],
        help = "Additional preprocessing to apply to control video, tile = blurred video for upscaling",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)