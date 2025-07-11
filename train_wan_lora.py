import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    raise Exception("unable to initialize CUDA")

import os
import gc
import math
import random
import numpy as np
import argparse
import json
import datetime
from tqdm import tqdm
from contextlib import contextmanager
from time import perf_counter
from glob import glob

from torchvision.transforms import v2, InterpolationMode
from safetensors.torch import load_file, save_file
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
import bitsandbytes as bnb
# from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from utils.temp_rng import temp_rng
from utils.dataset import CombinedDataset
from utils.dwt_loss import dwt_loss


def make_dir(base, folder):
    new_dir = os.path.join(base, folder)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


@contextmanager
def load_timer(target):
    print(f"loading {target}...")
    start_time = perf_counter()
    yield
    end_time = perf_counter()
    print(f"loaded {target} in {end_time - start_time:0.2f} seconds")


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


def download_model(args):
    from huggingface_hub import snapshot_download
    
    # get text encoder, they're all identical
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/UMT5",
        allow_patterns = "google/*",
    )
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/UMT5",
        allow_patterns = "models_t5_umt5-xxl-enc-bf16.pth",
        max_workers = 1,
    )
    
    # get vae, they're all identical
    snapshot_download(
        repo_type = "model",
        repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
        local_dir = "./models/vae",
        allow_patterns = "Wan2.1_VAE.pth",
        max_workers = 1,
    )
    
    # get clip vision if it's i2v
    if "-I2V-" in args.pretrained_model_name_or_path:
        snapshot_download(
            repo_type = "model",
            repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir = "./models/clipvision",
            allow_patterns = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            max_workers = 1,
        )
        snapshot_download(
            repo_type = "model",
            repo_id = "Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir = "./models/clipvision",
            allow_patterns = "xlm-roberta-large/*",
        )
    
    # get the correct diffusion model for HF identifier
    snapshot_download(
        repo_type = "model",
        repo_id = args.pretrained_model_name_or_path,
        local_dir = "./models/" + args.pretrained_model_name_or_path,
        allow_patterns = ["config.json", "diffusion_pytorch_model*"],
        max_workers = 1,
    )

@torch.inference_mode()
def cache_embeddings(args):
    if os.path.exists("./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"):
        with load_timer("UMT5 text encoder"):
            if os.path.exists("./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"):
                ckpt_dir = "./models/UMT5/models_t5_umt5-xxl-enc-bf16.pth"
                tk_dir = "./models/UMT5/google/umt5-xxl"
            else:
                ckpt_dir = args.pretrained_model_name_or_path
                tk_dir = args.pretrained_model_name_or_path + "/google/umt5-xxl"
            
            umt5_model = T5EncoderModel(
                text_len = 512,
                dtype = torch.bfloat16,
                device = device,
                checkpoint_path = ckpt_dir,
                tokenizer_path = tk_dir,
            )
    else:
        raise Exception("UMT5 model missing, download it first with --download_model")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # it would probably be better to dirwalk and batch the encoding
    # scanning all files at the start can be slow, and encoding one prompt at a time is inefficient
    caption_files = glob(os.path.join(args.dataset, "**", "*.txt" ), recursive=True)
    for file in tqdm(caption_files, desc="encoding captions"):
        embedding_path = os.path.splitext(file)[0] + "_wan.safetensors"
        
        if not os.path.exists(embedding_path):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    caption = f.read()
                    context = umt5_model([caption], umt5_model.device)[0]
                    embedding_dict = {"context": context}
                    save_file(embedding_dict, embedding_path)
            except UnicodeDecodeError as e:
                tqdm.write(f"unable to decode {file}: UnicodeDecodeError: {e}")
    
    del umt5_model
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_output_dir = make_dir(args.output_dir, date_time)
    checkpoint_dir = make_dir(real_output_dir, "checkpoints")
    t_writer = SummaryWriter(log_dir=real_output_dir, flush_secs=60)
    with open(os.path.join(real_output_dir, "command_args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    
    train_dataset = os.path.join(args.dataset, "train")
    if not os.path.exists(train_dataset):
        train_dataset = args.dataset
        print(f"WARNING: train subfolder not found, using root folder {train_dataset} as train dataset")
    
    val_dataset = None
    for subfolder in ["val", "validation", "eval", "test"]:
        subfolder_path = os.path.join(args.dataset, subfolder)
        if os.path.exists(subfolder_path):
            val_dataset = subfolder_path
            break
    
    if val_dataset is None:
        val_dataset = args.dataset
        print(f"WARNING: val/validation/test subfolder not found, using root folder {val_dataset} for stable loss validation")
        print("\033[33mThis will make it impossible to judge overfitting by the validation loss. Using a val split held out from training is highly recommended\033[m")
    
    with load_timer("train dataset"):
        train_dataset = CombinedDataset(
            root_folder = train_dataset,
            token_limit = args.token_limit,
            max_frame_stride = args.max_frame_stride,
            random_frame_length = args.random_frame_length,
            bucket_resolutions = args.base_res,
            load_control = args.load_control,
            control_suffix = args.control_suffix,
        )
    with load_timer("validation dataset"):
        val_dataset = CombinedDataset(
            root_folder = val_dataset,
            token_limit = args.token_limit,
            limit_samples = args.val_samples,
            max_frame_stride = args.max_frame_stride,
            random_frame_length = args.random_frame_length,
            bucket_resolutions = args.base_res,
            load_control = args.load_control,
            control_suffix = args.control_suffix,
        )
    
    def collate_batch(batch):
        pixels = [sample["pixels"][0].movedim(0, 1) for sample in batch] # BFCHW -> FCHW -> CFHW
        context = [sample["embedding_dict"]["context"] for sample in batch]
        control = [sample["control"][0].movedim(0, 1) for sample in batch] if args.load_control else None
        return pixels, context, control
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        collate_fn = collate_batch,
        batch_size = 1,
        num_workers = 0,
        pin_memory = True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle = False,
        collate_fn = collate_batch,
        batch_size = 1,
        num_workers = 0,
        pin_memory = True,
    )
    
    with load_timer("VAE"):
        if os.path.exists("./models/vae/Wan2.1_VAE.pth"):
            ckpt_dir = "./models/vae/Wan2.1_VAE.pth"
        else:
            ckpt_dir = args.pretrained_model_name_or_path
        
        vae = WanVAE(vae_pth=ckpt_dir, dtype=torch.bfloat16, device=device)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    with load_timer("diffusion model"):
        if os.path.exists("./models/" + args.pretrained_model_name_or_path):
            ckpt_dir = "./models/" + args.pretrained_model_name_or_path
        else:
            ckpt_dir = args.pretrained_model_name_or_path
        
        diffusion_model = WanModel.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16).to(device)
        diffusion_model.requires_grad_(False)
        
        if args.gradient_checkpointing:
            diffusion_model.enable_gradient_checkpointing()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    if args.control_lora:
        with torch.no_grad():
            in_cls = diffusion_model.patch_embedding.__class__ # nn.Conv3d
            old_in_dim = diffusion_model.in_dim # 16
            new_in_dim = old_in_dim * 2
            
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
    
    if args.fuse_lora is not None:
        loaded_lora_sd = load_file(args.fuse_lora)
        loaded_lora_sd = convert_sd(loaded_lora_sd)
        diffusion_model.load_lora_adapter(loaded_lora_sd, adapter_name="fuse_lora")
        diffusion_model.fuse_lora(adapter_names="fuse_lora", lora_scale=args.fuse_lora_weight, safe_fusing=True)
        diffusion_model.unload_lora_weights()
        del loaded_lora_sd
    
    lora_params = []
    if args.control_lora:
        lora_params.append("patch_embedding")
    
    lora_targets = args.lora_targets.replace(" ", "").split(",")
    lora_blocks = args.lora_blocks.replace(" ", "").split(",") if args.lora_blocks is not None else None
    for name, param in diffusion_model.named_parameters():
        if name.endswith(".weight"):
            for target in lora_targets:
                if target in name and ".norm_" not in name:
                    if lora_blocks is None:
                        lora_params.append(name.replace(".weight", ""))
                    else:
                        for block in lora_blocks:
                            if f"blocks.{block}" in name:
                                lora_params.append(name.replace(".weight", ""))
    
    # if args.lora_target == "attn":
        # for name, param in diffusion_model.named_parameters():
            # if name.endswith(".weight"):
                # if args.lora_target in name and ".norm_" not in name:
                    # lora_params.append(name.replace(".weight", ""))
    
    # elif args.lora_target == "all-linear":
    
    # else: raise NotImplementedError(f"{args.lora_target}")
    
    lora_config = LoraConfig(
        r = args.lora_rank,
        lora_alpha = args.lora_alpha or args.lora_rank,
        init_lora_weights = "gaussian",
        target_modules = lora_params,
    )
    diffusion_model.add_adapter(lora_config, adapter_name="default")
    
    if args.init_lora is not None:
        loaded_lora_sd = load_file(args.init_lora)
        loaded_lora_sd = convert_sd(loaded_lora_sd)
        outcome = set_peft_model_state_dict(diffusion_model, loaded_lora_sd)
        if len(outcome.unexpected_keys) > 0:
            for key in outcome.unexpected_keys:
                print(f"not loaded: {key}")
            exit()
        else:
            print("init lora loaded successfully, all keys matched")
        del loaded_lora_sd
    
    
    total_parameters = 0
    train_parameters = []
    
    for name, param in diffusion_model.named_parameters():
        if param.requires_grad:
            lr = args.learning_rate
            if "patch_embedding" in name:
                lr *= args.input_lr_scale
            param.data = param.to(torch.float32)
            train_parameters.append((param, lr))
            total_parameters += param.numel()
    
    print(f"total trainable parameters: {total_parameters:,}")
    
    # optimizer = bnb.optim.AdamW8bit(
        # [p for p, _ in train_parameters],
        # lr=args.learning_rate,
    # )
    
    # Instead of having just one optimizer, we will have a dict of optimizers
    # for every parameter so we could reference them in our hook.
    optimizer_dict = {p: bnb.optim.AdamW8bit([p], lr=lr) for p, lr in train_parameters}
    
    # Define our hook, which will call the optimizer step() and zero_grad()
    def optimizer_hook(parameter) -> None:
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()
    
    # Register the hook onto every trainable parameter
    for p, _ in train_parameters:
        # p.register_hook(lambda grad: torch.clamp(grad, -args.clip_grad, args.clip_grad))
        p.register_post_accumulate_grad_hook(optimizer_hook)
    
    context_negative = load_file(args.distill_negative)["context"].to(dtype=torch.bfloat16, device=device)
    
    if args.control_lora and args.control_preprocess == "depth":
        if not os.path.exists("./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth"):
            print("depth model not found, downloading to ./models/Depth-Anything-V2-Small")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_type = "model",
                repo_id = "depth-anything/Depth-Anything-V2-Small",
                local_dir = "./models/Depth-Anything-V2-Small",
                allow_patterns = "*.pth",
            )
        
        from utils.depth_anything_v2.dpt import DepthAnythingV2
        depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        depth_model.load_state_dict(torch.load("./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth", map_location='cpu', weights_only=True))
        depth_model = depth_model.to(device)
        depth_model.requires_grad_(False)
        depth_model.eval()
    
    def preprocess_control(pixels):
        if args.control_preprocess == "tile":
            control = pixels.movedim(0, 1).unsqueeze(0) # CFHW -> BFCHW
            height, width = control.shape[-2:]
            
            blur = v2.Compose([
                v2.Resize(size=(height // 4, width // 4)),
                v2.Resize(size=(height, width)),
                # v2.GaussianBlur(kernel_size=15, sigma=random.uniform(3, 6)),
                v2.GaussianBlur(kernel_size=15, sigma=3),
            ])
            
            control = torch.clamp(torch.nan_to_num(blur(control)), min=-1, max=1)
            control = control[0].movedim(0, 1) # BFCHW -> CFHW
        
        elif args.control_preprocess == "depth":
            depth_frames = []
            for i in range(pixels.shape[1]):
                d_input = pixels[:, i].movedim(0, -1).cpu().float().numpy() * 0.5 + 0.5 # CFHW -> CHW -> HWC
                depth = depth_model.infer_image(d_input)
                depth = (depth - depth.min()) / (depth.max() - depth.min()) # normalized to near=1, far=0
                depth_frames.append(depth * 2 - 1)
            
            depth_frames = torch.stack(depth_frames).unsqueeze(0).repeat(3, 1, 1, 1) # HW -> FHW -> CFHW
            control = depth_frames.to(dtype=torch.bfloat16, device=device)
        
        else:
            raise NotImplementedError(f"{args.control_preprocess}")
        
        return control
    
    def prepare_conditions(batch, timestep=None):
        pixels, context, control = batch
        
        pixels  = [p.to(dtype=torch.bfloat16, device=device) for p in pixels]
        context = [c.to(dtype=torch.bfloat16, device=device) for c in context]
        
        latents = vae.encode(pixels)
        noise = [torch.randn_like(l) for l in latents]
        
        if timestep is not None:
            timesteps = torch.ones(len(latents)).to(device) * timestep
            sigmas = timesteps.float() / 1000
        else:
            sigmas = torch.rand(len(latents)).to(device)
            sigmas = (args.shift * sigmas) / (1 + (args.shift - 1) * sigmas)
            timesteps = torch.round(sigmas * 1000).long()
            sigmas = timesteps.float() / 1000
        
        if args.control_lora:
            if control is not None:
                control = [p.to(dtype=torch.bfloat16, device=device) for p in control]
            else:
                control = [preprocess_control(p) for p in pixels]
            
            if args.control_i2v:
                for i, p in enumerate(pixels):
                    if p.shape[1] > 1:
                        control[i][:, 0] = p[:, 0]
            
            control_latents = vae.encode(control)
            
            if args.control_inject_noise > 0:
                for i in range(len(control_latents)):
                    inject_strength = torch.rand(1).item() * args.control_inject_noise
                    control_latents[i] += torch.randn_like(control_latents[i]) * inject_strength
        
        target = []
        noisy_model_input = []
        for i in range(len(latents)):
            if args.use_dwt_loss:
                target.append(latents[i])
            else:
                target.append(noise[i] - latents[i])
            
            noisy = noise[i] * sigmas[i] + latents[i] * (1 - sigmas[i])
            
            if args.control_lora:
                noisy = torch.cat([noisy, control_latents[i]], dim=0) # CFHW, so channel dim is 0
            
            noisy_model_input.append(noisy.to(torch.bfloat16))
        
        return {
            "sigmas": sigmas,
            "target": target,
            "context": context,
            "timesteps": timesteps,
            "noisy_model_input": noisy_model_input,
        }
    
    def predict_loss(conditions, log_cfg_loss=False):
        target = torch.stack(conditions["target"])
        sigmas = conditions["sigmas"]
        noisy_model_input = conditions["noisy_model_input"]
        c, f, h, w = noisy_model_input[0].shape
        seq_len = math.ceil((h / 2) * (w / 2) * f)
        
        pred = torch.stack(
            diffusion_model(
                x = noisy_model_input,
                t = conditions["timesteps"],
                context = conditions["context"],
                seq_len = seq_len,
            )
        )
        
        if args.distill_cfg > 0:
            with torch.no_grad():
                diffusion_model.set_adapters(adapter_names="default", weights=0.0)
                
                base_pred_cond = torch.stack(
                    diffusion_model(
                        x = noisy_model_input,
                        t = conditions["timesteps"],
                        context = conditions["context"],
                        seq_len = seq_len,
                    )
                )
                
                base_pred_uncond = torch.stack(
                    diffusion_model(
                        x = noisy_model_input,
                        t = conditions["timesteps"],
                        context = [context_negative] * len(noisy_model_input),
                        seq_len = seq_len,
                    )
                )
                
                diffusion_model.set_adapters(adapter_names="default", weights=1.0)
                
                if log_cfg_loss:
                    cfg_loss = F.mse_loss(pred, base_pred_cond)
                    t_writer.add_scalar("loss/cfg", cfg_loss.item(), global_step)
            
                target += args.distill_cfg * (base_pred_cond - base_pred_uncond)
        
        assert not torch.isnan(pred).any()
        
        if args.use_dwt_loss:
            # need to get clean prediction instead of velocity
            pred = torch.stack([p * (-sigmas[i]) + noisy_model_input[i] for i, p in enumerate(pred)])
            
            pred = torch.cat([p.movedim(1, 0) for p in pred.float()], dim=0) # (B)CFHW -> FCHW
            target = torch.cat([t.movedim(1, 0) for t in target.float()], dim=0)
            return dwt_loss(pred, target)
        
        return F.mse_loss(pred.float(), target.float())
    
    gc.collect()
    torch.cuda.empty_cache()
    diffusion_model.train()
    
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps))
    while global_step < args.max_train_steps:
        for step, batch in enumerate(train_dataloader):
            start_step = perf_counter()
            with torch.inference_mode():
                conditions = prepare_conditions(batch)
            
            gc.collect()
            torch.cuda.empty_cache()
            
            loss = predict_loss(conditions, log_cfg_loss=True)
            t_writer.add_scalar("loss/train", loss.detach().item(), global_step)
            loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()
            
            gc.collect()
            torch.cuda.empty_cache()
            
            t_writer.add_scalar("debug/step_time", perf_counter() - start_step, global_step)
            progress_bar.update(1)
            global_step += 1
            
            if global_step == 1 or global_step % args.val_steps == 0:
                with torch.inference_mode(), temp_rng(args.val_seed or args.seed):
                    val_loss = 0.0
                    for step, batch in enumerate(tqdm(val_dataloader, desc="validation", leave=False)):
                        conditions = prepare_conditions(batch, timestep=args.val_timestep)
                        loss = predict_loss(conditions)
                        val_loss += loss.detach().item()
                    t_writer.add_scalar("loss/validation", val_loss / len(val_dataloader), global_step)
                
                gc.collect()
                torch.cuda.empty_cache()
                progress_bar.unpause()
            
            if global_step >= args.max_train_steps or global_step % args.checkpointing_steps == 0:
                save_file(
                    get_peft_model_state_dict(diffusion_model),
                    os.path.join(checkpoint_dir, f"wan-lora-{global_step:08}.safetensors"),
                )
                gc.collect()
                torch.cuda.empty_cache()
            
            if global_step >= args.max_train_steps:
                break


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Wan2.1 lora training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--download_model",
        action = "store_true",
        help = "auto download all necessary models to ./models if missing",
    )
    parser.add_argument(
        "--cache_embeddings",
        action = "store_true",
        help = "preprocess dataset to encode captions",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B",
        help="Path to pretrained model or model identifier from huggingface",
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "./outputs",
        help = "Output directory for training results",
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = None,
        help = "Path to dataset directory with train and val subdirectories",
    )
    parser.add_argument(
        "--val_samples",
        type = int,
        default = 4,
        help = "Maximum number of samples to use for validation loss",
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Seed for reproducible training",
    )
    parser.add_argument(
        "--val_seed",
        type = int,
        default = None,
        help = "Optional separate seed for validation, to have consistent validation rng when changing training seed",
    )
    parser.add_argument(
        "--fuse_lora",
        type = str,
        default = None,
        help = "LoRA checkpoint to fuse into model, before adding a new trainable LoRA",
    )
    parser.add_argument(
        "--fuse_lora_weight",
        type = float,
        default = 1.0,
        help = "strength to merge --fuse_lora into the base model",
    )
    parser.add_argument(
        "--init_lora",
        type = str,
        default = None,
        help = "LoRA checkpoint to load instead of random init, must be the same rank and target layers",
    )
    # parser.add_argument(
        # "--lora_target",
        # type = str,
        # default = "attn",
        # choices=["attn", "all-linear"],
        # help = "layers to target with LoRA, default is attention only",
    # )
    parser.add_argument(
        "--lora_targets",
        type = str,
        default = "self_attn, cross_attn",
        help = "layers to target with LoRA, default is self attention only",
    )
    parser.add_argument(
        "--lora_blocks",
        type = str,
        default = None,
        help = "blocks to target with LoRA, default is all blocks, otherwise comma separated string of block numbers",
    )
    parser.add_argument(
        "--control_lora",
        action = "store_true",
        help = "Train lora as control lora (extra input channels)",
    )
    parser.add_argument(
        "--control_i2v",
        action = "store_true",
        help = "Train control lora as i2v instead of t2v, replacing first control frame with image",
    )
    parser.add_argument(
        "--load_control",
        action = "store_true",
        help = "Load control files from disk instead of calculating on the fly. Bypasses --control_preprocess",
    )
    parser.add_argument(
        "--control_suffix",
        type = str,
        default = "_control",
        help = "suffix to append to media file name (ignoring extension) to get the control file",
    )
    parser.add_argument(
        "--control_preprocess",
        type = str,
        default = "none",
        choices=["none", "tile", "depth"],
        help = "Preprocess to apply if not loading a control video. Ignored if using --load_control",
    )
    parser.add_argument(
        "--control_inject_noise",
        type = float,
        default = 0.0,
        help = "Add noise to the control latents, at a random strength up to this amount",
    )
    parser.add_argument(
        "--lora_rank",
        type = int,
        default = 16,
        help = "The dimension of the LoRA update matrices",
    )
    parser.add_argument(
        "--lora_alpha",
        type = int,
        default = None,
        help = "The alpha value for LoRA, defaults to alpha=rank. Note: changing alpha will affect the learning rate, and if alpha=rank then changing rank will also affect learning rate",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action = "store_true",
        help = "use gradient checkpointing to reduce memory usage at the cost of speed",
    )
    # parser.add_argument(
        # "--clip_grad",
        # type = float,
        # default = 100.0,
        # help = "Clip gradients at +- this value (at each parameter via hook)",
    # )
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 1e-5,
        help = "Base learning rate",
    )
    parser.add_argument(
        "--input_lr_scale",
        type = float,
        default = 1,
        help = "Multiplier to learning rate for the input patch_embedding layer if training control lora",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default = 3.0,
        help = "Noise schedule shift for training (shift > 1 will spend more effort on early timesteps/high noise",
    )
    parser.add_argument(
        "--use_dwt_loss",
        action = "store_true",
        help = "Use discrete wavelet transform loss from https://arxiv.org/abs/2503.18352",
    )
    parser.add_argument(
        "--distill_cfg",
        type = float,
        default = 0.0,
        help = "CFG scale to use for distillation, 0=disabled",
    )
    parser.add_argument(
        "--distill_negative",
        type = str,
        default = "./embeddings/default_video_negative_wan.safetensors",
        help = "Precalculated embedding file to use as negative prompt for CFG distillation",
    )
    parser.add_argument(
        "--base_res",
        type = int,
        nargs = '+',
        default = 624,
        choices=[624, 960, 1024],
        help = "Base resolution buckets, resized to equal area based on aspect ratio. Supports multiple buckets.",
    )
    parser.add_argument(
        "--token_limit",
        type = int,
        default = 10_000,
        help = "Combined resolution/frame limit based on transformer patch sequence length: (width // 16) * (height // 16) * ((frames - 1) // 4 + 1)",
    )
    parser.add_argument(
        "--max_frame_stride",
        type = int,
        default = 2,
        help = "1: use native framerate only. Higher values allow randomly choosing lower framerates (skipping frames to speed up the video)",
    )
    parser.add_argument(
        "--random_frame_length",
        action = "store_true",
        help = "Randomize frame length instead of always trying to fill the max length",
    )
    parser.add_argument(
        "--val_steps",
        type = int,
        default = 100,
        help = "Validate after every n steps",
    )
    parser.add_argument(
        "--val_timestep",
        type = int,
        default = None,
        help = "Override timestep for validation samples",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type = int,
        default = 100,
        help = "Save a checkpoint of the training state every X steps",
    )
    parser.add_argument(
        "--max_train_steps",
        type = int,
        default = 1_000,
        help = "Total number of training steps",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.download_model:
        download_model(args)
        exit()
    
    if args.dataset is not None:
        if args.cache_embeddings:
            cache_embeddings(args)
            exit()
        
        main(args)
    else:
        raise Exception("--dataset is required but not provided")