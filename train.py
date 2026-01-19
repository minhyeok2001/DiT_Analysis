
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import wandb
import random

import data.get_data 
import data.dataloader 
from tqdm import tqdm
from model import DiT
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from torchvision.utils import make_grid, save_image
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from torchmetrics.image.fid import FrechetInceptionDistance


def test():
    mode_list = ["adaLN-Zero","Cross-Attention","In-Context Conditioning"]

    model = DiT(mode=mode_list[0],num_blocks=12,num_head=4)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"=============================================")
    print(f"전체 파라미터 (VAE/CLIP 포함): {total_params:,} 개")
    print(f"학습 가능 파라미터 (DiT 본체): {trainable_params:,} 개")
    print(f"---------------------------------------------")
    print(f"모델 사이즈 (Trainable): {trainable_params / 1e6:.2f} M (Million)")
    print(f"=============================================")

    sample = torch.randn(3,3,128,128)
    timestep = torch.randint(0,1,(3,))
    label = ["cat","cat","wild"]
    print(model(sample,label,timestep).shape)

def calculate_fid(valloader, noise_scheduler, model, vae, text_encoder, tokenizer, device, out_dir="checkpoints/fid_samples", cfg_weight=2.5, num_inference_steps=250):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    real_dir = os.path.join(out_dir, "real")
    gen_dir = os.path.join(out_dir, "gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    with torch.no_grad():
        for idx, (img, cls) in enumerate(tqdm(valloader, desc="FID")):
            img = img.to(device)
            B = img.shape[0]
            
            real_imgs_uint8 = (img * 255).clamp(0, 255).to(dtype=torch.uint8)
            fid.update(real_imgs_uint8, real=True)
            
            for j in range(B):
                save_image(img[j], os.path.join(real_dir, f"{idx*B+j:05d}.png"))

            prompts = [f"A photo of {l}" for l in cls]
            text_input = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
            text_embeddings = text_encoder(**text_input).pooler_output
            
            uncond_input = tokenizer([""] * B, padding="max_length", truncation=True, return_tensors="pt").to(device)
            uncond_embeddings = text_encoder(**uncond_input).pooler_output

            latents = torch.randn(B, 4, 16, 16).to(device)

            for t in noise_scheduler.timesteps:
                latent_model_input = noise_scheduler.scale_model_input(latents, t)
                
                noise_pred_uncond = model(x=latent_model_input, label=uncond_embeddings, timestep=t)
                noise_pred_text = model(x=latent_model_input, label=text_embeddings, timestep=t)
                
                noise_pred = noise_pred_uncond + cfg_weight * (noise_pred_text - noise_pred_uncond)
                
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            decoded = vae.decode(latents / 0.13025).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            
            gen_imgs_uint8 = (decoded * 255).to(dtype=torch.uint8)
            fid.update(gen_imgs_uint8, real=False)

            for j in range(B):
                save_image(decoded[j], os.path.join(gen_dir, f"{idx*B+j:05d}.png"))

    fid_score = fid.compute().item()
    print(f"FID Score: {fid_score:.4f}")
    
    return fid_score


def run(args):
    
    ## 이렇게 하면 안되지만, colab 이용해야하므로 ..,,
    wandb.login(key="08198b7be027ddffa5241b9acf2f45cd4d42e993")
    
    device = "cuda"
    epoch = args.epoch 
    lr = args.lr 
    batch_size = args.batch_size
    cfg_weight = args.cfg_weight
    mode = args.mode
    num_blocks = args.num_blocks
    num_head = args.num_head
    cfg_dropout = args.cfg_dropout
    num_inference_steps = args.num_inference_steps
    
    wandb.init(
        project="DiT Analysis",
        config={
            "epochs": epoch,
            "lr": lr,
            "batch_size": batch_size,
            "cfg_weight" : cfg_weight,
            "mode": mode,
            "num_blocks" : num_blocks,
            "num_head" : num_head,
            "cfg_dropout" : cfg_dropout,
            "num_inference_step" : num_inference_steps
        }
    )
    
    dataset = data.dataloader.CustomDataset()
    trainloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,collate_fn=data.dataloader.collate_ft,num_workers=6,shuffle=True)
    
    valset = data.dataloader.CustomDataset(test=True)
    valloader = torch.utils.data.DataLoader(valset,batch_size=batch_size,num_workers=6,shuffle=False)
    
    ## 2. Model definition & setting stuffs..
    
    #mode_list = ["adaLN-Zero","Cross-Attention","In-Context Conditioning"]
    
    model = DiT(mode=args.mode,num_blocks=12,num_head=4).to(device)

    ## 스케줄러 라이브러리
    noise_scheduler = DDPMScheduler()
        
    wandb.watch(model, log="all")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"=============================================")
    print(f"학습 가능 파라미터 : {trainable_params:,} 개")
    print(f"---------------------------------------------")
    print(f"모델 사이즈 : {trainable_params / 1e6:.2f} M (Million)")
    print(f"=============================================")
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)    
    
    ema = EMAModel(model.parameters(), decay=0.9999) 
    ema.to(device)
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device) ## 결과 : torch.Size([1, 4, 16, 16])
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M").to(device) ## 결과 : torch.Size([1, 77, 256])    
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    sample_dir = "checkpoints/val_samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    def show_prediction(step):
        model.eval()

        img, cls = next(iter(valloader))
        B = img.shape[0]
        cls = list(cls)
        prompts = [f"A photo of {l}" for l in cls]

        with torch.no_grad():
            text_input = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
            text_embeddings = text_encoder(**text_input).pooler_output

            uncond_input = tokenizer([""] * B, padding="max_length", truncation=True, return_tensors="pt").to(device)
            uncond_embeddings = text_encoder(**uncond_input).pooler_output

            latents = torch.randn(B, 4, 16, 16).to(device)
            
            noise_scheduler.set_timesteps(num_inference_steps, device=device)
            
            num_snapshots = 10
            snap_idxs = set(torch.linspace(0, num_inference_steps - 1, steps=num_snapshots).round().long().tolist())
            snapshots = []

            for i, t in enumerate(noise_scheduler.timesteps):
                t = t.to(device)
                latent_model_input = noise_scheduler.scale_model_input(latents, t)

                noise_pred_uncond = model(x=latent_model_input, label=uncond_embeddings, timestep=t)
                noise_pred_text = model(x=latent_model_input, label=text_embeddings, timestep=t)

                noise_pred = noise_pred_uncond + cfg_weight * (noise_pred_text - noise_pred_uncond)

                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                if i in snap_idxs or i == num_inference_steps - 1:
                    decoded_snap = vae.decode(latents / 0.13025).sample
                    decoded_snap = (decoded_snap / 2 + 0.5).clamp(0, 1)
                    limit = min(8, B)
                    snapshots.append(decoded_snap[:limit].cpu())

            timeline_strip = torch.cat(snapshots, dim=-1)
            grid = make_grid(timeline_strip, nrow=1, padding=2, normalize=False)
            
            save_path = os.path.join(sample_dir, f"iter_{step}_timeline.png")
            save_image(grid, save_path)

        return save_path
    
    for i in range(epoch) :
        
        model.train()
        running_loss = 0.0
        total_len = len(trainloader)
        for img, cls in tqdm(trainloader):
            optimizer.zero_grad()
            
            #print("img shape", img.shape)
            img = img.to(device)
            cls = list(cls)
            img = img * 2 - 1
            
            for idx in range(len(cls)):
                if random.random() < cfg_dropout:
                    cls[idx] = "" 

            prompts = [f"A photo of {l}" if l != "" else "" for l in cls]
            
            with torch.no_grad():
                label_tokens = tokenizer(
                    prompts, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).to(device) 
                
                label_emb = text_encoder(**label_tokens).pooler_output
                latents = vae.encode(img).latent_dist.sample() * 0.13025
            
            #print(prompts)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (img.shape[0],), device=device)
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, t)
        
            noise_pred = model(x=noisy_latents, label=label_emb, timestep=t)
            loss = F.mse_loss(noise_pred, noise) 

            loss.backward()
            optimizer.step()
            
            ema.step(model.parameters())
            
            running_loss += loss.item()

        avg_train_loss = running_loss / total_len
        print(f"Epoch [{i+1}/{epoch}] | Train Loss: {avg_train_loss:.6f}")
        
        val_loss = 0.0
        val_batches = len(valloader)
        model.eval()
        
        with torch.no_grad():
            for idx,(img, cls) in tqdm(enumerate(valloader)):
                if idx == 1000 : 
                    break
                img = img.to(device)
                img = img * 2 - 1

                prompts = [f"A photo of {l}" if l != "" else "" for l in cls]
                
                label_tokens = tokenizer(
                    prompts, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).to(device)
                
                label_emb = text_encoder(**label_tokens).pooler_output

                latents = vae.encode(img).latent_dist.sample() * 0.13025
                    
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (img.shape[0],), device=device)
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, t)
            
                noise_pred = model(x=noisy_latents, label=label_emb, timestep=t)
                loss = F.mse_loss(noise_pred, noise) 
                val_loss += loss.item()
            
        avg_val_loss = val_loss / val_batches
        print(f"Epoch [{i+1}/{epoch}] | Val Loss: {avg_val_loss:.6f}")
        
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        img_path = show_prediction(step=i)
        ema.restore(model.parameters())
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": i + 1,
            "sample":wandb.Image(img_path)
        })

    torch.save(ema.state_dict(), "dit_model_final.pth")
    

    fid_score = calculate_fid(valloader, noise_scheduler, model, vae, text_encoder, tokenizer, device, out_dir="checkpoints/fid_samples", cfg_weight=cfg_weight, num_inference_steps=num_inference_steps)
    wandb.log({"FID_Score": fid_score})
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--cfg_weight", type=float, default=2.5)
    parser.add_argument("--mode", type=str, default="adaLN-Zero", choices=["adaLN-Zero", "Cross-Attention", "In-Context Conditioning"])
    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--num_head", type=int, default=4)
    parser.add_argument("--cfg_dropout", type=float, default=0.3)
    parser.add_argument("--num_inference_steps", type=int, default=250)

    args = parser.parse_args()
    
    run(args)