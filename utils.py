import os 
import torch
from tqdm import tqdm
from torchvision.utils import make_grid, save_image


def show_prediction(valloader, model, device="cuda", sample_dir="checkpoints/val_samples"):
    model.eval()
    os.makedirs(sample_dir, exist_ok=True)

    real_dir = os.path.join(sample_dir,"real")
    gen_dir  = os.path.join(sample_dir,"gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)

    save_idx = 0

    with torch.no_grad():
        for img, cls in tqdm(valloader):

            img = img.to(device)
        
            pred, mu, sigma = model(img) 

            img = (img + 1) / 2
            pred = (pred + 1) / 2
            originals = img.detach().cpu()
            recon = pred.detach().cpu()

            bsz = originals.size(0)
            for i in range(bsz):
                save_image(originals[i], os.path.join(real_dir, f"{save_idx:06d}.png"))
                save_image(recon[i],     os.path.join(gen_dir,  f"{save_idx:06d}.png"))
                save_idx += 1
                