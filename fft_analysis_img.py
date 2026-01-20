import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import DDPMScheduler

path = "./data/dataset/afhq/val/cat/pixabay_cat_004373.jpg"
bands = 8
plot_log_y = True
device = "cuda" if torch.cuda.is_available() else "cpu"

img = Image.open(path).convert("L")
img = np.array(img).astype(np.float32) / 255.0

x0 = torch.from_numpy(img)[None, None].to(device)
x0 = x0 * 2.0 - 1.0

scheduler = DDPMScheduler(num_train_timesteps=1000)
timesteps = list(range(0, 1000, 100))
noise = torch.randn_like(x0)

def fft_power2d(x2d: np.ndarray):
    F = np.fft.fft2(x2d)
    Fs = np.fft.fftshift(F)
    mag = np.abs(Fs)
    power = mag**2
    log_mag = np.log1p(mag)
    return power, log_mag

def radial_band_energy(power_2d: np.ndarray, bands: int = 8):
    H, W = power_2d.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0, rr.max(), bands + 1)
    band_mean = []
    for i in range(bands):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        band_mean.append(power_2d[m].mean() if m.sum() else 0.0)
    return np.array(band_mean)

group_size = 4
groups = [timesteps[i:i+group_size] for i in range(0, len(timesteps), group_size)]

eps = 1e-12

for g_idx, g in enumerate(groups, start=1):
    cache = []
    for t in g:
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        xt = scheduler.add_noise(x0, noise, t_tensor)
        xt_img = ((xt[0, 0].detach().cpu().numpy() + 1.0) / 2.0).clip(0.0, 1.0)
        x_for_fft = xt_img - xt_img.mean()
        pow_2d, log_mag = fft_power2d(x_for_fft)
        band = radial_band_energy(pow_2d, bands)
        cache.append((t, xt_img, log_mag, band))

    all_bands = np.concatenate([c[3] for c in cache], axis=0)
    y_min = max(all_bands.min(), eps)
    y_max = max(all_bands.max(), y_min * 10)

    fig = plt.figure(figsize=(12, 3 * len(g)))
    for i, (t, xt_img, log_mag, band) in enumerate(cache):
        ax1 = plt.subplot(len(g), 3, i * 3 + 1)
        ax1.imshow(xt_img, cmap="gray")
        ax1.set_title(f"t={t} (noisy)")
        ax1.axis("off")

        ax2 = plt.subplot(len(g), 3, i * 3 + 2)
        ax2.imshow(log_mag, cmap="gray")
        ax2.set_title("log(1+|FFT|) (shifted)")
        ax2.axis("off")

        ax3 = plt.subplot(len(g), 3, i * 3 + 3)
        ax3.plot(band, marker="o")
        ax3.set_title("Radial band mean power")
        ax3.set_xlabel("Band (low->high)")
        ax3.set_ylabel("Mean power")
        if plot_log_y:
            ax3.set_yscale("log")
        ax3.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()