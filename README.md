# Diffusion Transformer Analysis

<p align="center">
   <img width="935" height="423" alt="스크린샷 2026-01-21 오후 2 57 01" src="https://github.com/user-attachments/assets/475ea686-3941-42c0-8e3d-7019dcce950c" /><br>
   <i>Image from "Scalable Diffusion Models with Transformers"</i>
</p>

This repository implements the Diffusion Transformer (DiT) from scratch ,

proposes an extended architecture focused on **Frequency-Aware Conditioning**. 

This project consists of **TWO PARTS**.

1. **Reproduction:** Complete implementation and comparison of standard DiT conditioning mechanisms.

2. **Proposed Method:** Applying my own idea — Freq-Gate-adaLN — to enhance detail restoration based on the frequency dynamics of diffusion.



## 🐾 Dataset & Environment 
AFHQ (Animal Faces-HQ) consists of 16,130 high-quality images with a resolution of 512×512.

Due to environmental constraints, we resized the images to 128×128 for training.

Since this is a generative modeling task rather than a classification one, only minimal preprocessing was applied.

<p align="center">
   <img width="2354" height="337" alt="image" src="https://github.com/user-attachments/assets/287be022-c4ba-4157-b4cd-24d0de5691ca" /><br>
   <i>9000 samples for Train, 5100 samples for evaluating FID score</i>
</p>

Training was conducted using RTX5090 32GB hosted by [vast.ai](https://vast.ai/)

All codes were developed and tested on a Mac beforehand.

---

## 1. Reproduction & Benchmarks

### Full Implementation of DiT Variants

I implemented the DiT architecture and training pipeline from scratch

To verify the observations of the original paper, I implemented and experimented with all three major conditioning mechanisms:

  * In-Context Conditioning: Appending vector embeddings to the input sequence.
  
  * Cross-Attention: Using vector embeddings as keys/values in multi-head attention.
  
  * adaLN-Zero: Modulating layer normalization parameters directly via embeddings.



<p align="center">
   <img width="1277" height="518" alt="스크린샷 2026-01-21 오후 3 01 13" src="https://github.com/user-attachments/assets/9dcaaa71-fd17-43e8-be70-d34483e6908e" /><br>
   <i>Loss from three different methods</i>
</p>

| Conditioning Mechanism  | FID Score (↓) |
| :--- | :--- |
| **Cross-Attention** | 95.97 |
| **In-Context** | 64.80 |
| **adaLN-Zero** | **60.11** |
> *Note: FID scores were calculated using Euler Scheduler (20 steps) with CFG.*

Consistent with the original paper, **adaLN-Zero** demonstrated the most efficient convergence and superior generation quality.

---

## 2. Proposed Method: Freq-Gate-adaLN

### Motivation: Aligning Conditioning with Denoising Dynamics

I observed that the diffusion process naturally follows a **Coarse-to-Fine** progression:

<p align="center">
   <img width="626" height="107" alt="스크린샷 2026-01-21 오후 3 07 12" src="https://github.com/user-attachments/assets/67dd33aa-ea67-4197-8bd8-28e3252d8791" /><br>
   <i>Denoising process. We can observe that the model construct Coarse -> Fine details</i>
</p>

* **Early Steps ($t \approx T$):** The model focuses on reconstructing low-frequency components (Global structure, Layout).
* **Late Steps ($t \approx 0$):** The model shifts focus to high-frequency details (Fine textures, Edges).

FFT analysis quantitatively validates this spectral evolution, confirming a distinct shift in the frequency domain as the denoising steps progress.


<p align="center">
   <img width="3694" height="2715" alt="image" src="https://github.com/user-attachments/assets/25dacea3-61e2-4666-8e0a-e679e8f5f6bf" /><br>
   <i>FFT analysis</i>
</p>

The restoration of low-frequency content (structure) dominates the early denoising steps, 

whereas the suppression of high-frequency magnitudes (noise removal and detail enhancement) occurs primarily in the final stages. 

This confirms that the model synthesizes images by progressively resolving features from coarse to fine scales


**This observation raises a critical question:**
> *"If the image restoration process evolves from structure to detail, shouldn't the conditioning guidance evolve as well?"*

I argue that providing a static class condition vector is suboptimal. 

Instead, the conditioning mechanism should **explicitly reflect this frequency evolution**,

prioritizing structural guidance in the early stages and textural guidance in the later stages.

For instance, when generating a **"smiling furry cat,"** 

the model needs to establish the fundamental shape of a **"cat"** first (in early denoising stage).

Fine-grained attributes like **"fur texture"** or a **"smiling expression"** become meaningful only after this structural foundation is established.


<p align="center">
<img width="550" height="500" alt="스크린샷 2026-01-21 오후 4 56 27" src="https://github.com/user-attachments/assets/10601d9e-7e44-4547-9c4c-7b2f5fc541b1" /><br>
  <i> Architecture of Freq-Gate-adaLN. Close shot of DiT block </i>
</p>

To incorporate this inductive bias, I proposed **Freq-Gate-adaLN**.

- **Time-Dependent Gating**: A gating network, conditioned on $t$, dynamically interpolates between `Cond_Low` (Structure) and `Cond_High` (Detail) adapters.

- **FFT loss**: Applying **Spectral Auxiliary Loss** during training to ensure the faithful reconstruction of **high-frequency details**.

---

### Results

I compared our proposed method against the standard baselines. 

To match the number of parameters, I added two additional blocks to the baseline model.

* **Baseline:** Standard `adaLN-Zero` implementation (with 14 DiT blocks)
* **Ours:** `Freq-Gate-adaLN` with Spectral Loss.

<p align="center">
   <img width="650" height="533" alt="스크린샷 2026-01-21 오후 8 43 47" src="https://github.com/user-attachments/assets/83e9301f-4a3d-4453-ac77-061da0664ed5" /><br>
   <i>Loss of baseline and My own Freq-Gate-adaLN</i>
</p>

| Method | Conditioning | FID Score (↓) | Val Loss (MSE) | Params (M) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | adaLN-Zero | **54.26617** | **0.15092** | 68.58 M |
| **Ours** | Freq-Gate-adaLN | 58.44975 | 0.152 | 68.32 M |

> *Note: FID scores were calculated using Euler Scheduler (20 steps) with CFG.*

Unfortunately, The results show that there is no surprising gap and even worse than the baseline.

You can check the full experiment logs on W&B — [click here](https://wandb.ai/mhroh01-ajou-university/DiT%20Analysis).

---


### Conclusion

Despite the results showing comparable performance to the baseline rather than a significant leap, 

this project offers meaningful interpretations regarding the internal mechanisms of diffusion models.

While **Freq-Gate-adaLN** performed on par with the standard **adaLN-Zero**, 

it successfully validated the underlying frequency dynamics of the denoising process.

**Key Findings**

* **Frequency Dynamics:** Experiments verified that the denoising process inherently follows a **Low → High restoration order**
  
* **Interpretable Mechanism:** The gating module autonomously learned to prioritize low frequencies in early steps and high frequencies in later steps,
   empirically proving the **coarse-to-fine** generation behavior.

In conclusion, my approach maintains baseline quality while offering a more **interpretable architecture**, 

confirming that explicit frequency modeling is a valid direction for future research.

