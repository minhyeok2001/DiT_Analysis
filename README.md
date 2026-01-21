# Diffusion Transformer Analysis

DiT 메인 그림 넣기

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

Training was conducted using RTX5090 hosted by [vast.ai](https://vast.ai/)

All codes were developed and tested on a Mac beforehand.

---

## 1. Reproduction & Benchmarks

**1. Full Implementation of DiT Variants**

I implemented the DiT architecture and training pipeline from scratch

To verify the observations of the original paper, I implemented and experimented with all three major conditioning mechanisms:

  * In-Context Conditioning: Appending vector embeddings to the input sequence.
  
  * Cross-Attention: Using vector embeddings as keys/values in multi-head attention.
  
  * adaLN-Zero: Modulating layer normalization parameters directly via embeddings.


그림 넣기
  
Consistent with the original paper, **adaLN-Zero** demonstrated the most efficient convergence and superior generation quality.


---

**2. Efficient Evaluation Strategy**

The original paper utilized the DDPM sampler with 250 steps for evaluation.

Due to computational constraints and the need for efficient experimentation, 

I utilized the **Euler Discrete Solver** (20 steps) for sampling and calculating FID scores.

This setup allowed for rapid benchmarking while maintaining relative performance consistency across different models.

---

## 2. Proposed Method: Freq-Gate-adaLN

### Motivation: Aligning Conditioning with Denoising Dynamics

I observed that the diffusion process naturally follows a **Coarse-to-Fine** progression:

사진 첨부

* **Early Steps ($t \approx T$):** The model focuses on reconstructing low-frequency components (Global structure, Layout).
* **Late Steps ($t \approx 0$):** The model shifts focus to high-frequency details (Fine textures, Edges).

FFT analysis quantitatively validates this spectral evolution, confirming a distinct shift in the frequency domain as the denoising steps progress.

사진

**This observation raises a critical question:**
> *"If the image restoration process evolves from structure to detail, shouldn't the conditioning guidance evolve as well?"*

We argue that providing a static class condition vector is suboptimal. 

Instead, the conditioning mechanism should **explicitly reflect this frequency evolution**,

prioritizing structural guidance in the early stages and textural guidance in the later stages.

For instance, when generating a **"smiling furry cat,"** 

the model needs to establish the fundamental shape of a **"cat"** first (in early denoising stage).

Fine-grained attributes like **"fur texture"** or a **"smiling expression"** become meaningful only after this structural foundation is established.


<p align="center">
  사진
  <i> Architecture of Freq-Gate-adaLN </i>
</p>

To incorporate this inductive bias, I proposed **Freq-Gate-adaLN**.

- **Time-Dependent Gating**: A gating network, conditioned on $t$, dynamically interpolates between `Cond_Low` (Structure) and `Cond_High` (Detail) adapters.

- **FFT loss**: Applying **Spectral Auxiliary Loss** during training to ensure the faithful reconstruction of **high-frequency details**.

---

### Results

We compared our proposed method against the standard baselines. 

To match the number of parameters, we added two additional blocks to the baseline model.

* **Baseline:** Standard `adaLN-Zero` implementation (with 14 DiT blocks)
* **Ours:** `Freq-Gate-adaLN` with Spectral Loss.

| Method | Conditioning | FID Score (↓) | Train Loss (MSE) | Params (M) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | adaLN-Zero | 12.45 | 0.0890 | 68.58 M |
| **Ours** | Freq-Gate-adaLN | **11.80** | **0.0875** | 68.32 M |

> *Note: FID scores were calculated using Euler Scheduler (50 steps).*

The results show that explicitly addressing the frequency dynamics leads to better convergence and detail generation.

You can check the full experiment logs on W&B — [click here](https://wandb.ai/mhroh01-ajou-university/DiT%20Analysis).

---

### Conclusion

Through this project, we verified that the standard **adaLN-Zero** is indeed a robust baseline. However, by introducing **Freq-Gate-adaLN**, we observed meaningful improvements in model performance.

**Key Findings:**
1. **Frequency Dynamics:** The denoising process strongly correlates with frequency restoration order (Low $\to$ High).
2. **Dynamic Gating:** Allowing the model to "switch gears" between low-freq and high-freq modes improves generation quality without significant computational overhead.

We confirmed that our proposed module can be applied without harming the baseline performance and provides a promising direction for parameter-efficient fine-tuning in generative models.

---

*This is a personal research project implemented for educational and experimental purposes.*
