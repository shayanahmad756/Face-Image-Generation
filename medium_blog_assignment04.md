# From Theory to Faces: Building a DDPM-Style Image Generator with PyTorch on CelebA-HQ

Generative AI often looks simple on the surface: give a prompt, get an image. But the real learning happens under the hood, where data preprocessing, model design, training stability, and evaluation all connect. For Assignment 04, I built a DDPM-style diffusion pipeline in PyTorch to move from theory to practical implementation.

This project uses CelebA-HQ face images and focuses on end-to-end understanding: how noise is added, how denoising is learned, how image quality is evaluated, and how results are presented through a Gradio demo.

**Screenshot 1 (Project Header / Notebook Setup):**  
Add a screenshot of your notebook title and environment setup cell (showing GPU detection).

## Why diffusion models?

Diffusion models generate images through a gradual process. In the forward phase, clean images are corrupted with Gaussian noise over many timesteps. In the reverse phase, a neural network learns to remove that noise step by step.

This iterative design is useful for students because it exposes key generative AI ideas clearly: probabilistic transitions, time-dependent learning, reconstruction objectives, and hyperparameter sensitivity.

## Project objective

The assignment goal was to implement and train a DDPM-style image generator on CelebA-HQ, evaluate outputs, and make the results demonstrable.

Tools used:

- PyTorch for model and training
- torchvision for dataset loading/transforms
- piqa for SSIM and PSNR
- matplotlib for visual sample inspection
- Gradio for interactive demonstration

The notebook runs on GPU when available, with assignment-appropriate settings for image size, batch size, timesteps, and epochs.

**Screenshot 2 (Libraries + Hyperparameters):**  
Add a screenshot of the import cell and parameter definitions (`IMG_SIZE`, `BATCH_SIZE`, `TIMESTEPS`, `EPOCHS`, `LR`).

## Dataset and preprocessing pipeline

CelebA-HQ images were loaded through an `ImageFolder` pipeline. To keep training consistent and efficient, images were resized to 128 by 128 and normalized to match model input expectations.

Key preprocessing steps:

- `Resize((128, 128))`
- `ToTensor()`
- `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- shuffled DataLoader batches

A practical detail was adding a `DummyDataset` fallback so the workflow stays testable if the dataset path is missing.

**Screenshot 3 (Dataset Load Confirmation):**  
Add a screenshot showing successful dataset loading output (image count and dataset path).

**Screenshot 4 (Batch of Real Images):**  
Add your plotted grid of real CelebA-HQ samples from `show_images(...)`.

## Core training pipeline (DDPM-style)

The diffusion training loop has five main pieces:

### 1) Noise schedule

A linear beta schedule was defined across fixed timesteps. These beta values control how much noise is added at each step.

### 2) Forward process

For each image, a timestep is sampled and corresponding noise is injected to produce a noisy version.

### 3) Denoising objective

The model learns to predict the injected noise from noisy input. This predict-noise objective is the standard DDPM formulation and is usually optimized with MSE loss.

### 4) Optimization

Training used stable optimizer settings and controlled learning rate across multiple epochs, with attention to convergence behavior and output progression.

### 5) Sampling

After training, reverse diffusion starts from random noise and iteratively reconstructs images. This is where learning quality becomes visible.

**Screenshot 5 (Noise Schedule / Diffusion Logic):**  
Add a screenshot of the beta schedule function and the core diffusion-related training code block.

## Evaluation: metrics + visual quality

A single metric is not enough for generative modeling, so I used a mixed approach:

- **SSIM** for structural similarity trends
- **PSNR** for reconstruction-style signal
- **Qualitative inspection** of generated image grids

This gave a more honest assessment. Metric trends provided consistency, while visual checks revealed artifacts and realism quality that numbers alone cannot fully capture.

**Screenshot 6 (Training Curves or Metric Logs):**  
Add a screenshot of epoch-wise training output, or any logged loss/SSIM/PSNR trends.

**Screenshot 7 (Generated Image Results):**  
Add a before/after-style comparison: early noisy samples vs later improved generated faces.

## Gradio integration for demonstration

To make results easy to present, I integrated Gradio as a lightweight interface. This transformed the notebook work into something interactive and easier for others to test.

For an assignment, this matters: it shows not only model training, but also usability and communication of outcomes.

**Screenshot 8 (Gradio Interface):**  
Add a screenshot of your Gradio app UI while generating sample outputs.

## Challenges and lessons learned

This project highlighted several practical ML realities:

1. Diffusion training is compute-heavy, so image size, batch size, and timesteps involve tradeoffs.
2. Hyperparameters interact strongly; small scheduler or LR changes can affect sharpness and stability.
3. SSIM/PSNR are useful but incomplete without visual inspection.
4. End-to-end pipeline thinking is essential because issues in one stage affect all later stages.

## Outcome and reflection

By the end of Assignment 04, I had a complete DDPM-style workflow that:

- trains on CelebA-HQ preprocessed images,
- learns denoising through timestep-conditioned noise prediction,
- evaluates with SSIM/PSNR plus visual checks,
- and is demonstrated through a Gradio interface.

Most importantly, this project shifted me from conceptual understanding of diffusion models to practical confidence in building and explaining one end-to-end.

## Future work

Next improvements would include:

- stronger model variants for sharper details,
- broader evaluation protocols,
- more systematic schedule/timestep ablations,
- and sampling controls for consistency.

Diffusion modeling rewards iteration, and this assignment gave me a strong foundation to keep improving both research understanding and implementation skills.

---

## Screenshot Checklist (Before Publishing on Medium)

- Screenshot 1: Notebook title + CUDA/GPU detection output
- Screenshot 2: Imports and hyperparameters
- Screenshot 3: Dataset loading confirmation
- Screenshot 4: Grid of real images
- Screenshot 5: Beta schedule and diffusion training logic
- Screenshot 6: Training logs or metric trend snapshot
- Screenshot 7: Final generated image samples
- Screenshot 8: Gradio interface demo
