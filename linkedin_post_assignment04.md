Built and deployed a DDPM-style diffusion model on CelebA-HQ as part of my Assignment 04, and this project was a major hands-on step in my generative AI journey.

I implemented an end-to-end workflow in PyTorch: dataset preprocessing (resize/normalize + DataLoader), linear noise scheduling, denoising-objective training, iterative reverse sampling, and evaluation using SSIM/PSNR combined with qualitative image inspection.

To make the project usable beyond the notebook, I also integrated a Gradio interface for interactive generation and demonstration.

Key takeaways:
- Diffusion training quality depends heavily on scheduler and hyperparameter interactions.
- Quantitative metrics are useful, but visual assessment is essential for generative tasks.
- Building the full pipeline taught me much more than using a prebuilt model.

This assignment strengthened my understanding of both the theory and engineering of modern image generation systems.

#GenerativeAI #DiffusionModels #PyTorch #DeepLearning #ComputerVision #MachineLearning #Gradio #StudentProject #AIEngineering
