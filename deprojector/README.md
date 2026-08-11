# Harbinger.Deprojector
Developed by **Special Research Group 263/4, Confoederatio Research Division** (SRG263-CRD, SRG264-CRD).

> [!WARNING]
> **Deprojector** is extremely compute hungry, and takes ~5m to run per projection on a modern workstation, and ~10-15m over Colab. You will either need parallel compute, or a lot of patience. Confoederatio developers are working on getting this compute time down.

Plots should show up in Python IDEs like Spyder to track your progress.

Installing dependencies on local:
1. `conda activate sam_env`
2. `conda install -c conda-forge opencv numpy scipy matplotlib pillow spyder`
3. Installing core ML libraries:
  1. If you have an NVIDIA GPU: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
  2. Otherwise: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
4. `pip install kornia`
5. `pip install romatch`