# Layer Normalization CUDA Kernels

## Overview

This project implements optimized CUDA kernels for Layer Normalization, a fundamental normalization technique in deep learning. Layer Normalization normalizes inputs across the feature dimension for each sample independently, providing stable training dynamics and improved convergence.

The mathematical formulation is:
```
LayerNorm(x) = γ * (x - μ) / σ + β

Where:
- μ = mean(x) = (1/K) * Σx_i        (mean across feature dimension)
- σ = std(x) = sqrt((1/K) * Σ(x_i - μ)²)  (standard deviation)
- γ = scale parameter (learnable)
- β = bias parameter (learnable)
```
