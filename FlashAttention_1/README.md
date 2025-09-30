# flash attention re-implementation

This is a re-implementation of Flash Attention #1 with CUDA and PyTorch. The kernel is ran on NVIDIA RTX 4060. 

RTX 4060 config:
- 24 SM
- up to 99KB shmem



Command line:
```bash
CUDA_ARCH_LIST="8.9+PTX" python bench.py