import torch
print("torch           ", torch.__version__)
print("cuda build      ", torch.version.cuda)
print("device          ", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("capability      ", "sm_%d%d" % cap)
print("arch_list       ", torch.cuda.get_arch_list())
print("total VRAM GiB  ", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

# The real test of TORCH_CUDA_ARCH_LIST="7.5;8.6+PTX": compiled detectron2
# CUDA kernels must launch on sm_86 without a JIT fallback or a no-kernel-image error.
from detectron2 import _C
from detectron2.layers import nms, ROIAlign
b = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]], device="cuda")
s = torch.tensor([0.9, 0.8, 0.7], device="cuda")
print("nms keep        ", nms(b, s, 0.5).tolist())
feat = torch.randn(1, 3, 32, 32, device="cuda")
rois = torch.tensor([[0., 0., 0., 16., 16.]], device="cuda")
print("ROIAlign out    ", tuple(ROIAlign((7, 7), 1.0, 2, aligned=True)(feat, rois).shape))

# Ampere-only capabilities that were unavailable on the 2080 Ti.
print("bf16 supported  ", torch.cuda.is_bf16_supported())
print("tf32 allowed    ", torch.backends.cuda.matmul.allow_tf32, "(default)")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul     ", (x @ x).dtype)
print("peak VRAM GiB   ", round(torch.cuda.max_memory_allocated() / 1024**3, 3))
