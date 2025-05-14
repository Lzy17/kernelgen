import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

hinge_loss_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = 1 - predictions[idx] * targets[idx];
        out[idx] = val > 0 ? val : 0;
    }
}

torch::Tensor hinge_loss_rocm(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto out = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hinge_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), out.data_ptr<float>(), size);

    return out.mean();
}
"""

hinge_loss_cpp_source = "torch::Tensor hinge_loss_rocm(torch::Tensor predictions, torch::Tensor targets);"

hinge_loss = load_inline(
    name='hinge_loss',
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=['hinge_loss_rocm'],
    verbose=True,
    extra_cflags=['-I/opt/rocm/include'],
    extra_cuda_cflags=['-I/opt/rocm/include'],
    extra_ldflags=['-L/opt/rocm/lib', "-lamdhip64"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_rocm(predictions, targets)

"""
hardware AMD Instinct MI300X
runtime 0.0489
"""
