import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define the custom ROCm kernel for 3D tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

__global__ void tensor_matmul_kernel(const float* A, const float* B, float* out, int N, int M, int K, int L) {
    int n = blockIdx.z;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && l < L) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[n * M * K + m * K + k] * B[k * L + l];
        }
        out[n * M * L + m * L + l] = sum;
    }
}

torch::Tensor tensor_matmul_rocm(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    auto out = torch::zeros({N, M, L}, A.options());

    dim3 block_size(16, 16);
    dim3 num_blocks((L + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y, N);

    tensor_matmul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M, K, L);

    return out;
}
"""

tensor_matmul_cpp_source = "torch::Tensor tensor_matmul_rocm(torch::Tensor A, torch::Tensor B);"

# Compile the inline ROCm code for 3D tensor-matrix multiplication
tensor_matmul = load_inline(
    name='tensor_matmul',
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_source,
    functions=['tensor_matmul_rocm'],
    verbose=True,
    extra_cflags=['-I/opt/rocm/include'],
    extra_cuda_cflags=['-I/opt/rocm/include'],
    extra_ldflags=['-L/opt/rocm/lib', "-lamdhip64"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matmul = tensor_matmul

    def forward(self, A, B):
        return self.tensor_matmul.tensor_matmul_rocm(A, B)

"""
hardware AMD Instinct MI300X
runtime 11.4
"""
