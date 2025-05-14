import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom ROCm kernel for 3D tensor-matrix multiplication
tensor_matmul_source = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

__global__ void tensor_matmul_kernel(const float* A, const float* B, float* out, int N, int M, int K, int L) {
    int n = blockIdx.z;
    int m = blockIdx.y;
    int l = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < N && m < M && l < L) {
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

    auto out = torch::zeros({N, M, L}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    int block_size = 256;
    int grid_l = (L + block_size - 1) / block_size;

    dim3 grid(grid_l, M, N);
    dim3 block(block_size, 1, 1);

    tensor_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M, K, L);

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
    """
    Performs 3D tensor-matrix multiplication using custom ROCm kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matmul = tensor_matmul
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using custom ROCm kernel.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return self.tensor_matmul.tensor_matmul_rocm(A, B)

"""
hardware AMD Instinct MI300X
runtime 10.2
"""
