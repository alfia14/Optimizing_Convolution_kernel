#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_SIZE 256

inline void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

// Fused kernel: unroll + matmul + permute in one step
__global__ void fused_conv_kernel(const float *input, const float *mask, float *output,
                                  int Batch, int Map_out, int Channel,
                                  int Height, int Width, int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    size_t total_elements = (size_t)Batch * Map_out * Height_out * Width_out;
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_elements) {
        // Decode (b, m, h, w) from index
        int w = (int)(index % Width_out);
        int tmp1 = (int)(index / Width_out);
        int h = tmp1 % Height_out;
        int tmp2 = tmp1 / Height_out;
        int m = tmp2 % Map_out;
        int b = tmp2 / Map_out;

        // Compute convolution
        float acc = 0.0f;
        // i runs over (Channel*K*K)
        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                int in_h = h + p;
                // Early exit if in_h >= Height to save time
                if (in_h >= Height) continue;
                for (int q = 0; q < K; q++) {
                    int in_w = w + q;
                    if (in_w < Width) {
                        // Compute input and mask indices safely
                        size_t input_idx = ((size_t)b * (Channel*Height*Width)) 
                                         + ((size_t)c * (Height*Width)) 
                                         + (in_h * (size_t)Width) 
                                         + in_w;
                        size_t mask_idx = ((size_t)m * (Channel*K*K))
                                        + ((size_t)c * (K*K))
                                        + (p*K)
                                        + q;
                        // Check boundaries again (c, m already checked logically)
                        if (b < Batch && c < Channel && m < Map_out && in_h < Height && in_w < Width) {
                            acc += input[input_idx] * mask[mask_idx];
                        }
                    }
                }
            }
        }

            output[index] = acc;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input,
                                                    const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr,
                                                    float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
{
    if (K > Height || K > Width) {
        std::cerr << "Invalid configuration: K cannot be greater than Height or Width." << std::endl;
        exit(-1);
    }

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    if (Height_out <= 0 || Width_out <= 0) {
        std::cerr << "Invalid output dimensions: Height_out=" << Height_out << ", Width_out=" << Width_out << std::endl;
        exit(-1);
    }

    size_t input_size = (size_t)Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = (size_t)Map_out * Channel * K * K * sizeof(float);
    size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);

    checkCudaError(cudaMalloc((void**)device_input_ptr, input_size), "malloc device_input");
    checkCudaError(cudaMalloc((void**)device_mask_ptr, mask_size), "malloc device_mask");
    checkCudaError(cudaMalloc((void**)device_output_ptr, output_size), "malloc device_output");

    checkCudaError(cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice), "memcpy input H2D");
    checkCudaError(cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice), "memcpy mask H2D");
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input,
                                             const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel,
                                             const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    size_t total_elements = (size_t)Batch * Map_out * Height_out * Width_out;

    int threads_per_block = BLOCK_SIZE;
    int num_blocks = (int)((total_elements + threads_per_block - 1) / threads_per_block);

    // Launch the fused kernel
    fused_conv_kernel<<<num_blocks, threads_per_block>>>(device_input, device_mask, device_output,
                                                         Batch, Map_out, Channel,
                                                         Height, Width, K);
    checkCudaError(cudaPeekAtLastError(), "fused_conv_kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "fused_conv_kernel sync");
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out,
                                                    const int Channel, const int Height,
                                                    const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    size_t output_size = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);

    checkCudaError(cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost), "D2H output");

    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);

    checkCudaError(cudaGetLastError(), "epilog end");
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "
                 <<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "
                 <<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
