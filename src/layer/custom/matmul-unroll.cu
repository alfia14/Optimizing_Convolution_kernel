#include <cmath>
#include <iostream>
#include <stdint.h> // For int64_t
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

// Unroll input feature maps into a matrix with batch tiling
__global__ void matrix_unrolling_kernel(const float *input, float *output,const int Batch, const int Channel, const int Height, const int Width,
                                        const int K, const int batch_start, const int current_batch_size)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    size_t H_unroll = (size_t)Channel * K * K;
    size_t W_unroll = (size_t)current_batch_size * Height_out * Width_out;

    size_t h_unroll = blockIdx.y * blockDim.y + threadIdx.y; // Row index in unrolled matrix
    size_t w_unroll = blockIdx.x * blockDim.x + threadIdx.x; // Column index in unrolled matrix

    if (h_unroll < H_unroll && w_unroll < W_unroll)
    {
        int c = h_unroll / (K * K);
        int s = h_unroll % (K * K);
        int p = s / K;
        int q = s % K;

        int b_local = w_unroll / (Height_out * Width_out);
        int b = b_local + batch_start;
        int t = w_unroll % (Height_out * Width_out);
        int h_out = t / Width_out;
        int w_out = t % Width_out;

        int h_in = h_out + p;
        int w_in = w_out + q;

        if (b < Batch && c < Channel && h_in < Height && w_in < Width)
        {
            size_t input_idx = ((size_t)b * Channel * Height * Width) + 
            ((size_t)c * Height * Width) + (size_t)h_in * Width + w_in;
            size_t output_idx = h_unroll * W_unroll + w_unroll;
            output[output_idx] = input[input_idx];
        }
    }
}

// Tiled matrix multiplication kernel
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++)
    {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns)
        {
            tileA[ty][tx] = A[(size_t)row * numAColumns + tileId * TILE_WIDTH + tx];
        }
        else
        {
            tileA[ty][tx] = 0.0f;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows)
        {
            tileB[ty][tx] = B[(size_t)(tileId * TILE_WIDTH + ty) * numBColumns + col];
        }
        else
        {
            tileB[ty][tx] = 0.0f;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns)
        {
            for (int i = 0; i < TILE_WIDTH; i++)
            {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns){
        C[(size_t)row * numCColumns + col] = val;
    }
}

// Permute the matmul result with batch tiling
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int Height_out, int Width_out,
                                      int batch_start, int current_batch_size)
{
    int out_image_size = Height_out * Width_out;
    int b_local = blockIdx.y;
    int b = b_local + batch_start;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (x < out_image_size && b < Batch)
    {
        for (int m = 0; m < Map_out; m++)
        {
            size_t input_idx = ((size_t)m * current_batch_size + b_local) * out_image_size + x;
            size_t output_idx = ((size_t)b * Map_out + m) * out_image_size + x;
            output[output_idx] = input[input_idx];
        }
    }
}

// Reshape the convolution mask (weights) for matrix multiplication
void reshape_mask(const float *host_mask, float *host_mask_reshaped, int Map_out, int Channel, int K)
{
    int H_unroll = Channel * K * K;

    for (int m = 0; m < Map_out; ++m)
    {
        for (int c = 0; c < Channel; ++c)
        {
            for (int p = 0; p < K; ++p)
            {
                for (int q = 0; q < K; ++q)
                {
                    int h_unroll = c * K * K + p * K + q;
                    host_mask_reshaped[m * H_unroll + h_unroll] = host_mask[((m * Channel + c) * K + p) * K + q];
                }
            }
        }
    }
}

// Prolog function: Allocate memory and copy data to GPU
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input,
                                                    const float *host_mask, float **device_output_ptr,
                                                    float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
{
    cudaError_t err;

    // Allocate device memory for input
    err = cudaMalloc((void **)device_input_ptr, (size_t)Batch * Channel * Height * Width * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating device_input_ptr: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate device memory for output
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    err = cudaMalloc((void **)device_output_ptr, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating device_output_ptr: " << cudaGetErrorString(err) << std::endl;
        cudaFree(*device_input_ptr);
        exit(EXIT_FAILURE);
    }

    // Allocate device memory for reshaped mask
    int H_unroll = Channel * K * K;
    err = cudaMalloc((void **)device_mask_ptr, (size_t)Map_out * H_unroll * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating device_mask_ptr: " << cudaGetErrorString(err) << std::endl;
        cudaFree(*device_input_ptr);
        cudaFree(*device_output_ptr);
        exit(EXIT_FAILURE);
    }

    // Copy input to device
    err = cudaMemcpy(*device_input_ptr, host_input, (size_t)Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Error copying input to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(*device_input_ptr);
        cudaFree(*device_output_ptr);
        cudaFree(*device_mask_ptr);
        exit(EXIT_FAILURE);
    }

    // Reshape and copy mask to device
    float *host_mask_reshaped = (float *)malloc((size_t)Map_out * H_unroll * sizeof(float));
    reshape_mask(host_mask, host_mask_reshaped, Map_out, Channel, K);

    err = cudaMemcpy(*device_mask_ptr, host_mask_reshaped, (size_t)Map_out * H_unroll * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Error copying mask to device: " << cudaGetErrorString(err) << std::endl;
        free(host_mask_reshaped);
        cudaFree(*device_input_ptr);
        cudaFree(*device_output_ptr);
        cudaFree(*device_mask_ptr);
        exit(EXIT_FAILURE);
    }
    free(host_mask_reshaped);
}

// Main convolution function with batch tiling
__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input,
                                             const float *device_mask, const int Batch,
                                             const int Map_out, const int Channel, const int Height,
                                             const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int H_unroll = Channel * K * K;

    // Determine maximum batch tile size based on available device memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Estimate memory needed per batch element
    size_t per_image_unrolled_size = (size_t)H_unroll * Height_out * Width_out * sizeof(float);
    size_t per_image_matmul_output_size = (size_t)Map_out * Height_out * Width_out * sizeof(float);

    size_t max_images_in_tile = free_mem / (per_image_unrolled_size + per_image_matmul_output_size);

    // Limit to a reasonable number to avoid too small tiles
    int batch_tile_size = (int)std::min(max_images_in_tile, (size_t)1000);
    batch_tile_size = std::min(batch_tile_size, 500); // Further limit to 500
    if (batch_tile_size < 1)
    {
        std::cerr << "Not enough device memory for even one image." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Allocate memory for the batch tile
    size_t max_W_unroll = (size_t)batch_tile_size * Height_out * Width_out;
    float *unrolled_input;
    cudaError_t err = cudaMalloc((void **)&unrolled_input, (size_t)H_unroll * max_W_unroll * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating unrolled_input: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    float *matmul_output;
    err = cudaMalloc((void **)&matmul_output, (size_t)Map_out * max_W_unroll * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating matmul_output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(unrolled_input);
        exit(EXIT_FAILURE);
    }

    for (int b = 0; b < Batch; b += batch_tile_size)
    {
        int current_batch_size = std::min(batch_tile_size, Batch - b);
        size_t W_unroll = (size_t)current_batch_size * Height_out * Width_out;

        // Launch unrolling kernel
        dim3 blockDim_unroll(16, 16);
        dim3 gridDim_unroll((W_unroll + blockDim_unroll.x - 1) / blockDim_unroll.x,
                            (H_unroll + blockDim_unroll.y - 1) / blockDim_unroll.y);

        if (gridDim_unroll.x > 2147483647 || gridDim_unroll.y > 65535)
        {
            std::cerr << "Grid dimensions exceed maximum allowed size in unrolling kernel." << std::endl;
            cudaFree(unrolled_input);
            cudaFree(matmul_output);
            exit(EXIT_FAILURE);
        }

        matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll>>>(device_input, unrolled_input,
                                                                     Batch, Channel, Height, Width, K,
                                                                     b, current_batch_size);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error after matrix_unrolling_kernel: " << cudaGetErrorString(err) << std::endl;
            cudaFree(unrolled_input);
            cudaFree(matmul_output);
            exit(EXIT_FAILURE);
        }

        // Perform matrix multiplication
        int numARows = Map_out;
        int numAColumns = H_unroll;
        int numBRows = H_unroll;
        int numBColumns = (int)W_unroll;
        int numCRows = Map_out;
        int numCColumns = (int)W_unroll;

        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH,
                     (numCRows + TILE_WIDTH - 1) / TILE_WIDTH);

        if (gridDim.x > 2147483647 || gridDim.y > 65535)
        {
            std::cerr << "Grid dimensions exceed maximum allowed size in matrix multiplication kernel." << std::endl;
            cudaFree(unrolled_input);
            cudaFree(matmul_output);
            exit(EXIT_FAILURE);
        }

        matrixMultiplyShared<<<gridDim, blockDim>>>(device_mask, unrolled_input, matmul_output,
                                                    numARows, numAColumns, numBRows, numBColumns,
                                                    numCRows, numCColumns);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error after matrixMultiplyShared: " << cudaGetErrorString(err) << std::endl;
            cudaFree(unrolled_input);
            cudaFree(matmul_output);
            exit(EXIT_FAILURE);
        }

        // Permute the result
        int out_image_size = Height_out * Width_out;
        dim3 permute_block_dim(BLOCK_SIZE);
        dim3 permute_grid_dim((out_image_size + BLOCK_SIZE - 1) / BLOCK_SIZE, current_batch_size);

        if (permute_grid_dim.x > 2147483647 || permute_grid_dim.y > 65535)
        {
            std::cerr << "Grid dimensions exceed maximum allowed size in permute kernel." << std::endl;
            cudaFree(unrolled_input);
            cudaFree(matmul_output);
            exit(EXIT_FAILURE);
        }

        matrix_permute_kernel<<<permute_grid_dim, permute_block_dim>>>(matmul_output, device_output,
                                                                       Map_out, Batch, Height_out, Width_out,
                                                                       b, current_batch_size);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error after matrix_permute_kernel: " << cudaGetErrorString(err) << std::endl;
            cudaFree(unrolled_input);
            cudaFree(matmul_output);
            exit(EXIT_FAILURE);
        }
    }

    // Free temporary memory
    cudaFree(unrolled_input);
    cudaFree(matmul_output);
}

// Epilog function: Copy output back to host and free device memory
__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out,
                                                    const int Channel, const int Height,
                                                    const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    // Copy output back to host
    cudaError_t err = cudaMemcpy(host_output, device_output, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "Error copying output to host: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
