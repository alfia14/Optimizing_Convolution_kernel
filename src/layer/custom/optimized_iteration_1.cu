#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

inline void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

// Unroll the input feature maps into a 2D matrix for GEMM
__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        int Batch, int Channel,
                                        int Height, int Width, int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    size_t Rows = (size_t)Channel * K * K;
    size_t Cols = (size_t)Batch * Height_out * Width_out;

    size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Rows && col < Cols) {
        int c = (int)(row / (K*K));
        int rem = (int)(row % (K*K));
        int p = rem / K;
        int q = rem % K;

        int b = (int)(col / (Height_out * Width_out));
        int temp = (int)(col % (Height_out * Width_out));
        int h = temp / Width_out;
        int w = temp % Width_out;

        #define in_4d(i3, i2, i1, i0) input[((size_t)(i3)*(Channel*Height*Width) + (i2)*(Height*Width) + (i1)*Width + (i0))]

        if (b < Batch && c < Channel && (h+p) < Height && (w+q) < Width) {
            output[row * Cols + col] = in_4d(b, c, h+p, w+q);
        } else {
            // Out of range, set to 0
            output[row * Cols + col] = 0.0f;
        }
        #undef in_4d
    }
}

// Joint Register and Shared Memory Tiling kernel for matrix multiplication
__global__ void matrixMultiplyJointTiling(const float *A, const float *B, float *C,
                                          int numARows, int numAColumns,
                                          int numBRows, int numBColumns,
                                          int numCRows, int numCColumns)
{
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float val = 0.0f;

    int phases = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < phases; m++) {
        int A_col = m*TILE_WIDTH + threadIdx.x;
        int B_row = m*TILE_WIDTH + threadIdx.y;

        float aVal = 0.0f;
        float bVal = 0.0f;

        if (row < numARows && A_col < numAColumns) {
            aVal = A[(size_t)row * numAColumns + A_col];
        }
        if (col < numCColumns && B_row < numBRows) {
            bVal = B[(size_t)B_row * numBColumns + col];
        }

        sharedA[threadIdx.y][threadIdx.x] = aVal;
        sharedB[threadIdx.y][threadIdx.x] = bVal;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            val += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[(size_t)row * numCColumns + col] = val;
    }
}

// Permute kernel as before, ensure boundaries
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y; // 0 <= b < Batch
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x; 
    if (b < Batch && x < image_size) {
        // input: shape Map_out x Batch x image_size
        // output: shape Batch x Map_out x image_size
        // Both total size: Map_out * Batch * image_size
        for (int m = 0; m < Map_out; m++) {
            // Check indexing
            size_t in_idx = (size_t)m * Batch * image_size + (size_t)b * image_size + x;
            size_t out_idx = (size_t)b * Map_out * image_size + m * image_size + x;
            output[out_idx] = input[in_idx];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input,
                                                    const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr,
                                                    float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K)
{
    // Dimension checks
    if (K > Height || K > Width) {
        std::cerr << "Invalid configuration: K cannot be greater than Height or Width." << std::endl;
        exit(-1);
    }

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    if (Height_out <= 0 || Width_out <= 0) {
        std::cerr << "Invalid output dimensions: Height_out=" << Height_out << " Width_out=" << Width_out << std::endl;
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
    size_t Height_unrolled = (size_t)Channel * K * K;
    size_t Width_unrolled = (size_t)Batch * Height_out * Width_out;

    // Intermediate buffers
    float *unrolled_matrix;  
    float *matmul_output;    

    checkCudaError(cudaMalloc((void**)&unrolled_matrix, Height_unrolled * Width_unrolled * sizeof(float)), "malloc unrolled_matrix");
    checkCudaError(cudaMalloc((void**)&matmul_output, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float)), "malloc matmul_output");

    // Launch unrolling kernel
    {
        dim3 blockDim(16,16);
        dim3 gridDim((unsigned)((Width_unrolled + 15)/16), (unsigned)((Height_unrolled + 15)/16));
        matrix_unrolling_kernel<<<gridDim, blockDim>>>(device_input, unrolled_matrix,
                                                       Batch, Channel, Height, Width, K);
        checkCudaError(cudaPeekAtLastError(), "unrolling kernel launch");
        checkCudaError(cudaDeviceSynchronize(), "unrolling kernel sync");
    }

    // Matrix multiplication
    {
        int numARows = Map_out;
        int numAColumns = Channel*K*K;
        int numBRows = Channel*K*K;
        int numBColumns = (int)Width_unrolled;
        int numCRows = Map_out;
        int numCColumns = (int)Width_unrolled;

        // Basic dimension checks
        if (numAColumns != numBRows) {
            std::cerr << "Dimension mismatch: AColumns=" << numAColumns << " BRows=" << numBRows << std::endl;
            exit(-1);
        }

        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((unsigned)((numCColumns + TILE_WIDTH -1)/TILE_WIDTH),
                     (unsigned)((numCRows + TILE_WIDTH -1)/TILE_WIDTH));

        matrixMultiplyJointTiling<<<gridDim, blockDim>>>(device_mask, unrolled_matrix, matmul_output,
                                                          numARows, numAColumns,
                                                          numBRows, numBColumns,
                                                          numCRows, numCColumns);
        checkCudaError(cudaPeekAtLastError(), "matmul kernel launch");
        checkCudaError(cudaDeviceSynchronize(), "matmul kernel sync");
    }

    // Permute the result
    {
        int out_image_size = Height_out * Width_out;
        dim3 permute_kernel_grid_dim((unsigned)((out_image_size - 1)/BLOCK_SIZE + 1), (unsigned)Batch, 1);
        matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
            matmul_output, device_output, Map_out, Batch, out_image_size
        );
        checkCudaError(cudaPeekAtLastError(), "permute kernel launch");
        checkCudaError(cudaDeviceSynchronize(), "permute kernel sync");
    }

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);

    checkCudaError(cudaGetLastError(), "forward end");
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
    cudaFree(device_output);
    cudaFree(device_mask);

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
