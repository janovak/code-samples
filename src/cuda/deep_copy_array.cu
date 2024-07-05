#include <iostream>

#define CUDA_CHECK_ERROR(err) {                                               \
    if (err != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;  \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
}

template<class T>
struct MyArray {
    T* data;
    int elementCount;
};

template<class T>
struct MyStruct {
    MyArray<T> arrayA;
    MyArray<T> arrayB;
};

template<class T>
__global__ void SumArrays(MyStruct<T> inStruct, MyArray<T> outArray){
    outArray.data[threadIdx.x] = inStruct.arrayA.data[threadIdx.x] + inStruct.arrayB.data[threadIdx.x];
}

int main() {
    constexpr int size = 3;

    // Declare and initialize struct on the host
    MyStruct<float> h_myStruct;

    h_myStruct.arrayA.elementCount = size;
    h_myStruct.arrayA.data = new float[size];
    h_myStruct.arrayA.data[0] = 1.0f;
    h_myStruct.arrayA.data[1] = 2.0f;
    h_myStruct.arrayA.data[2] = 3.0f;

    h_myStruct.arrayB.elementCount = size;
    h_myStruct.arrayB.data = new float[size];
    h_myStruct.arrayB.data[0] = 4.0f;
    h_myStruct.arrayB.data[1] = 5.0f;
    h_myStruct.arrayB.data[2] = 6.0f;

    // Declare struct and malloc memory on the device
    MyStruct<float> d_myStruct;
    CUDA_CHECK_ERROR(cudaMalloc(&(d_myStruct.arrayA.data), h_myStruct.arrayA.elementCount * sizeof(h_myStruct.arrayA.data[0])));
    CUDA_CHECK_ERROR(cudaMalloc(&(d_myStruct.arrayB.data), h_myStruct.arrayB.elementCount * sizeof(h_myStruct.arrayB.data[0])));

    // Copy the host struct to the device
    d_myStruct.arrayA.elementCount = h_myStruct.arrayA.elementCount;
    CUDA_CHECK_ERROR(cudaMemcpy(d_myStruct.arrayA.data, h_myStruct.arrayA.data, h_myStruct.arrayA.elementCount * sizeof(h_myStruct.arrayA.data[0]), cudaMemcpyHostToDevice));
    d_myStruct.arrayB.elementCount = h_myStruct.arrayB.elementCount;
    CUDA_CHECK_ERROR(cudaMemcpy(d_myStruct.arrayB.data, h_myStruct.arrayB.data, h_myStruct.arrayB.elementCount * sizeof(h_myStruct.arrayB.data[0]), cudaMemcpyHostToDevice));

    // Declare struct and malloc memory on the device to hold the result
    MyArray<float> d_out;
    CUDA_CHECK_ERROR(cudaMalloc(&(d_out.data), h_myStruct.arrayA.elementCount * sizeof(h_myStruct.arrayA.data[0])));

    // Spawn the kernel to sum the arrays in the struct and store it in d_out
    SumArrays<float><<<1,size>>>(d_myStruct, d_out);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Allocate output array on the host and copy the result back
    MyArray<float> h_out;
    h_out.elementCount = size;
    h_out.data = new float[size];

    // Copy the device array to the host
    CUDA_CHECK_ERROR(cudaMemcpy(h_out.data, d_out.data, h_out.elementCount * sizeof(h_out.data[0]), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < h_out.elementCount; ++i) {
        std::cout << h_out.data[i] << std::endl;
    }

    // Free host memory
    delete[] h_myStruct.arrayA.data;
    delete[] h_myStruct.arrayB.data;
    delete[] h_out.data;

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(d_myStruct.arrayA.data));
    CUDA_CHECK_ERROR(cudaFree(d_myStruct.arrayB.data));
    CUDA_CHECK_ERROR(cudaFree(d_out.data));
}