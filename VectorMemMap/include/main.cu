#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

#include "helper.h"

#include "VectorMemMap.cuh"

__global__ void exec_kernel()
{
    printf("\n I am thread %d from exec_kernel\n", threadIdx.x);
}


int main(int argc, char** argv)
{
    CUcontext ctx;
    CUdevice  dev;

    CHECK_DRV(cuInit(0));
    CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, 0));
    CHECK_DRV(cuCtxSetCurrent(ctx));
    CHECK_DRV(cuCtxGetDevice(&dev));

    size_t free;

    auto print_free = [&free]() {
        CHECK_DRV(cuMemGetInfo(&free, NULL));
        std::cout << "Total free memory: " << (float)free / std::giga::num
                  << "GB\n";
    };
    print_free();

    using ElemType = uint32_t;

    const size_t minN =
        (2ULL * 1024ULL * 1024ULL + sizeof(ElemType) - 1ULL) / sizeof(ElemType);

    const size_t maxN = 3ULL * free / (4ULL * sizeof(ElemType));

    int supportsVMM = 0;
    CHECK_DRV(cuDeviceGetAttribute(
        &supportsVMM,
        CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        dev));

    VectorMemMap<ElemType> vector(ctx);
    CHECK_DRV(vector.reserve(maxN));
    CHECK_DRV(vector.grow(maxN));
    print_free();

    CHECK_DRV(cuDevicePrimaryCtxRelease(0));

    return 0;
}
