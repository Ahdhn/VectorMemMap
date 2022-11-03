#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include "helper.h"

#include "VectorMemMap.cuh"

__global__ void exec_kernel()
{
    printf("\n I am thread %d from exec_kernel\n", threadIdx.x);
}

TEST(Test, exe)
{
    CUcontext    ctx;
    VectorMemMap vector(ctx);

    exec_kernel<<<1, 1>>>();
    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);    
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
