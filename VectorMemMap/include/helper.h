#pragma once

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define ROUND_UP_TO_NEXT_MULTIPLE(num, mult) (DIVIDE_UP(num, mult) * mult)

// used for integer rounding
#define DIVIDE_UP(num, divisor) (num + divisor - 1) / (divisor)


static inline void checkDrvError(CUresult    res,
                                 const char* tok,
                                 const char* file,
                                 unsigned    line)
{
    if (res != CUDA_SUCCESS) {
        const char* errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok << "failed ("
                  << (unsigned)res << "): " << errStr << std::endl;
        abort();
    }
}
#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);


static inline void HandleError(cudaError_t err, const char* file, int line)
{
    // Error handling micro, wrap it around function whenever possible
    if (err != cudaSuccess) {
        printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))


class CUDATimer
{
   public:
    CUDATimer()
    {
        CUDA_ERROR(cudaEventCreate(&m_start));
        CUDA_ERROR(cudaEventCreate(&m_stop));
    }
    ~CUDATimer()
    {
        CUDA_ERROR(cudaEventDestroy(m_start));
        CUDA_ERROR(cudaEventDestroy(m_stop));
    }
    void start(cudaStream_t stream = 0)
    {
        m_stream = stream;
        CUDA_ERROR(cudaEventRecord(m_start, m_stream));
    }
    void stop()
    {
        CUDA_ERROR(cudaEventRecord(m_stop, m_stream));
        CUDA_ERROR(cudaEventSynchronize(m_stop));
    }
    float elapsed_millis()
    {
        float elapsed = 0;
        CUDA_ERROR(cudaEventElapsedTime(&elapsed, m_start, m_stop));
        return elapsed;
    }

   private:
    cudaEvent_t  m_start, m_stop;
    cudaStream_t m_stream;
};
