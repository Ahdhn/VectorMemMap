#pragma once
#include <cuda.h>
#include <vector>

#include "helper.h"


struct VectorMemMap
{
    CUdeviceptr         d_p;
    CUmemAllocationProp prop;
    CUmemAccessDesc     access_desc;
    struct Range
    {
        CUdeviceptr start;
        size_t      size;
    };
    std::vector<Range>                        va_ranges;
    std::vector<CUmemGenericAllocationHandle> handles;
    std::vector<size_t>                       handle_sizes;
    size_t                                    alloc_size;
    size_t                                    reserve_size;
    size_t                                    chunk_size;

    VectorMemMap(CUcontext context)
        : d_p(0ULL),
          prop(),
          handles(),
          alloc_size(0ULL),
          reserve_size(0ULL),
          chunk_size(0ULL)
    {
        CUdevice  device;
        CUcontext prev_ctx;

        CHECK_DRV(cuCtxGetCurrent(&prev_ctx));
        CHECK_DRV(cuCtxGetDevice(&device));
        CHECK_DRV(cuCtxSetCurrent(prev_ctx));

        prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id         = (int)device;
        prop.win32HandleMetaData = NULL;

        access_desc.location = prop.location;
        access_desc.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CHECK_DRV(cuMemGetAllocationGranularity(
            &chunk_size, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }

    ~VectorMemMap()
    {
        if (d_p != 0ULL) {
            CHECK_DRV(cuMemUnmap(d_p, alloc_size));
            for (size_t i = 0; i < va_ranges.size(); ++i) {
                CHECK_DRV(
                    cuMemAddressFree(va_ranges[i].start, va_ranges[i].size));
            }
            for (size_t i = 0ULL; i < handles.size(); ++i) {
                CHECK_DRV(cuMemRelease(handles[i]));
            }
        }
    }

    CUdeviceptr get_pointer() const
    {
        return d_p;
    }

    size_t get_size() const
    {
        return alloc_size;
    }

    /**
     * @brief reserve some extra space in order to speedup grow()
     */
    CUresult reserve(size_t new_size)
    {
        CUresult status = CUDA_SUCCESS;
        if (new_size <= reserve_size) {
            return status;
        }

        CUdeviceptr new_ptr = 0ULL;

        const size_t aligned_size =
            chunk_size * DIVIDE_UP(new_size, chunk_size);

        // try to reserve an address just after what we already have reserved
        status = cuMemAddressReserve(&new_ptr,
                                     (aligned_size - reserve_size),
                                     0ULL,
                                     d_p + reserve_size,
                                     0ULL);
        if (status != CUDA_SUCCESS || (new_ptr != d_p + reserve_size)) {
            if (new_ptr != 0ULL) {
                // avoid memory leaks
                CHECK_DRV(
                    cuMemAddressFree(new_ptr, (aligned_size - reserve_size)));
            }

            // slow path: try to find a new address reservation
            status = cuMemAddressReserve(&new_ptr, aligned_size, 0ULL, 0U, 0);
            if (status == CUDA_SUCCESS && d_p != 0ULL) {
                CUdeviceptr ptr = new_ptr;
                // found a range, now unmap the previous allocation
                CHECK_DRV(cuMemUnmap(d_p, alloc_size));
                for (size_t i = 0ULL; i < handles.size(); ++i) {

                    status =
                        cuMemMap(ptr, handle_sizes[i], 0ULL, handles[i], 0ULL);
                    if (status != CUDA_SUCCESS) {
                        break;
                    }

                    status = cuMemSetAccess(
                        ptr, handle_sizes[i], &access_desc, 1ULL);

                    if (status != CUDA_SUCCESS) {
                        break;
                    }

                    status;
                    ptr += handle_sizes[i];
                }

                if (status != CUDA_SUCCESS) {
                    // failed the mapping somehow.. cleanup
                    CHECK_DRV(cuMemUnmap(new_ptr, aligned_size));
                    CHECK_DRV(cuMemAddressFree(new_ptr, aligned_size));
                }
            } else {
                // clean up our old VA reservations
                for (size_t i = 0ULL; i < va_ranges.size(); ++i) {
                    CHECK_DRV(cuMemAddressFree(va_ranges[i].start,
                                               va_ranges[i].size));
                }
                va_ranges.clear();
            }

            if (status == CUDA_SUCCESS) {
                d_p          = new_ptr;
                reserve_size = aligned_size;

                Range r;
                r.start = new_ptr;
                r.size  = aligned_size;
                va_ranges.push_back(r);
            }
        } else {
            Range r;
            r.start = new_ptr;
            r.size  = aligned_size - reserve_size;
            va_ranges.push_back(r);
            if (d_p == 0ULL) {
                d_p = new_ptr;
            }
            reserve_size = aligned_size;
        }

        return status;
    }

    /**
     * @brief actually commits new_size (num bytes) of additional memory
     */
    CUresult grow(size_t new_size)
    {
        CUresult status = CUDA_SUCCESS;

        if (new_size <= alloc_size) {
            return status;
        }
        CUmemGenericAllocationHandle handle;

        const size_t size_diff = new_size - alloc_size;

        const size_t size = chunk_size * DIVIDE_UP(size_diff, chunk_size);

        status = reserve(alloc_size + size);
        if (status != CUDA_SUCCESS) {
            return status;
        }

        status = cuMemCreate(&handle, size, &prop, 0);
        if (status == CUDA_SUCCESS) {
            status = cuMemMap(d_p + alloc_size, size, 0ULL, handle, 0ULL);
            if (status == CUDA_SUCCESS) {
                status =
                    cuMemSetAccess(d_p + alloc_size, size, &access_desc, 1ULL);
                if (status == CUDA_SUCCESS) {
                    handles.push_back(handle);
                    handle_sizes.push_back(size);
                    alloc_size += size;
                }
                if (status != CUDA_SUCCESS) {
                    cuMemUnmap(d_p + alloc_size, size);
                }
            }
            if (status != CUDA_SUCCESS) {
                cuMemRelease(handle);
            }
        }
        return status;
    }
};