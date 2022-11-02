#pragma once
#include <cuda.h>
#include <vector>

#include "helper.h"


class MemMapAlloc
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

    MemMapAlloc(CUcontext context)
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

    ~MemMapAlloc()
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
    void reserve(size_t new_size)
    {
        if (new_size <= reserve_size) {
            return;
        }

        CUdeviceptr new_ptr = 0ULL;

        const size_t aligned_size =
            chunk_size * DIVIDE_UP(new_size, chunk_size);

        // try to reserve an address just after what we already have reserved
        CUresult status = cuMemAddressReserve(&new_ptr,
                                              (aligned_size - reserve_size),
                                              0ULL,
                                              d_p + reserve_size,
                                              0ULL);
        if (status != CUDA_SUCCESS || (new_ptr != d_p + reserve_size)) {
            if (new_ptr != 0ULL) {
                //avoid memory leaks 
                CHECK_DRV(
                    cuMemAddressFree(new_ptr, (aligned_size - reserve_size)));
            }

            // slow path: try to find a new address reservation
            status = cuMemAddressReserve(&new_ptr, aligned_size, 0ULL, 0U, 0);

        }
    }

    /**
     * @brief actually commits new_size (num bytes) of additional memory
     */
    void grow(size_t new_size)
    {
    }
};


template <typename T>
struct VectorMemMap
{
    VectorMemMap(size_t size, int current_device_id = 0)
        : m_ptr(nullptr), m_padded_size(0), m_size(size)
    {
        size_t granularity = 0;

        // prop, and granularity for alignment
        CUmemAllocationProp prop = {};
        prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id         = current_device_id;

        CHECK_DRV(cuMemGetAllocationGranularity(
            &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        m_padded_size = ROUND_UP_TO_NEXT_MULTIPLE(m_size, granularity);

        // handle
        CHECK_DRV(cuMemCreate(&m_alloc_handle, m_padded_size, &prop, 0));

        // reserve a virtual address range
        CHECK_DRV(cuMemAddressReserve(&m_ptr, m_padded_size, 0, 0, 0));

        // map the virtual address range to physical allocation
        CHECK_DRV(cuMemMap(m_ptr, m_padded_size, 0, m_alloc_handle, 0));

        // allow accessing the VA from device
        CUmemAccessDesc access_desc = {};
        access_desc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id     = current_device_id;
        access_desc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // now we can access [ptr, ptr+size] from the device
        CHECK_DRV(cuMemSetAccess(m_ptr, m_ptr + m_size, &access_desc, 1));
    }

    CUresult reserve(size_t new_size)
    {
    }

    void free()
    {
        // unmap the virtual address (VA) range to reverts the VA range back to
        // the state it was in just before cuMemAddressReserve
        CHECK_DRV(cuMemUnmap(m_ptr, m_size));

        // to invalidate the handle, and if there is no mapped references left,
        // releases the backing store of memory back to the operating system
        CHECK_DRV(cuMemRelease(m_alloc_handle));

        // returns the VA range to CUDA to use for other things
        CHECK_DRV(cuMemAddressFree(m_ptr, m_size));
    }

    T*                           m_ptr;
    size_t                       m_padded_size;
    size_t                       m_size;
    CUmemGenericAllocationHandle m_alloc_handle;
};