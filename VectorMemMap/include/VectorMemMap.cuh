#pragma once
#include <cuda.h>

#include "helper.h"
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