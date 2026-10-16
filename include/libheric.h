#ifndef __LIBHERIC_LIBHERIC_H__
#define __LIBHERIC_LIBHERIC_H__


#include <constants.h>
#include <simulation_box.h>
#include <cpu_acc.h>
#include <cuda_acc.h>
#include <hip_acc.h>
#include <oneapi_acc.h>
#include <xrt_acc.h>

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace libheric
{
    bool initialize(uint32_t mpi_rank);
    void release();

    std::vector<acc_device> discover_accelerators(std::vector<acc_kind> acc_query);

    template <typename acc_context_t> 
    abstract_acc_context* alloc_acc_context(const acc_device& acc_dev, 
                                            const simulation_box& sim_box,
                                            uint32_t ctx_id,
                                            uint64_t mem_available,
                                            bool compute_raw_cartesian,
                                            abstract_acc_context* master_ctx);
}

#endif
