#ifndef __LIBHERIC_LIBHERIC_H__
#define __LIBHERIC_LIBHERIC_H__


#include "constants.h"
#include "simulation_box.h"
#include "default_acc.h"
#include "cuda_acc.h"
#include "hip_acc.h"
#include "oneapi_acc.h"
#include "xrt_acc.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace libheric
{
    std::map<acc_kind, std::function<std::vector<acc_device>(const acc_kind&)>> discover_acc_kind = 
        {
            {{"A100","CUDA","libintx"}, discover_cuda_accelerators},
            {{"MI210","HIP","libintx"}, discover_hip_accelerators},
            {{"P520","ONEAPI","xin_eris"}, discover_oneapi_accelerators},
            {{"U280","XRT","seri"}, discover_xrt_accelerators},
            {{"VCK5000","XRT","eri_aie"}, discover_xrt_accelerators},
            {{"CPU", "CPU", "libint"}, discover_cpu_accelerator}
        };
    
    template <typename acc_context_t> 
    acc_context* alloc_acc_context(const acc_device& acc_dev, 
                                   const simulation_box& sim_box)
    {
        static_assert(std::is_base_of_v<acc_context, acc_context_t>, 
                     "Cannot allocate acc_context_t which is not derived from acc_context!");
        acc_context* ctx = new acc_context_t(acc_dev, sim_box);
        return ctx;
    }

    std::map<acc_kind, std::function<acc_context*(const acc_device& acc_dev, 
                                                  const simulation_box& sim_box)>> alloc_acc_context_picker = 
        {
            {{"A100","CUDA","libintx"}, alloc_acc_context<cuda_acc_context>},
            {{"MI210","HIP","libintx"}, alloc_acc_context<hip_acc_context>},
            {{"P520","ONEAPI","xin_eris"}, alloc_acc_context<oneapi_acc_context>},
            {{"U280","XRT","seri"}, alloc_acc_context<xrt_acc_context>},
            {{"VCK5000","XRT","eri_aie"}, alloc_acc_context<xrt_acc_context>}
        };
    std::vector<acc_device> discover_accelerators(std::vector<acc_kind> acc_query);
}

#endif
