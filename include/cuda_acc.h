#ifndef __LIBHERIC_CUDA_ACC_H__
#define __LIBHERIC_CUDA_ACC_H__

#include "default_acc.h"
#include <vector>
namespace libheric
{
    std::vector<acc_device> discover_cuda_accelerators(const acc_kind& acc_query); 

    class cuda_acc_context : public acc_context
    {
    public:
        cuda_acc_context(const acc_device& acc_dev, 
                         const simulation_box& sim_box);
        virtual ~cuda_acc_context();
    };
}

#endif
