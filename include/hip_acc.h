#ifndef __LIBHERIC_HIP_ACC_H__
#define __LIBHERIC_HIP_ACC_H__

#include "default_acc.h"
#include <vector>
namespace libheric
{
    std::vector<acc_device> discover_hip_accelerators(const acc_kind& acc_query); 

    class hip_acc_context : public acc_context
    {
    public:
        hip_acc_context(const acc_device& acc_dev, 
                        const simulation_box& sim_box);
        virtual ~hip_acc_context();
    };
}

#endif
