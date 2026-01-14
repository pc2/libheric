#ifndef __LIBHERIC_XRT_ACC_H__
#define __LIBHERIC_XRT_ACC_H__

#include "default_acc.h"
#include <vector>
namespace libheric
{
    std::vector<acc_device> discover_xrt_accelerators(const acc_kind& acc_query); 

    class xrt_acc_context : public acc_context
    {
    public:
        xrt_acc_context(const acc_device& acc_dev, 
                        const simulation_box& sim_box);
        virtual ~xrt_acc_context();
    };
}

#endif
