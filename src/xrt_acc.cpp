#include <xrt_acc.h>

#include <algorithm>
#include <cctype>
#include <iostream>

#ifdef ENABLE_XRT
#include <xrt/xrt_device.h>
#include <experimental/xrt_system.h>
#endif

namespace libheric
{
    std::vector<acc_device> discover_xrt_accelerators(const acc_kind& acc_query) 
    {
        std::vector<acc_device> found_devices;
#ifdef ENABLE_XRT
        uint32_t num_devices = xrt::system::enumerate_devices();
        for (uint32_t i = 0;i < num_devices; i++)
        {
            xrt::device dev = xrt::device(i);
            std::string dev_name = dev.get_info<xrt::info::device::name>();
            std::string dev_bdf  = dev.get_info<xrt::info::device::bdf >();

            std::string dev_name_upper = dev_name;
            std::for_each(dev_name_upper.begin(), dev_name_upper.end(), [](char& c){c = std::toupper(c);});

            if(dev_name_upper.find(acc_query.name) != std::string::npos)
            {
                found_devices.push_back({dev_name, dev_bdf, "", 1, acc_query});
            }

        }  
#endif
        return found_devices;
    }

    xrt_acc_context::xrt_acc_context(const acc_device& acc_dev, 
                                     const simulation_box& sim_box,
                                     uint32_t num_contexts,
                                     uint64_t mem_available,
                                     bool compute_raw_cartesian,
                                     abstract_acc_context* master_ctx)
       : abstract_acc_context(acc_dev, sim_box, num_contexts, mem_available, compute_raw_cartesian, master_ctx) 
    {
    }

    xrt_acc_context::~xrt_acc_context()
    {
    }
    
    void xrt_acc_context::set_B_offset(int off_x, int off_y, int off_z)
    {
    }
    
    void xrt_acc_context::set_D_offset(int off_x, int off_y, int off_z)
    {
    }

    void xrt_acc_context::set_exponent_ids(int exp_a, int exp_b, int exp_c, int exp_d) 
    {
    }
    
    void xrt_acc_context::set_Q_offset(int off_x, int off_y, int off_z)
    {
    }

    void xrt_acc_context::submit_peri()
    {
    }
}
