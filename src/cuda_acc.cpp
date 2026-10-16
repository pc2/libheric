#include <cuda_acc.h>

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif


namespace libheric
{
    std::vector<acc_device> discover_cuda_accelerators(const acc_kind& acc_query) 
    {
        std::vector<acc_device> found_devices;
#ifdef ENABLE_CUDA
        int32_t num_devices;
        auto cu_res = cudaGetDeviceCount(&num_devices);
        if (cu_res != cudaSuccess)
        {
            std::string error_name = cudaGetErrorName(cu_res);
            std::string error_str  = cudaGetErrorString(cu_res);
            std::cout << "[WARNING] " << error_name << " : " << error_str << '\n';
            num_devices = 0;
        }
        for (int32_t i = 0;i < num_devices; i++)
        {
            cudaDeviceProp prop;
            auto cu_res = cudaGetDeviceProperties(&prop, i);
            if (cu_res != cudaSuccess)
            {
                std::string error_name = cudaGetErrorName(cu_res);
                std::string error_str  = cudaGetErrorString(cu_res);
                std::cout << "[WARNING] Device #" << i << " -- " << error_name << " : " << error_str << '\n';
                continue;
            }

            std::string dev_name = prop.name; 
            std::stringstream s_bdf;
            s_bdf << std::setfill('0') << std::hex  
                  << std::setw(4) << prop.pciDomainID 
                  << std::setw(1) << ":" 
                  << std::setw(2) << prop.pciBusID
                  << std::setw(1) << ":" 
                  << std::setw(2) << prop.pciDeviceID
                  << std::setw(2) << ".0";
            std::string dev_bdf  = s_bdf.str();

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

    cuda_acc_context::cuda_acc_context(const acc_device& acc_dev, 
                                       const simulation_box& sim_box,
                                       uint32_t ctx_id,
                                       uint64_t mem_available,
                                       bool compute_raw_cartesian,
                                       abstract_acc_context* master_ctx)
       : abstract_acc_context(acc_dev, sim_box, ctx_id, mem_available, compute_raw_cartesian, master_ctx) 
    {
    }
    
    cuda_acc_context::~cuda_acc_context()
    {
    }
    
    void cuda_acc_context::set_B_offset(int off_x, int off_y, int off_z)
    {
    }
    
    void cuda_acc_context::set_D_offset(int off_x, int off_y, int off_z)
    {
    }

    void cuda_acc_context::set_exponent_ids(int exp_a, int exp_b, int exp_c, int exp_d) 
    {
    }
    
    void cuda_acc_context::set_Q_offset(int off_x, int off_y, int off_z)
    {
    }

    void cuda_acc_context::submit_peri()
    {
    }

}
