#include "cuda_acc.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>
namespace libheric
{
    std::vector<acc_device> discover_cuda_accelerators(const acc_kind& acc_query) 
    {
        std::vector<acc_device> found_devices;
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
        return found_devices;
    }

    cuda_acc_context::cuda_acc_context(const acc_device& acc_dev, 
                                       const simulation_box& sim_box)
       : acc_context(acc_dev, sim_box) 
    {
    }
    
    cuda_acc_context::~cuda_acc_context()
    {
    }

}
