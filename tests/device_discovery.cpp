#include <iostream>

#include "device_discovery.h"
#include "xrt_discovery.h"
#include "cuda_discovery.h"
#include "hip_discovery.h"

#include <hwloc.h>

int main()
{
    auto vcks = discover_accelerator<ACC_TYPE::XRT>("VCK5000");
    auto u55cs = discover_accelerator<ACC_TYPE::XRT>("U55C");
    auto a100s = discover_accelerator<ACC_TYPE::CUDA>("A100");
    auto mi210s = discover_accelerator<ACC_TYPE::HIP>("MI210");

    hwloc_topology_t topology;
    if(hwloc_topology_init(&topology) != 0)
    {
        std::cout << "[ERROR] Could not initiliaze HWLOC topology\n";
        exit(1);
    }
    
    if(hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT) != 0)
    {
        std::cout << "[ERROR] Could not enable IO device filter\n";
        exit(1);
    }
    
    if(hwloc_topology_load(topology) != 0)
    {
        std::cout << "[ERROR] Failed to load HWLOC topology\n";
        exit(1);
    }
    


    for (const auto& vck : vcks)
    {
        int id = get_deviceID_by_BDF<ACC_TYPE::XRT>(vck.second);
        if(id != -1)
            std::cout << "Found VCK5000 (id = " << id << ", name = " << vck.first << ") @ BDF: " << vck.second << '\n';
        hwloc_cpuset_t device_cpuset = get_accelerator_device_cpuset(topology, vck.second);
        char cpuset_str[256];
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, device_cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            exit(1);
        }
        std::cout << "cpuset of device: " << cpuset_str << '\n';
        
        hwloc_cpuset_t rank_cpuset = get_current_proc_cpuset(topology);
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, rank_cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            exit(1);
        }
        std::cout << "cpuset of rank:   " << cpuset_str << '\n'; 
    }
    
    for (const auto& u55c : u55cs)
    {
        int id = get_deviceID_by_BDF<ACC_TYPE::XRT>(u55c.second); 
        if(id != -1)
            std::cout << "Found U55C (id = " << id << ", name = " << u55c.first << ") @ BDF: " << u55c.second << '\n';
        hwloc_cpuset_t device_cpuset = get_accelerator_device_cpuset(topology, u55c.second);
        char cpuset_str[256];
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, device_cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            exit(1);
        }
        std::cout << "cputset of device: " << cpuset_str << '\n';
        
        hwloc_cpuset_t rank_cpuset = get_current_proc_cpuset(topology);
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, rank_cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            exit(1);
        }
        std::cout << "cpuset of rank:   " << cpuset_str << '\n'; 
    }
    
    for (const auto& a100 : a100s)
    {
        int id = get_deviceID_by_BDF<ACC_TYPE::CUDA>(a100.second); 
        if(id != -1)
            std::cout << "Found A100 (id = " << id << ", name = " << a100.first << ") @ BDF: " << a100.second << '\n';
        hwloc_cpuset_t device_cpuset = get_accelerator_device_cpuset(topology, a100.second);
        char cpuset_str[256];
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, device_cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            exit(1);
        }
        std::cout << "cputset of device: " << cpuset_str << '\n';
        
        hwloc_cpuset_t rank_cpuset = get_current_proc_cpuset(topology);
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, rank_cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            exit(1);
        }
        std::cout << "cpuset of rank:   " << cpuset_str << '\n'; 
    }
    
    for (const auto& mi210 : mi210s)
    {
        int id = get_deviceID_by_BDF<ACC_TYPE::HIP>(mi210.second); 
        if(id != -1)
            std::cout << "Found MI210 (id = " << id << ", name = " << mi210.first << ") @ BDF: " << mi210.second << '\n';
        hwloc_cpuset_t device_cpuset = get_accelerator_device_cpuset(topology, mi210.second);
        char cpuset_str[256];
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, device_cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            exit(1);
        }
        std::cout << "cputset of device: " << cpuset_str << '\n';
        
        hwloc_cpuset_t rank_cpuset = get_current_proc_cpuset(topology);
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, rank_cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            exit(1);
        }
        std::cout << "cpuset of rank:   " << cpuset_str << '\n'; 
    }
}
