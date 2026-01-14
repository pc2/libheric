
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <hwloc.h>
#include <mpi.h>

#include <device_discovery.h>

int main(int argc, char** argv)
{
    int comm_rank, comm_size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size); 
    
    char hostname[256];
    int hostname_len;
    MPI_Get_processor_name(hostname, &hostname_len);

    for(int i = 0;i < comm_size;i++)
    {
        if(i == comm_rank)
        {
            std::cout << "Rank #" << std::setw(2) << i << " -- hostname: " << hostname << std::endl; 
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    char* hostnames;
    int* node_ids;
    if(comm_rank == 0)
    {
        hostnames = new char[256 * comm_size];
        node_ids  = new int[comm_size];
    }

    MPI_Gather(hostname, 256, MPI_CHAR, hostnames, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    int node_id; 
    if(comm_rank == 0)
    {
        std::cout << "Main rank received hostnames:\n";
        for(int i = 0;i < comm_size;i++)
        {
            std::cout << "  -> " << &hostnames[i * 256] << '\n';
        }

        std::vector<std::string> unique_names;
        for(int i = 0;i < comm_size;i++)
        {
            std::string curr_name = &hostnames[i * 256];
            auto name_loc = std::find(unique_names.begin(), unique_names.end(), curr_name);
            if(name_loc == unique_names.end())
            {
                node_ids[i] = unique_names.size();
                unique_names.push_back(curr_name);
            }
            else
            {
                node_ids[i] = name_loc - unique_names.begin();
            }
        }
        std::cout << "Main rank determined unique hostnames:\n";
        for(const auto& hname : unique_names)
        {
            std::cout << "  -> " << hname << '\n';
        }
    }

    MPI_Scatter(node_ids, 1, MPI_INTEGER, &node_id, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    
    for(int i = 0;i < comm_size;i++)
    {
        if(i == comm_rank)
        {
            std::cout << "Rank #" << std::setw(2) << i << " -- host_id: " << node_id << std::endl; 
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Comm NODE_COMM; 
    MPI_Comm_split(MPI_COMM_WORLD, node_id, 0, &NODE_COMM);

    int node_rank;
    int node_size;
    
    MPI_Comm_rank(NODE_COMM, &node_rank);
    MPI_Comm_size(NODE_COMM, &node_size);

    MPI_Comm INTER_NODE_COMM;
    int inter_node_rank = -1;
    int inter_node_size = -1;
    if(node_rank == 0)
    {
        MPI_Comm_split(MPI_COMM_WORLD, 0, 0, &INTER_NODE_COMM);
        MPI_Comm_rank(INTER_NODE_COMM, &inter_node_rank);
        MPI_Comm_size(INTER_NODE_COMM, &inter_node_size);
    }
    else
    {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 0, &INTER_NODE_COMM);
    }

    if(comm_rank == 0)std::cout << "Created inter-node communicator of size " << inter_node_size << std::endl; 
    MPI_Barrier(MPI_COMM_WORLD);

    for(int i = 0;i < inter_node_size;i++)
    {
        if(node_rank == 0 && i == inter_node_rank)
        {
            std::cout << "Rank #" << std::setw(2) << comm_rank 
                      << " is root of node " << node_id 
                      << " with size = " << node_size
                      << ", and has inter-node rank " << inter_node_rank 
                      << std::endl; 
        }
        MPI_Barrier(INTER_NODE_COMM);
    }
    
    hwloc_topology_t topology;
    if(hwloc_topology_init(&topology) != 0)
    {
        std::cout << "[ERROR] Could not initiliaze HWLOC topology\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if(hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT) != 0)
    {
        std::cout << "[ERROR] Could not enable IO device filter\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if(hwloc_topology_load(topology) != 0)
    {
        std::cout << "[ERROR] Failed to load HWLOC topology\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    hwloc_obj_t topo_root = hwloc_get_root_obj(topology);
    if(comm_rank == 0)
    {
        char cpuset_str[256];
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, topo_root->cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "Machine mask" 
                  << " -- " << std::setfill('0') << std::setw(32) << &cpuset_str[2] << std::endl; 
    } 
    
    int allocation_size_guess = hwloc_bitmap_nr_ulongs(topo_root->cpuset);
    if(allocation_size_guess < 0)
    {
        std::cout << "[ERROR] Failed to query allocation_size_guess from topo_root (i.e. Machine description)" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    if(hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD))
    {
        std::cout << "[ERROR] HWLOC failed to query cpubind for current thread!\n";
        hwloc_bitmap_free(cpuset);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    for(int i = 0;i < comm_size;i++)
    {
        char cpuset_str[256];
        if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, cpuset) == -1)
        {
            std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if(i == comm_rank)
        {
            std::cout << "Rank #" << std::setw(2) << i << " -- " << std::setfill('0') << std::setw(32) << &cpuset_str[2] << std::endl; 
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    unsigned long* bitmap_raw = new unsigned long[allocation_size_guess + 1];
    for(int i = 0;i < allocation_size_guess;i++)bitmap_raw[i] = 0;
    unsigned long* bitmaps_raw_on_node;
    if(node_rank == 0)bitmaps_raw_on_node = new unsigned long[node_size * (allocation_size_guess + 1)];
    int cpuset_size = hwloc_bitmap_nr_ulongs(cpuset);
    if(cpuset_size < 0 || cpuset_size > allocation_size_guess)
    {
        if(cpuset_size < 0)
            std::cout << "[ERROR] Failed to query cpuset_size!" << std::endl;
        else
            std::cout << "[ERROR] cpuset_size @ rank " << comm_rank 
                      << " is " << cpuset_size 
                      << " but allocation_size_guess = " << allocation_size_guess
                      << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    hwloc_bitmap_to_ulongs(cpuset, cpuset_size, bitmap_raw);
    bitmap_raw[allocation_size_guess] = static_cast<unsigned long>(cpuset_size);
    
    MPI_Gather(bitmap_raw,          allocation_size_guess + 1, MPI_UNSIGNED_LONG, 
               bitmaps_raw_on_node, allocation_size_guess + 1, MPI_UNSIGNED_LONG, 0, NODE_COMM);

    if(node_rank == 0)
    {
        hwloc_bitmap_t* cpusets_on_node = new hwloc_bitmap_t[node_size];
        for(int i = 0;i < node_size;i++)
        {
            cpusets_on_node[i] = hwloc_bitmap_alloc();
            unsigned long curr_cpuset_size = bitmaps_raw_on_node[i * (allocation_size_guess + 1) + allocation_size_guess];
            unsigned long* curr_cpuset_longs = &bitmaps_raw_on_node[i * (allocation_size_guess + 1)];

            // std::cout << "Node " << inter_node_rank << ": Processing cpuset " << i << " of size " << curr_cpuset_size << std::endl;

            if(hwloc_bitmap_from_ulongs(cpusets_on_node[i], curr_cpuset_size, curr_cpuset_longs))
            {
                std::cout << "[ERROR] Node " << inter_node_rank 
                          << ": Failed to initialize cpuset #" << i 
                          << " from raw data!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }    
        for(int i = 0;i < inter_node_size;i++)
        {
            if(i == inter_node_rank)
            {
                for(int j = 0;j < node_size;j++)
                {
                    char cpuset_str[256];
                    if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, cpusets_on_node[j]) == -1)
                    {
                        std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    std::cout << "Node " << i 
                              << ": Rank #" << std::setw(2) << i 
                              << " -- " << std::setfill('0') << std::setw(32) << &cpuset_str[2] << std::endl; 
                }
            }
            MPI_Barrier(INTER_NODE_COMM);
        }
        for(int i = 0;i < node_size;i++)
        {
            for(int j = i + 1;j < node_size;j++)
            {
                if(hwloc_bitmap_intersects(cpusets_on_node[i], cpusets_on_node[j]))
                {
                    std::cout << "[ERROR] Node " << inter_node_rank 
                              << ": cpusets of rank " << i 
                              << " and rank " << j 
                              << " intersect!" << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }

        decltype(discover_accelerator<ACC_TYPE::XRT>(std::declval<std::string>())) accs;
        std::string hostname_str(hostname);
        if(hostname_str.find("n2hacc") != std::string::npos)
        {
            auto vcks   = discover_accelerator<ACC_TYPE::XRT>("VCK5000");
            auto u55cs  = discover_accelerator<ACC_TYPE::XRT>("U55C");
            auto mi210s = discover_accelerator<ACC_TYPE::HIP>("MI210");
            accs.reserve(vcks.size() + u55cs.size() + mi210s.size());
            accs.insert(accs.end(), vcks.begin(),   vcks.end());
            accs.insert(accs.end(), u55cs.begin(),  u55cs.end());
            accs.insert(accs.end(), mi210s.begin(), mi210s.end());
        }
        else if(hostname_str.find("n2fpga") != std::string::npos)
        {
            accs = discover_accelerator<ACC_TYPE::XRT>("U280");
        }
        else if(hostname_str.find("n2gpu") != std::string::npos)
        {
            accs = discover_accelerator<ACC_TYPE::CUDA>("A100");
        }
        for(const auto& acc : accs)
        {
            hwloc_cpuset_t acc_cpuset = get_accelerator_device_cpuset(topology, acc.second);
            char acc_cpuset_str[256];
            if(hwloc_bitmap_taskset_snprintf(acc_cpuset_str, 256, acc_cpuset) == -1)
            {
                std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::vector<int> host_candidates;
            for(int i = 0;i < node_size;i++)
            {
                if(hwloc_bitmap_isincluded(cpusets_on_node[i], acc_cpuset))
                {
                    host_candidates.push_back(i);
                }
            }
            if(host_candidates.size() == 0)
            {
                std::cout << "[ERROR] Node " << inter_node_rank 
                          << " -- Accelerator " << acc.first 
                          << " with BDF " << acc.second
                          << " can not be attached to any rank on node because its cpuset "
                          << std::setfill('0') << std::setw(32) << &acc_cpuset_str[2]
                          << " does not include any of the rank cpusets!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            else
            {
                std::stringstream ss_hc;
                for(const auto& hc : host_candidates)
                {
                    char cpuset_str[256];
                    if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, cpusets_on_node[hc]) == -1)
                    {
                        std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf" << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    ss_hc << "  -> " << " rank " << hc 
                          << " -- cpuset: " << std::setfill('0') << std::setw(32) << &cpuset_str[2] << '\n';
                }
                std::cout << "Node " << inter_node_rank 
                          << " -- Accelerator " << acc.first 
                          << " with BDF " << acc.second
                          << " and cpuset " << std::setfill('0') << std::setw(32) << &acc_cpuset_str[2]
                          << " can be attached to the following ranks: \n" 
                          << ss_hc.str() << std::endl;
            }
        }

        for(int i = 0;i < node_size;i++)
        {
            hwloc_bitmap_free(cpusets_on_node[i]);
        }
        delete[] cpusets_on_node;
    }  
    
    delete[] bitmap_raw;
    if(node_rank == 0)
    {
        delete[] bitmaps_raw_on_node;
    }

    hwloc_bitmap_free(cpuset);
    hwloc_topology_destroy(topology);

    if(comm_rank == 0)
    {
        delete[] hostnames;
        delete[] node_ids;
    }

    MPI_Finalize();
    return 0;
}
