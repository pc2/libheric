
#include <iostream>
#include <string>

#include <hwloc.h>

void explore_io_tree(hwloc_obj_t curr, int depth)
{
    std::string tab(depth * 2, ' ');
    if(hwloc_obj_type_is_io(curr->type) == 0)
    {
        std::cout << tab << "IO at depth " << curr->depth << " with io_arity = " << curr->io_arity << " and normal arity = " << curr->arity << " is faulty! (enum_val = " << curr->type << ")\n";
        if(hwloc_compare_types(curr->type, HWLOC_OBJ_MISC) == 0)std::cout << tab << " -> It is actually of type MISC!\n";
        if(hwloc_compare_types(curr->type, HWLOC_OBJ_BRIDGE) == 0)std::cout << tab << " -> It is actually of type BRIDGE!\n";
        if(hwloc_compare_types(curr->type, HWLOC_OBJ_GROUP) == 0)std::cout << tab << " -> It is actually of type GROUP!\n";
        if(hwloc_compare_types(curr->type, HWLOC_OBJ_PCI_DEVICE) == 0)std::cout << tab << " -> It is actually of type PCI_DEVICE!\n";
        if(hwloc_compare_types(curr->type, HWLOC_OBJ_OS_DEVICE) == 0)std::cout << tab << " -> It is actually of type OS_DEVICE!\n";
        return;
    }
    std::cout << tab << "IO at depth " << curr->depth << " with io_arity = " << curr->io_arity << " and normal arity = " << curr->arity <<'\n';
    for(int i = 0;i < curr->infos_count;i++)
    {
        std::cout << tab << ' ' << curr->infos[i].name << " : " << curr->infos[i].value << '\n';
    }

    if(curr->io_arity == 0)return;

    
    curr = curr->io_first_child;
    explore_io_tree(curr, depth + 1);
    while(curr->next_sibling != NULL)
    {
        curr = curr->next_sibling;
        explore_io_tree(curr, depth + 1);
    }
}

void explore_obj_tree(hwloc_obj_t curr)
{
    //std::cout << "Current obj depth: " << curr->depth << '\n';
    //std::cout << "   Current obj arity: " << curr->arity << '\n';
    //std::cout << "   Current obj IO-arity: " << curr->io_arity << '\n';
    
    if(curr->io_arity > 0)
    {
        std::cout << "Object at depth " << curr->depth << " has non-zero IO-arity!\n";
        explore_io_tree(curr->io_first_child, 1);
    }
    
    for(int i = 0;i < curr->arity;i++)
    {
        explore_obj_tree(curr->children[i]);
    }
}

int main()
{
    hwloc_topology_t topo;
    if(hwloc_topology_init(&topo) != 0)
    {
        std::cout << "Failed to initialize HWLOC topology\n";
        exit(1);
    }

    if(hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_IMPORTANT) != 0)
    {
        std::cout << "Failed to enable IO device filter\n";
        exit(1);
    }
    
    if(hwloc_topology_load(topo) != 0)
    {
        std::cout << "Failed to load HWLOC topology\n";
        exit(1);
    }

    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    if(hwloc_get_cpubind(topo, cpuset, 0) != 0)
    {
        std::cout << "Failed to get cpubind of current thread\n";
        exit(1);
    }

    char cpuset_str[256];
    if(hwloc_bitmap_taskset_snprintf(cpuset_str, 256, cpuset) == -1)
    {
        std::cout << "Failed to execute hwloc_bitmap_taskset_snprintf\n";
        exit(1);
    }

    std::cout << "cputset of current thread: " << cpuset_str << '\n';
    
    hwloc_obj_t root_obj = hwloc_get_root_obj(topo);
    std::cout << root_obj->infos_count << std::endl;
    for(int i = 0;i < root_obj->infos_count;i++)
    {
        std::cout << root_obj->infos[i].name << " : " << root_obj->infos[i].value << '\n';
    }

    for(int i = 0;i < root_obj->arity;i++)
    {
        std::cout << "Child #" << i << '\n';
        hwloc_obj_t child = root_obj->children[i];
        for(int j = 0;j < child->infos_count;j++)
        {
            std::cout << '\t' << child->infos[j].name << " : " << child->infos[j].value << '\n';
        }

    }

    explore_obj_tree(root_obj);

    hwloc_topology_destroy(topo);

    unsigned int hwloc_api_version = hwloc_get_api_version();
    return hwloc_api_version;
}
