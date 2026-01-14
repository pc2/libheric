#include "libheric.h"

#include <stdlib.h>

#include <iostream>

//Forward declare helper functions
namespace libheric
{
    std::vector<libheric::acc_kind> unpack_kinds(char*, int);
    std::vector<libheric::acc_device> unpack_devices(char*, int);
    void pack_devices(const std::vector<libheric::acc_device>&, char*);
}
//FORTRAN Interface
extern "C" {

void get_accelerator_devices(char* packed_kinds, int num_kinds, char* packed_devices, int* num_devices)
{
    auto kinds = libheric::unpack_kinds(packed_kinds, num_kinds);
    auto devices = libheric::discover_accelerators(kinds);
    *num_devices = devices.size();
    libheric::pack_devices(devices, packed_devices);
}

void alloc_accelerator_context(void** ctx, 
                               char* packed_device, 
                               double* raw_cell_mat, 
                               double* raw_positions,   
                               int* raw_kinds,           int num_positions,
                               double* raw_exponents,    int num_exponents,
                               double* raw_coefficients, int num_shells,
                               int* raw_kind_offsets,  
                               int* raw_kind_sizes,      int num_kinds,
                               int* raw_set_offsets,     
                               int* raw_set_sizes,       int max_sets,
                               int* raw_shell_offsets,
                               int* raw_shell_sizes)
{
    auto dev = libheric::unpack_devices(packed_device, 1)[0];
    if(libheric::alloc_acc_context_picker.contains(dev.kind))
    {
        Eigen::Map<Eigen::Matrix3d>  cell_mat(raw_cell_mat);
        Eigen::Map<Eigen::Matrix3Xd> positions(raw_positions, 3, num_positions);
        Eigen::Map<Eigen::VectorXi>  kinds(raw_kinds, num_positions);
        Eigen::Map<Eigen::VectorXd>  exponents(raw_exponents, num_exponents);
        Eigen::Map<Eigen::MatrixXd>  coefficients(raw_coefficients, num_exponents, num_shells);
        Eigen::Map<Eigen::VectorXi>  kind_offsets(raw_kind_offsets, num_kinds);
        Eigen::Map<Eigen::VectorXi>  kind_sizes(raw_kind_sizes, num_kinds);
        Eigen::Map<Eigen::MatrixXi>  set_offsets(raw_set_offsets, max_sets, num_kinds);
        Eigen::Map<Eigen::MatrixXi>  set_sizes(raw_set_sizes, max_sets, num_kinds);
        Eigen::Map<Eigen::Vector<int, libheric::MAX_AM + 1>> shell_offsets(raw_shell_offsets);
        Eigen::Map<Eigen::Vector<int, libheric::MAX_AM + 1>> shell_sizes(raw_shell_sizes);

        *ctx = libheric::alloc_acc_context_picker[dev.kind](
                dev, 
                libheric::simulation_box(
                    cell_mat, 
                    positions, kinds, 
                    exponents, coefficients,
                    kind_offsets, kind_sizes, 
                    set_offsets, set_sizes,  
                    shell_offsets, shell_sizes
                )
             );
    }
    else
    {
        *ctx = nullptr;
        std::cout << "[ERROR] No allocator found for acc_kind = " << dev.kind << std::endl;
    }
}

void free_accelerator_context(void* ctx)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    if(acc_ctx)delete acc_ctx;
}

void accelerator_context_functional(void* ctx, int* is_functional)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
	*is_functional = acc_ctx->functional();
}

void accelerator_context_set_quartet(void* ctx, int la, int lb, int lc, int ld)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->set_quartet(la, lb, lc, ld); 
}

void accelerator_context_set_quartet_permutation(void* ctx, int perm)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->set_quartet_permutation(perm); 
}

void accelerator_context_set_atom_id(void* ctx, int quartet_pos, int atom_id)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->set_atom_id(quartet_pos, atom_id); 
}

void accelerator_context_set_set_id(void* ctx, int quartet_pos, int set_id)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->set_set_id(quartet_pos, set_id); 
}

void accelerator_context_set_shell_id(void* ctx, int quartet_pos, int shell_id)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->set_shell_id(quartet_pos, shell_id); 
}

void accelerator_context_set_exponent_id(void* ctx, int quartet_pos, int exp_id)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->set_exponent_id(quartet_pos, exp_id); 
}

void accelerator_context_set_offset(void* ctx, int off_pos, int off_x, int off_y, int off_z)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->set_offset(off_pos, off_x, off_y, off_z); 
}

void accelerator_context_submit_peri(void* ctx)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->submit_peri(); 
}

void accelerator_context_submit_ceri(void* ctx)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->submit_ceri(); 
}

void accelerator_context_finish_quartet_class(void* ctx)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->finish_quartet_class(); 
}

void accelerator_context_get_next_ceri(void* ctx, double* integral_buffer)
{
    libheric::acc_context* acc_ctx = (libheric::acc_context*)ctx;
    acc_ctx->get_next_ceri(integral_buffer); 
}

void pc2_hide_acl()
{
    setenv("HIDEACL", "1", true);
}

}

namespace libheric 
{
    std::vector<acc_device> discover_accelerators(std::vector<acc_kind> acc_query)
    {
        std::vector<acc_device> accs;
        for(const auto& kind : acc_query)
        {
            std::cout << "Searching for " << kind.name 
                      << " of type " << kind.type 
                      << " configured with " << kind.library << " library\n";
            if(kind.name == "CPU" || kind.type == "CPU")
            {
                std::cout << "[WARNING] CPU accelerator query is implicit, ignoring explicit query\n";
            }
            if(discover_acc_kind.contains(kind))
            {
                auto new_accs = discover_acc_kind[kind](kind);
                accs.insert(accs.begin(), new_accs.begin(), new_accs.end());
            }
        }
        acc_kind cpu_kind = {"CPU", "CPU", "libint"};
        auto cpu_acc = discover_acc_kind[cpu_kind](cpu_kind);
        if(cpu_acc.size() == 1)
        {
            for(auto& acc : accs)
            {
                acc.cpu_name  = cpu_acc[0].cpu_name;
                acc.num_cores = cpu_acc[0].num_cores;
            }
            accs.push_back(cpu_acc[0]);
        }
        else
        {
            std::cout << "[WARNING] no CPU accelerator/host found!\n";
        }  
        return accs;
    }

    std::vector<acc_kind> unpack_kinds(char* packed_kinds, int num_kinds)
    {
        std::vector<acc_kind> kinds;
        for(int i = 0;i < num_kinds;i++)
        {
            int name_off = (i * 3 + 0) * FSTR_LEN;
            int type_off = (i * 3 + 1) * FSTR_LEN;
            int lib_off  = (i * 3 + 2) * FSTR_LEN;
            kinds.push_back({&packed_kinds[name_off], &packed_kinds[type_off], &packed_kinds[lib_off]});
        }
        return kinds;
    }

    std::vector<acc_device> unpack_devices(char* packed_devices, int num_devices)
    {
        std::vector<acc_device> devices;
        for(int i = 0;i < num_devices;i++)
        {
            int dev_name_off  = (i * 7 + 0) * FSTR_LEN;
            int dev_bdf_off   = (i * 7 + 1) * FSTR_LEN;
            int dev_cpu_off   = (i * 7 + 2) * FSTR_LEN;
            int dev_cores_off = (i * 7 + 3) * FSTR_LEN;
            int kind_name_off = (i * 7 + 4) * FSTR_LEN;
            int kind_type_off = (i * 7 + 5) * FSTR_LEN;
            int kind_lib_off  = (i * 7 + 6) * FSTR_LEN;

            int num_cores = std::stoi(&packed_devices[dev_cores_off]); 
            devices.push_back({
                &packed_devices[dev_name_off], 
                &packed_devices[dev_bdf_off], 
                &packed_devices[dev_cpu_off],
                num_cores,
                {
                    &packed_devices[kind_name_off],
                    &packed_devices[kind_type_off],
                    &packed_devices[kind_lib_off]
                }
            });
        }
        return devices;
    }

    void pack_devices(const std::vector<acc_device>& devices, char* packed_devices)
    {
        int store_devices = std::min(devices.size(), static_cast<size_t>(MAX_DEVICES_ON_NODE));
        for(int i = 0;i < store_devices;i++)
        {
            auto dev = devices[i];
            int dev_name_off  = (i * 7 + 0) * FSTR_LEN;
            int dev_bdf_off   = (i * 7 + 1) * FSTR_LEN;
            int dev_cpu_off   = (i * 7 + 2) * FSTR_LEN;
            int dev_cores_off = (i * 7 + 3) * FSTR_LEN;
            int kind_name_off = (i * 7 + 4) * FSTR_LEN;
            int kind_type_off = (i * 7 + 5) * FSTR_LEN;
            int kind_lib_off  = (i * 7 + 6) * FSTR_LEN;

            // Guarantee that last character is a space (given that packed_devices is initially all spaces)
            for(size_t j = 0;j < static_cast<size_t>(FSTR_LEN - 1);j++)
            {
                std::string num_cores_str = std::to_string(dev.num_cores);

                packed_devices[dev_name_off  + j] = (j < dev.name.length()         ? dev.name[j]         : ' ');
                packed_devices[dev_bdf_off   + j] = (j < dev.bdf.length()          ? dev.bdf[j]          : ' ');
                packed_devices[dev_cpu_off   + j] = (j < dev.cpu_name.length()     ? dev.cpu_name[j]     : ' ');
                packed_devices[dev_cores_off + j] = (j < num_cores_str.length()    ? num_cores_str[j]    : ' ');
                packed_devices[kind_name_off + j] = (j < dev.kind.name.length()    ? dev.kind.name[j]    : ' ');
                packed_devices[kind_type_off + j] = (j < dev.kind.type.length()    ? dev.kind.type[j]    : ' ');
                packed_devices[kind_lib_off  + j] = (j < dev.kind.library.length() ? dev.kind.library[j] : ' ');
            }

        }
    }
}

