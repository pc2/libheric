#include <libheric.h>
#include <globals.h>
#include <logging.h>

#include <stdlib.h>

#include <iostream>

#include <libint2.h>

//Forward declare helper functions
namespace libheric
{
    namespace globals {
        uint32_t mpi_rank = 0;
    }
    bool initialize(uint32_t mpi_rank)
    {
        globals::mpi_rank = mpi_rank;
        // Explicitly initialize all build_eri entries for later sanity check
        for(uint32_t am1 = 0;am1 < MAX_AM;am1++)
        for(uint32_t am2 = 0;am2 < MAX_AM;am2++)
        for(uint32_t am3 = 0;am3 < MAX_AM;am3++)
        for(uint32_t am4 = 0;am4 < MAX_AM;am4++)
            libheric_libint2_build_eri[am1][am2][am3][am4] = nullptr;
        libheric_libint2_static_init();
        return initialize_loggers();
    }

    void release()
    {
        release_loggers();
        libheric_libint2_static_cleanup();
    }

    std::vector<libheric::acc_kind> unpack_kinds(char*, int);
    std::vector<libheric::acc_device> unpack_devices(char*, int);
    void pack_devices(const std::vector<libheric::acc_device>&, char*);
    static std::map<acc_kind, std::function<std::vector<acc_device>(const acc_kind&)>> 
        discover_acc_kind = {
            {{"A100","CUDA","libintx"}, discover_cuda_accelerators},
            {{"MI210","HIP","libintx"}, discover_hip_accelerators},
            {{"P520","ONEAPI","intel_eris"}, discover_oneapi_accelerators},
            {{"U280","XRT","seri"}, discover_xrt_accelerators},
            {{"VCK5000","XRT","eri_aie"}, discover_xrt_accelerators},
            {{"CPU", "CPU", "libint"}, discover_cpu_accelerator}
        };
    
    template <typename acc_context_t> 
    abstract_acc_context* alloc_acc_context(const acc_device& acc_dev, 
                                            const simulation_box& sim_box,
                                            uint32_t ctx_id,
                                            uint64_t mem_available,
                                            bool compute_raw_cartesian,
                                            abstract_acc_context* master_ctx)
    {
        static_assert(std::is_base_of_v<abstract_acc_context, acc_context_t>, 
                     "Cannot allocate acc_context_t which is not derived from abstract_acc_context!");
        abstract_acc_context* ctx = new acc_context_t(acc_dev, sim_box, ctx_id, mem_available, 
                                                      compute_raw_cartesian, master_ctx);
        return ctx;
    }

    static std::map<acc_kind, std::function<abstract_acc_context*(const acc_device& acc_dev, 
                                                                  const simulation_box& sim_box,
                                                                  uint32_t ctx_id,
                                                                  uint64_t mem_available, 
                                                                  bool compute_raw_cartesian,
                                                                  abstract_acc_context* master_ctx)>> 
        alloc_acc_context_picker = {
            {{"A100","CUDA","libintx"}, alloc_acc_context<cuda_acc_context>},
            {{"MI210","HIP","libintx"}, alloc_acc_context<hip_acc_context>},
            {{"P520","ONEAPI","intel_eris"}, alloc_acc_context<oneapi_acc_context>},
            {{"U280","XRT","seri"}, alloc_acc_context<xrt_acc_context>},
            {{"VCK5000","XRT","eri_aie"}, alloc_acc_context<xrt_acc_context>},
            {{"CPU", "CPU", "libint"}, alloc_acc_context<cpu_acc_context>}
        };
    // The keys match the CP2K constants do_potential_* in input_constants.F
    static std::map<int, potential_operator> op_table = {{1, potential_operator::coulomb}, {2, potential_operator::short_range}};
}
//FORTRAN Interface
extern "C" {

bool libheric_initialize(int mpi_rank)
{
    return libheric::initialize(mpi_rank);
}

void libheric_release()
{
    libheric::release();
}

void get_accelerator_devices(char* packed_kinds, int num_kinds, char* packed_devices, int* num_devices)
{
    auto kinds = libheric::unpack_kinds(packed_kinds, num_kinds);
    auto devices = libheric::discover_accelerators(kinds);
    *num_devices = devices.size();
    libheric::pack_devices(devices, packed_devices);
}

void alloc_accelerator_contexts(void** ctx, 
                                char* packed_device, 
                                int num_contexts,
                                long long int mem_available,
                                bool compute_raw_cartesian,
                                double* raw_cell_mat, 
                                double* raw_positions,   
                                int* raw_kinds,           int num_positions,
                                double* raw_exponents,    int num_exponents,
                                double* raw_coefficients, int num_coefficients,
                                int* raw_kind_nset,       int num_kinds, int l_max,
                                int* raw_set_nexp,        int max_sets,
                                int* raw_set_nshl)
{
    auto dev = libheric::unpack_devices(packed_device, 1)[0];
    if(libheric::alloc_acc_context_picker.contains(dev.kind))
    {
        Eigen::Map<Eigen::Matrix3d>  cell_mat(raw_cell_mat);
        Eigen::Map<Eigen::Matrix3Xd> positions(raw_positions, 3, num_positions);
        std::vector<int>    kinds(raw_kinds, raw_kinds + num_positions);
        std::vector<double> exponents(raw_exponents, raw_exponents + num_exponents);
        std::vector<double> coefficients(raw_coefficients, raw_coefficients + num_coefficients);
        std::vector<int>    kind_nset(raw_kind_nset, raw_kind_nset + num_kinds * (l_max + 1)); 
        std::vector<int>    set_nexp(raw_set_nexp, raw_set_nexp + max_sets * num_kinds * (l_max + 1)); 
        std::vector<int>    set_nshl(raw_set_nshl, raw_set_nshl + max_sets * num_kinds * (l_max + 1)); 

        auto sim_box = libheric::simulation_box(
                                                   cell_mat, 
                                                   positions, kinds, 
                                                   exponents, coefficients,
                                                   kind_nset, set_nexp, set_nshl,
                                                   max_sets, num_kinds, l_max, num_positions
                                               );
        *ctx = libheric::alloc_acc_context_picker[dev.kind](dev, sim_box, 0, mem_available, 
                                                            compute_raw_cartesian, nullptr);
        libheric::abstract_acc_context* master_ref = (libheric::abstract_acc_context*)*ctx;
        for(int i = 1;i < num_contexts;i++)
        {
            ctx++;
            *ctx = libheric::alloc_acc_context_picker[dev.kind](dev, sim_box, i, mem_available, 
                                                                compute_raw_cartesian, master_ref); 
        }
    }
    else
    {
        *ctx = nullptr;
        std::cout << "[ERROR] No allocator found for acc_kind = " << dev.kind << std::endl;
    }
}

void free_accelerator_contexts(void** ctx, int num_contexts)
{
    for(int i = num_contexts - 1; i >= 0;i--)
    {
        libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx[i];
        if(acc_ctx)delete acc_ctx;
    }
}

bool accelerator_context_is_functional(void* ctx)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
	return acc_ctx->functional();
}

void accelerator_context_set_potential_operator(void* ctx, int op_id, double op_param)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    libheric::potential_operator op = libheric::op_table.contains(op_id) ? libheric::op_table[op_id] : libheric::potential_operator::invalid; 
    acc_ctx->set_potential_operator(op, op_param); 
}

void accelerator_context_set_quartet_class(void* ctx, uint32_t la, uint32_t lb, uint32_t lc, uint32_t ld)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->set_quartet_class(la, lb, lc, ld); 
}

void accelerator_context_set_atom_ids(void* ctx, int atom_a, int atom_b, int atom_c, int atom_d)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->set_atom_ids(atom_a, atom_b, atom_c, atom_d); 
}

void accelerator_context_set_B_offset(void* ctx, int off_x, int off_y, int off_z)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->set_B_offset(off_x, off_y, off_z); 
}

void accelerator_context_set_D_offset(void* ctx, int off_x, int off_y, int off_z)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->set_D_offset(off_x, off_y, off_z); 
}

void accelerator_context_set_Q_offset(void* ctx, int off_x, int off_y, int off_z)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->set_Q_offset(off_x, off_y, off_z); 
}

void accelerator_context_set_set_ids(void* ctx, int set_a, int set_b, int set_c, int set_d, double pmax)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->set_set_ids(set_a, set_b, set_c, set_d, pmax); 
}

void accelerator_context_set_shell_ids(void* ctx, int shl_a, int shl_b, int shl_c, int shl_d)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->set_shell_ids(shl_a, shl_b, shl_c, shl_d); 
}

void accelerator_context_set_exponent_ids(void* ctx, int exp_a, int exp_b, int exp_c, int exp_d)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->set_exponent_ids(exp_a, exp_b, exp_c, exp_d); 
}

void accelerator_context_submit_peri(void* ctx)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->submit_peri(); 
}

void accelerator_context_finish_quartet_class(void* ctx)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    acc_ctx->finish_quartet_class(); 
}

bool internal_accelerator_context_get_next_staged_atom_quartet(void* ctx, int* quartet_id, void** integral_buffer, 
                                                               void** shellset_ids, int64_t* n_shellset_quartets, 
                                                               int64_t* eris_allocated_per_shellset)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    if constexpr(libheric::DEBUG_PRINT)
    {
        auto logger = libheric::get_logger();
        *logger << "Requesting next quartet\n";
    }
    libheric::abstract_acc_context::staging_index* quartet = acc_ctx->get_next_staged_atom_quartet();
    if(quartet)
    {
        std::copy(quartet->key.begin(), quartet->key.end(), quartet_id);
        *integral_buffer = quartet->value.first_shellset_data_ptr;
        *shellset_ids = quartet->value.first_shellset_ptr;
        *n_shellset_quartets = (int64_t)quartet->value.n_shellset_quartets;
        *eris_allocated_per_shellset = (int64_t)quartet->eris_allocated_per_shellset;
            
        return true;
    }
    else return false;

}

bool accelerator_context_prefetch_cached_atom_quartet(void* ctx, int A, int B, int C, int D)
{
    libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
    return acc_ctx->prefetch_cached_atom_quartet(A, B, C, D);
}

void accelerator_context_update_positions(void* ctx, double* raw_positions, int num_positions)
{
   libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
   Eigen::Map<Eigen::Matrix3Xd> positions(raw_positions, 3, num_positions);
   acc_ctx->update_positions(positions);
}

void accelerator_context_update_cell(void* ctx, double* raw_cell_mat)
{
   libheric::abstract_acc_context* acc_ctx = (libheric::abstract_acc_context*)ctx;
   Eigen::Map<Eigen::Matrix3d> cell_mat(raw_cell_mat);
   acc_ctx->update_cell(cell_mat);
}

void compute_single_quartet(int op_id, double op_param, uint32_t la, uint32_t lb, uint32_t lc, uint32_t ld,
                            double* pos_A, double* pos_B, double* pos_C, double* pos_D, 
                            double alpha, double beta, double gamma, double delta, double* quartet)
{
    Eigen::Map<Eigen::Vector3d> A(pos_A, 3), B(pos_B, 3), C(pos_C, 3), D(pos_D, 3);
    auto cell_mat = Eigen::Matrix3d::Zero();
    Eigen::Matrix3Xd positions(3, 4);
    for(uint32_t i = 0;i < 3;i++)
    {
        positions(i, 0) = pos_A[i];
        positions(i, 1) = pos_B[i];
        positions(i, 2) = pos_C[i];
        positions(i, 3) = pos_D[i];
    }
    std::vector<int> kinds({0, 1, 2, 3});
    std::vector<double> exponents;
    std::vector<double> coefficients;
    std::vector<int> kind_nset;
    std::vector<int> set_nexp;
    std::vector<int> set_nshl;
    double exps[4] = {alpha, beta, gamma, delta};


    for(int l = 0;l <= libheric::MAX_AM;l++)
    {
        for(int i = 0;i < 4;i++)
        {
            kind_nset.push_back(1);
            exponents.push_back(exps[i]);
            coefficients.push_back(1.0);
            set_nexp.push_back(1);
            set_nshl.push_back(1);
        }
    }

    libheric::simulation_box sim_box(cell_mat, positions, kinds, exponents, coefficients,
                                     kind_nset, set_nexp, set_nshl, 1, 4, libheric::MAX_AM, 4);
    libheric::acc_device dev;
    libheric::cpu_acc_context* acc = new libheric::cpu_acc_context(dev, sim_box, -1, 128 * 1024 * 1024, false, nullptr);
    accelerator_context_set_potential_operator(acc, op_id, op_param); 
    accelerator_context_set_quartet_class(acc, la, lb, lc, ld);
    accelerator_context_set_atom_ids(acc, 0u, 1u, 2u, 3u);
    accelerator_context_set_set_ids(acc, 0u, 0u, 0u, 0u, 0.0); 
    accelerator_context_set_shell_ids(acc, 0u, 0u, 0u, 0u); 
    accelerator_context_set_B_offset(acc, 0u, 0u, 0u); 
    accelerator_context_set_D_offset(acc, 0u, 0u, 0u); 
    accelerator_context_set_exponent_ids(acc, 0u, 0u, 0u, 0u); 
    accelerator_context_set_Q_offset(acc, 0u, 0u, 0u); 
    accelerator_context_submit_peri(acc);
    accelerator_context_finish_quartet_class(acc);

    int quartet_id[4];
    double* integral_buffer;
    uint8_t* shellset_ids;
    int64_t n_shellset_quartets, eris_allocated_per_shellset;
    internal_accelerator_context_get_next_staged_atom_quartet(acc, quartet_id, (void**)&integral_buffer, (void**)&shellset_ids, 
                                                              &n_shellset_quartets, &eris_allocated_per_shellset);
    size_t n_eris = libheric::nso[la] * libheric::nso[lb] * libheric::nso[lc] * libheric::nso[ld];
    std::copy(integral_buffer, integral_buffer + n_eris, quartet);
    delete acc; 
}

void flush_logger()
{
    std::ostream* logger = libheric::get_logger();
    *logger << std::flush;
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
            if constexpr(LOG_PRINT)
            {
                auto logger = get_logger();
                *logger << "Searching for " << kind.name 
                        << " of type " << kind.type 
                        << " configured with " << kind.library << " library\n";
                if(kind.name == "CPU" || kind.type == "CPU")
                {
                    *logger << "[WARNING] CPU accelerator query is implicit, ignoring explicit query\n";
                }
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

            //std::cout << &packed_devices[dev_name_off] << std::endl;
            //std::cout << &packed_devices[dev_bdf_off] << std::endl;
            //std::cout << &packed_devices[dev_cpu_off] << std::endl;
            //std::cout << &packed_devices[dev_cores_off] << std::endl;
            //std::cout << &packed_devices[kind_name_off] << std::endl;
            //std::cout << &packed_devices[kind_type_off] << std::endl;
            //std::cout << &packed_devices[kind_lib_off] << std::endl;

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

