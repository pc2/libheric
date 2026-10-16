#include <oneapi_acc.h>

#include <constants.h>
#include <math_util.h>
#include <transform.h>
#include <util.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <new>
#include <thread>
#include <utility>

#ifdef ENABLE_ONEAPI
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#include <dlfcn.h>

//extern "C" {
//#include <rys_roots.h>
//}


namespace libheric
{


    std::vector<acc_device> discover_oneapi_accelerators(const acc_kind& acc_query) 
    {
        std::vector<acc_device> found_devices;

#ifdef ENABLE_ONEAPI
        sycl::platform platform;
        try {
            platform = sycl::platform(sycl::ext::intel::fpga_selector_v);
        } catch (const sycl::exception& e) {
            std::cout << "[ERROR] Failed to query SYCL platform: " << e.what() << std::endl;
            return {};
        }
        std::string platform_name = platform.get_info<sycl::info::platform::name>();
        if(platform_name.find("Intel(R) FPGA SDK") == std::string::npos) return {};

        std::vector<sycl::device> devices;
        try {
            devices = platform.get_devices();
        } catch (const sycl::exception& e) {
            std::cout << "[ERROR] Failed to query SYCL devices from platform: " << e.what() << std::endl;
            return {};
        }

        for(size_t i = 0;i < devices.size(); i++)
        {
            std::string dev_name = devices[i].get_info<sycl::info::device::name>();
            std::string dev_name_upper = dev_name;
            std::for_each(dev_name_upper.begin(), dev_name_upper.end(), [](char& c){c = std::toupper(c);});
            
            if(dev_name_upper.find(acc_query.name) != std::string::npos)
            {
                std::string fake_dev_bdf = "0000:00:0" + std::to_string(i) + ".0";
                found_devices.push_back({dev_name, fake_dev_bdf, "", 1, acc_query});
            }
        }
#endif
        return found_devices;
    }

#ifdef ENABLE_ONEAPI
    oneapi_acc_context::oneapi_acc_context(const acc_device& acc_dev,
                                           const simulation_box& sim_box,
                                           uint32_t num_contexts,
                                           uint64_t mem_available,
                                           bool compute_raw_cartesian,
                                           abstract_acc_context* master_ctx)
       : abstract_acc_context(acc_dev, sim_box, num_contexts, mem_available, compute_raw_cartesian, master_ctx), 
         lib_handle(nullptr), kernel(nullptr),
		 batch_size(0), cbits(0), n_rtwt(0), n512_per_quartet(0), 
		 host_eri_d{nullptr, nullptr}, host_eri_c{nullptr, nullptr}, host_eps{nullptr, nullptr}, 
		 dev_abcd{nullptr, nullptr}, dev_rtwt{nullptr, nullptr}, dev_eri_c{nullptr, nullptr}, dev_eps{nullptr, nullptr},
         overwrite_quartet_batch{1}, start_consuming{0}
    {
        //this->accelerator_queue = new quartet_task_queue(num_contexts);
        //quartet_tasks.emplace_back(DEFAULT_TASK_SIZE, &task_consumer_callback, &task_consumer_callback_mtx);

        // PC2 specific solution might/should be changed in the future!
        if(acc_dev.bdf != "0000:00:00.0" && acc_dev.bdf != "0000:00:01.0")
        {
            std::cout << "[ERROR] Unexpected device BDF for Intel FPGA: " << acc_dev.bdf << std::endl;
            return;
        }

        std::optional<sycl::platform> platform;
        try {
            platform = sycl::platform(sycl::ext::intel::fpga_selector_v);
        } catch (const sycl::exception& e) {
            std::cout << "[ERROR] Failed to query SYCL platform: " << e.what() << std::endl;
            return;
        }

        std::string platform_name = platform->get_info<sycl::info::platform::name>();
        if(platform_name.find("Intel(R) FPGA SDK") == std::string::npos)
        {
            std::cout << "[ERROR] expected to find Intel(R) FPGA SDK platform but instead got " << platform_name << std::endl;
            return;
        }
        
        std::vector<sycl::device> devices;
        try {
            devices = platform->get_devices();
        } catch (const sycl::exception& e) {
            std::cout << "[ERROR] Failed to query SYCL devices from platform: " << e.what() << std::endl;
            return;
        }
        
        this->dev = devices.back();
        std::string dev_name = this->dev->get_info<sycl::info::device::name>();
        if(dev_name != acc_dev.name)
        {
            std::cout << "[ERROR] allocated device " << dev_name << " is not the requested device " << acc_dev.name << std::endl;
            return;
        }

		if (!this->dev->get_info<sycl::info::device::usm_device_allocations>())
        {
            std::cout << "[ERROR] allocated device " << dev_name << " does not support explicit USM device allocations" << std::endl;
			return;
		}

        this->ctx = sycl::context(this->dev.value());
        this->q = sycl::queue(this->ctx.value(), this->dev.value(), sycl::property::queue::enable_profiling{});
    }
    
    //oneapi_acc_context::oneapi_acc_context(oneapi_acc_context* master_ctx)
    //   : abstract_acc_context(master_ctx->acc_dev, master_ctx->sim_box), 
    //     lib_handle(nullptr), kernel(nullptr),
	//	 batch_size(0), cbits(0), n_rtwt(0), n512_per_quartet(0), 
	//	 host_eri_d{nullptr, nullptr}, host_eri_c{nullptr, nullptr}, host_eps{nullptr, nullptr}, 
	//	 dev_abcd{nullptr, nullptr}, dev_rtwt{nullptr, nullptr}, dev_eri_c{nullptr, nullptr}, dev_eps{nullptr, nullptr},
    //     overwrite_quartet_batch{1}, start_consuming{0},
    //     master_ctx(master_ctx)
    //{

    //}

    oneapi_acc_context::~oneapi_acc_context()
    {
		if(this->lib_handle != nullptr)
		{
            if(dlclose(this->lib_handle) != 0)
                std::cout << "[WARNING] Failed to close open shared object handle!" << std::endl;
		}
        for(int db = 0;db < 2;db++)
        {
            if(host_eri_d[db] != nullptr)delete[] host_eri_d[db];
            if(host_eri_c[db] != nullptr)delete[] host_eri_c[db];
            if(host_eps[db]   != nullptr)delete[] host_eps[db];
            
            if(this->q)
            {
                if(dev_abcd[db]  != nullptr)sycl::free(dev_abcd[db] , this->q.value());
                if(dev_rtwt[db]  != nullptr)sycl::free(dev_rtwt[db] , this->q.value());
                if(dev_eri_c[db] != nullptr)sycl::free(dev_eri_c[db], this->q.value());
                if(dev_eps[db]   != nullptr)sycl::free(dev_eps[db]  , this->q.value());
            }
        }
    }

    void oneapi_acc_context::set_quartet(int la, int lb, int lc, int ld, 
                                         int64_t max_prim_per_contr, 
                                         int64_t max_contr_per_quartet)
    {
        this->new_quartet_class_for_consume = true;
        this->abstract_acc_context::set_quartet(la, lb, lc, ld, max_prim_per_contr, max_contr_per_quartet);
        if(this->is_functional)
        {
            this->is_functional = false;
            std::string base_dir = "/pc2/groups/pc2-mitarbeiter/XXX/results-hpc-prf-erifpga/from-pc2data/XXX/For_YYY/1st_round/";


            std::stringstream ss_abcd_int;
            int abcd_id;
            if(am.is_fpga_canonical())
            {
                abcd_id = this->am.d() + (MAX_AM + 1) * ( this->am.c() + (MAX_AM + 1) * ( this->am.b() + (MAX_AM + 1) * this->am.a())); 
                ss_abcd_int << this->am.a() << this->am.b() << this->am.c() << this->am.d();
            }
            else
            {
                abcd_id = this->am.b() + (MAX_AM + 1) * ( this->am.a() + (MAX_AM + 1) * ( this->am.d() + (MAX_AM + 1) * this->am.c())); 
                ss_abcd_int << this->am.c() << this->am.d() << this->am.a() << this->am.b();
            }
            
            std::stringstream ss_kernel_dir;

            ss_kernel_dir << "t_" << std::setw(3) << std::setfill('0') << abcd_id << "_" << ss_abcd_int.str();
            std::filesystem::path kernel_dir_path(base_dir);
            kernel_dir_path /= ss_kernel_dir.str();
            if(!std::filesystem::exists(kernel_dir_path) || !std::filesystem::is_directory(kernel_dir_path))
            {
                std::cout << "[ERROR] kernel_dir_path = " << kernel_dir_path << " is not a directory!" << std::endl;
                return;
            }

            // Get dir name with random seed
            auto dir_iter = std::filesystem::directory_iterator(kernel_dir_path);
            auto num_entries = std::distance(std::filesystem::begin(dir_iter), std::filesystem::end(dir_iter));
            if(num_entries != 1)
            {
                std::cout << "[ERROR] kernel_dir_path = " << kernel_dir_path << " contains != 1 entries!" << std::endl;
                return;
            }
            dir_iter = std::filesystem::directory_iterator(kernel_dir_path);
            kernel_dir_path = dir_iter->path();
            
            // Get .so library path
            dir_iter = std::filesystem::directory_iterator(kernel_dir_path);
            num_entries = std::distance(std::filesystem::begin(dir_iter), std::filesystem::end(dir_iter));
            if(num_entries != 1)
            {
                std::cout << "[ERROR] kernel_dir_path = " << kernel_dir_path << " contains != 1 entries!" << std::endl;
                return;
            }
            dir_iter = std::filesystem::directory_iterator(kernel_dir_path);
            auto kernel_path = dir_iter->path();

            std::cout << "Discovered kernel: " << kernel_path << std::endl;
            
            if(!std::filesystem::is_regular_file(kernel_path))
            {
                std::cout << "[ERROR] kernel_path = " << kernel_path << " is not a file!" << std::endl;
                return;
            }

            if(kernel_path.extension() != ".so")
            {
                std::cout << "[ERROR] kernel_path = " << kernel_path << " is not a .so file!" << std::endl;
                return;
            }

            auto kernel_name = kernel_path.stem();

            std::vector<std::string> kernel_name_parts;
            std::stringstream ss_kernel_name(kernel_name);
            for(std::string part; std::getline(ss_kernel_name, part, '_');)
                kernel_name_parts.push_back(part);

            std::vector<std::string> expected_parts = {"Rys4Compute", "float", ss_abcd_int.str()};
            if(kernel_name_parts.size() != 6)
            {
                std::cout << "[ERROR] " << kernel_name 
                          << ".so at kernel_path = " << kernel_path 
                          << " has not the expected format: "
                          << expected_parts[0] << "_"
                          << expected_parts[1] << "_"
                          << expected_parts[2] << "_XXX_XXX_XXX.so" << std::endl;
                return;
            }

            std::array<uint32_t, 3> kernel_params;
            for(size_t i = 0;i < kernel_name_parts.size();i++)
            {
                if(i < 3)
                {
                    if(kernel_name_parts[i] != expected_parts[i])
                    {
                      
                        std::cout << "[ERROR] " << kernel_name 
                                  << ".so at kernel_path = " << kernel_path 
                                  << " has not the expected format: "
                                  << expected_parts[0] << "_"
                                  << expected_parts[1] << "_"
                                  << expected_parts[2] << "_XXX_XXX_XXX.so" << std::endl;
                        return;
                    }
                }
                else
                {
                    try
                    {
                        kernel_params[i - 3] = std::stoi(kernel_name_parts[i], nullptr, 10);
                    }
                    catch(std::invalid_argument& e)
                    {
                        std::cout << "[ERROR] Invalid argument while parsing integer param #" << i - 3 
                                  << " for " << kernel_name << ".so -> " << e.what() << std::endl;
                        return;
                    }
                    catch(std::out_of_range& e)
                    {
                        std::cout << "[ERROR] Out of range while parsing integer param #" << i - 3 
                                  << " for " << kernel_name << ".so -> " << e.what() << std::endl;
                        return;
                    }
                }
            }

            this->batch_size = kernel_params[0];
            this->cbits = kernel_params[2];
			uint32_t num_compressed_per_512 = 512 / this->cbits;
			this->n512_per_quartet = (this->am.sph_size() + num_compressed_per_512 - 1) / num_compressed_per_512;			
           
            for(int db = 0;db < 2;db++)
            { 
                this->input_quartets[db].clear();
                this->input_quartets[db].reserve(this->batch_size);
                this->input_rtwt[db].clear();
                this->input_rtwt[db].reserve(this->batch_size);
                this->quartet_coeffs[db].clear();
                this->quartet_coeffs[db].reserve(this->batch_size);
                this->ceri_offsets[db].clear();
                this->ceri_offsets[db].reserve(this->batch_size + 1);
            }
            this->ceri_offsets[0].push_back(0);

            std::cout << "Kernel_params: batch_size = " << this->batch_size << ", #bits for compression = " << this->cbits << std::endl;

            void* lib_handle = dlopen(kernel_path.c_str(), RTLD_NOW);
            if(lib_handle == nullptr)
            {
                std::cout << "[ERROR] Failed to open shared object: " << kernel_path << "!" << std::endl;
                return;
            }

            dlerror(); // Clear error state
            this->kernel = (Rys4Compute_kernel)dlsym(lib_handle, kernel_name.c_str());
            auto dlsym_err = dlerror();
            if(dlsym_err != nullptr)
            {
                std::cout << "[ERROR] while establishing symbolic link to " << kernel_name.c_str() << ": " << dlsym_err << std::endl;
                dlclose(lib_handle);
                return;
            }
				
			n_rtwt = am.total() / 2 + 1;
            
            for(int db = 0;db < 2;db++)
            {
                if(host_eri_d[db] != nullptr) delete[] host_eri_d[db];
                if(host_eri_c[db] != nullptr) delete[] host_eri_c[db];
                if(host_eps[db]   != nullptr) delete[] host_eps[db];

                host_eri_d[db] = new(std::align_val_t(32)) float  [this->batch_size * this->am.sph_size()];
                host_eri_c[db] = new(std::align_val_t(32)) uint8_t[this->batch_size * this->n512_per_quartet * this->n8_per_n512];
                host_eps[db]   = new(std::align_val_t(32)) float  [this->batch_size];
                
                if(this->q)
                {
                    if(dev_abcd[db]  != nullptr) sycl::free(dev_abcd[db] , this->q.value());
                    if(dev_rtwt[db]  != nullptr) sycl::free(dev_rtwt[db] , this->q.value());
                    if(dev_eri_c[db] != nullptr) sycl::free(dev_eri_c[db], this->q.value());
                    if(dev_eps[db]   != nullptr) sycl::free(dev_eps[db]  , this->q.value());

                    dev_abcd[db]  = sycl::malloc_device<float>  (this->batch_size * num_abcd_elems, this->q.value());
                    dev_rtwt[db]  = sycl::malloc_device<float>  (this->batch_size * num_rtwt_elems, this->q.value());
                    dev_eri_c[db] = sycl::malloc_device<uint8_t>(this->batch_size * this->n512_per_quartet * this->n8_per_n512, this->q.value());
                    dev_eps[db]   = sycl::malloc_device<float>  (this->batch_size, this->q.value());
                }
                else
                {
                    std::cout << "[ERROR] queue not initialized in oneapi_acc_context::set_quartet!" << std::endl;
                    return; 
                }
            }
			
            gen_db = 0;
            consume_db = 1;
            ceri_ctr = 0;
            this->new_batch = false;
            this->new_quartet_class_for_gen = true;
            comp_events.clear();
            // Push back two fake completed computation events to trigger new computation events
            comp_events.push_back(sycl::event());
            comp_events.push_back(sycl::event());

            this->is_functional = true;
        }
    }
    
    void oneapi_acc_context::set_quartet_permutation(int perm)
    {
        acc_context::set_quartet_permutation(perm);
        if(perm < 0 || perm >= NUM_PERMS)
        {
            std::cout << "[ERROR] quartet permutation number not in 0 <= " << perm << " < " << NUM_PERMS << "\n";
            this->is_functional = false;
            return;
        }
        this->perm = perm;

        // most quartets that are canonical are also "FPGA canonical", however there are some outliers:
        // e.g. (gs|dp) is canonical but not "FPGA canonical"
        if(am.is_fpga_canonical())
            this->fpga_perm = this->perm;
        else
            this->fpga_perm = perm ^ 0x4;
    }

    void oneapi_acc_context::set_atom_id(int quartet_pos, int atom_id)
    {
        acc_context::set_atom_id(quartet_pos, atom_id);
        Eigen::Vector3d tmp_pos = this->sim_box.positions(Eigen::all, atom_id);      
        this->quartet_tmpl.pos(quartet_pos) = tmp_pos; 
    }
    
    void oneapi_acc_context::set_exponent_id(int quartet_pos, int exp_id)
    {
        acc_context::set_exponent_id(quartet_pos, exp_id);

        this->quartet_tmpl.exp(quartet_pos) = exp_val;
    }
    
    void oneapi_acc_context::set_offset(int off_pos, int off_x, int off_y, int off_z)
    {
        acc_context::set_offset(off_pos, off_x, off_y, off_z);
        this->quartet_off.off(off_pos) = this->sim_box.cell_mat * this->offsets_d(Eigen::all, off_pos);  
    }

    void oneapi_acc_context::submit_peri()
    {
        quartet_input offset_quartet = this->quartet_tmpl;
        offset_quartet += this->quartet_off;

        std::cout << "Primitive #" << quartet_coeffs[gen_db].size() << '\n';
        std::cout << this->quartet_tmpl << '\n';
        std::cout << "Offset quartet:\n";
        std::cout << offset_quartet << '\n';

        auto cpu_rys_quartet = offset_quartet;
        auto fpga_quartet = offset_quartet;

        auto cpu_rys_am = am;

        cpu_rys_quartet.apply_permutation(this->perm);
        cpu_rys_am.apply_permutation(this->perm);
        fpga_quartet.apply_permutation(this->fpga_perm);
        
        auto A = offset_quartet.A();
        auto B = offset_quartet.B();
        auto C = offset_quartet.C();
        auto D = offset_quartet.D();

        double alpha = offset_quartet.alpha();
        double beta  = offset_quartet.beta();
        double gamma = offset_quartet.gamma();
        double delta = offset_quartet.delta();

        auto AB = A - B;
        double AB2 = AB.squaredNorm();
        auto CD = C - D;
        double CD2 = CD.squaredNorm();

        double zeta = alpha + beta;
        double eta  = gamma + delta;

        auto P = (alpha * A + beta * B) / zeta;
        auto Q = (gamma * C + delta * D) / eta;
        auto PQ = P - Q;
        double PQ2 = PQ.squaredNorm();
        

        //std::cout << "AB2 = " << AB2 << '\n';
        //std::cout << "CD2 = " << CD2 << '\n';
        double kc = two_pi_52 * norm_prefactor * norm_coeffs[3]
                  * std::exp(-(alpha * beta * AB2) / zeta - (gamma * delta * CD2) / eta)
                  / (zeta * eta * std::sqrt(zeta + eta));
        //std::cout << "norm_prefactor = " << norm_prefactor << '\n';
        //std::cout << "norm_coeffs = [" << norm_coeffs[0] << ", " << norm_coeffs[1] << ", " << norm_coeffs[2] << ", " << norm_coeffs[3] << "]\n";
        //std::cout << "my_norm = " << norm_prefactor * norm_coeffs[3] << '\n';
        //std::cout << "my_kappa = " << std::exp(-(alpha * beta * AB2) / zeta - (gamma * delta * CD2) / eta) << '\n';
        //std::cout << "my_rho = " << 1.0 / (zeta * eta * std::sqrt(zeta + eta)) << '\n';
        //std::cout << "my_kc = " << kc << '\n'; 
        double rhopq2 = zeta * eta / (zeta + eta) * PQ2; 
        CINTrys_roots(this->n_rtwt, rhopq2, this->rtwt_tmpl.roots(), this->rtwt_tmpl.weights());
        auto cpu_rys_rtwt = rtwt_tmpl;

        for(uint32_t u = 0;u < this->n_rtwt;u++)
        {
            double rt = rtwt_tmpl.roots()[u];
            double wt = rtwt_tmpl.weights()[u];
            rtwt_tmpl.roots()[u] = rt / (1.0 + rt);
            rtwt_tmpl.weights()[u] = std::cbrt(kc * wt);
        }
        
        this->input_quartets[gen_db].push_back(quartet_input<float>(fpga_quartet));
        this->input_rtwt[gen_db].push_back(rtwt_container<float>(rtwt_tmpl));
        this->quartet_coeffs[gen_db].push_back(this->coeff_prod(3));
        if(quartet_coeffs[gen_db].size() == batch_size)
        {
            gen_db = (gen_db + 1) % 2;
            new_batch = true;
        }
        
       
        int32_t lmax = am.max();
        //auto libint_res = libint_oshgp(lmax, am, offset_quartet, this->libint_perm);
        //libint_maxl_ref.push(cart2sph(cpu_rys_am, mnc2uns, libint_res[0]));
        //libint_unit_ref.push(cart2sph(cpu_rys_am, unc2uns, libint_res[1]));
        //libint2_ref.push(libint_res[2]);

        std::vector<double> rys_res = rys_quadrature<double>(cpu_rys_am, cpu_rys_quartet, cpu_rys_rtwt);
        rys_ref.push(cart2sph(cpu_rys_am, mnc2uns, rys_res));

    }
    
    void oneapi_acc_context::submit_ceri()
    {
        if(this->new_batch)
        {
            uint32_t prev_db = (gen_db + 1) % 2;
            this->ceri_offsets[prev_db].push_back(batch_size + this->input_quartets[gen_db].size());
            this->ceri_offsets[gen_db].clear();
            this->ceri_offsets[gen_db].push_back(0);

            size_t last_comp_event = this->comp_events.size() - 1;

            this->evt_abcd[prev_db] = this->q.value().memcpy(this->dev_abcd[prev_db], 
                                                             reinterpret_cast<float*>(this->input_quartets[prev_db].data()), 
                                                             this->batch_size * sizeof(quartet_input<float>), 
                                                             this->comp_events[last_comp_event - 1]);
            this->evt_rtwt[prev_db] = this->q.value().memcpy(this->dev_rtwt[prev_db], 
                                                             reinterpret_cast<float*>(this->input_rtwt[prev_db].data()), 
                                                             this->batch_size * sizeof(rtwt_container<float>), 
                                                             this->comp_events[last_comp_event - 1]);
            
            overwrite_quartet_batch.acquire();
            this->comp_events.push_back(sycl::event()); 
            kernel(this->q.value(),
                   this->dev_abcd[prev_db], this->dev_rtwt[prev_db],
                   this->dev_eri_c[prev_db], this->dev_eps[prev_db],
                   {this->evt_abcd[prev_db], this->evt_rtwt[prev_db], this->comp_events[last_comp_event - 1]},
                   this->comp_events.back());

            
            this->evt_eri_c[prev_db] = this->q.value().memcpy(this->host_eri_c[prev_db],
                                                              this->dev_eri_c[prev_db],
                                                              this->batch_size * this->n512_per_quartet * this->n8_per_n512,
                                                              this->comp_events.back());
            this->evt_eps[prev_db]   = this->q.value().memcpy(this->host_eps[prev_db],
                                                              this->dev_eps[prev_db],
                                                              this->batch_size * sizeof(float),
                                                              this->comp_events.back());
            if(this->new_quartet_class_for_gen)
            {
                this->start_consuming.release(); 
                this->new_quartet_class_for_gen = false;
            }
            this->new_batch = false;
        }

        uint32_t ceri_range_start = this->ceri_offsets[gen_db].back();
        this->ceri_offsets[gen_db].push_back(this->input_quartets[gen_db].size());
        uint32_t ceri_range_end = this->ceri_offsets[gen_db].back();
        //std::cout << "New contraction job for primitive range [" << ceri_range_start << ", " << ceri_range_end << std::endl;
        

        //std::cout << "Submit uncontracted quartets for contraction:\n";
        //for(uint32_t i = ceri_range_start;i < ceri_range_end;i++)
        //{
        //    std::cout << this->input_quartets[i] << this->input_rtwt[i];
        //    std::cout << "contraction coefficient = " << this->quartet_coeffs[i] << '\n';
        //    std::cout << "\n" << std::string(50, '-') << "\n\n";
        //}
    }

    void oneapi_acc_context::finish_quartet_class()
    {
        if(this->input_quartets[gen_db].size() > 0)
        {
            size_t last_comp_event = this->comp_events.size() - 1;

            //float* abcd_data = reinterpret_cast<float*>(this->input_quartets[gen_db].data());
            //float* rtwt_data = reinterpret_cast<float*>(this->input_rtwt[gen_db].data());
            //uint32_t num_elems = sizeof(quartet_input<float>)/sizeof(float);
            //for(uint32_t blk = 0;blk < this->input_quartets[gen_db].size();blk++)
            //{
            //    std::cout << "Primitive input #" << blk << ":\n";
            //    std::cout << "\tabcd = ";
            //    for(uint32_t elem = 0;elem < num_elems;elem++)std::cout << abcd_data[blk * num_elems + elem] << (elem < num_elems - 1 ? ", " : "\n");
            //    std::cout << "\trtwt = ";
            //    for(uint32_t elem = 0;elem < num_elems;elem++)std::cout << rtwt_data[blk * num_elems + elem] << (elem < num_elems - 1 ? ", " : "\n");
            //}


            this->evt_abcd[gen_db] = this->q.value().memcpy(this->dev_abcd[gen_db], 
                                                            reinterpret_cast<float*>(this->input_quartets[gen_db].data()), 
                                                            this->batch_size * sizeof(quartet_input<float>), 
                                                            this->comp_events[last_comp_event - 1]);
            this->evt_rtwt[gen_db] = this->q.value().memcpy(this->dev_rtwt[gen_db], 
                                                            reinterpret_cast<float*>(this->input_rtwt[gen_db].data()), 
                                                            this->batch_size * sizeof(rtwt_container<float>), 
                                                            this->comp_events[last_comp_event - 1]);
            
            overwrite_quartet_batch.acquire();
            this->comp_events.push_back(sycl::event()); 
            kernel(this->q.value(),
                   this->dev_abcd[gen_db], this->dev_rtwt[gen_db],
                   this->dev_eri_c[gen_db], this->dev_eps[gen_db],
                   {this->evt_abcd[gen_db], this->evt_rtwt[gen_db], this->comp_events[last_comp_event - 1]},
                   this->comp_events.back());

            
            this->evt_eri_c[gen_db] = this->q.value().memcpy(this->host_eri_c[gen_db],
                                                             this->dev_eri_c[gen_db],
                                                             this->batch_size * this->n512_per_quartet * this->n8_per_n512,
                                                             this->comp_events.back());
            this->evt_eps[gen_db]   = this->q.value().memcpy(this->host_eps[gen_db],
                                                             this->dev_eps[gen_db],
                                                             this->batch_size * sizeof(float),
                                                             this->comp_events.back());
            this->evt_eri_c[gen_db].wait();
            this->evt_eps[gen_db].wait();

            auto abcd_start = this->evt_abcd[gen_db].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto abcd_end   = this->evt_abcd[gen_db].get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "abcd event : " << abcd_start - abcd_start << " - " << abcd_end - abcd_start << std::endl;
            
            auto rtwt_start = this->evt_rtwt[gen_db].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto rtwt_end   = this->evt_rtwt[gen_db].get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "rtwt event : " << rtwt_start - abcd_start << " - " << rtwt_end - abcd_start << std::endl;
            
            auto comp_start = this->comp_events.back().get_profiling_info<sycl::info::event_profiling::command_start>();
            auto comp_end   = this->comp_events.back().get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "comp event : " << comp_start - abcd_start << " - " << comp_end - abcd_start << std::endl;
            
            auto eri_c_start = this->evt_eri_c[gen_db].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto eri_c_end   = this->evt_eri_c[gen_db].get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "eric event : " << eri_c_start - abcd_start << " - " << eri_c_end - abcd_start << std::endl;
            
            auto eps_start = this->evt_eps[gen_db].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto eps_end   = this->evt_eps[gen_db].get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "eps  event : " << eps_start - abcd_start << " - " << eps_end - abcd_start << std::endl;


            std::cout << "gen_db = " << gen_db << std::endl;
            if(this->comp_events.size() == 3)start_consuming.release();
        }
    }

    //void oneapi_acc_context::get_next_ceri(double* integral_buffer)
    //{
    //    if(this->new_quartet_class_for_consume || ceri_ctr == this->ceri_offsets[consume_db].size() - 1)
    //    {
    //        if(this->new_quartet_class_for_consume)
    //        {
    //            std::cout << "Before consuming" << std::endl;
    //            start_consuming.acquire();
    //            std::cout << "After consuming" << std::endl;
    //            this->new_quartet_class_for_consume = false;
    //        }

    //        overwrite_quartet_batch.release();
    //        consume_db = (consume_db + 1) % 2;
    //        this->evt_eri_c[consume_db].wait();
    //        this->evt_eps[consume_db].wait();
    //        ceri_ctr = 0;
    //    }
    //    std::cout << "ceri_ctr = " << ceri_ctr << ", consume_db = " << consume_db << std::endl;
    //    std::cout << "ceri_offsets[0].size = " << ceri_offsets[0].size() << ", ceri_offsets[1].size = " << ceri_offsets[1].size() << std::endl;

    //    uint32_t ceri_range_start = this->ceri_offsets[consume_db][ceri_ctr];
    //    uint32_t ceri_range_end   = this->ceri_offsets[consume_db][ceri_ctr + 1];
    //    uint32_t eris_per_n512 = 512 / this->cbits;
    //    float* _tmp_integral_buffer = new(std::align_val_t(32)) float[n512_per_quartet * eris_per_n512];
    //    std::cout << "Contracting " << ceri_range_end - ceri_range_start << " quartets" << std::endl;
    //    for(uint32_t c = ceri_range_start;c < ceri_range_end;c++)
    //    {
    //        uint32_t c_off = c * this->n512_per_quartet * this->n8_per_n512;
    //        //decomp.decompress(&(this->host_eri_c[consume_db][c_off]), 
    //        //                  this->host_eps[consume_db][c],
    //        //                  this->n512_per_quartet,
    //        //                  _tmp_integral_buffer
    //        //                  );
    //        sdecomp.decompress(&(this->host_eri_c[consume_db][c_off]), 
    //                           this->host_eps[consume_db][c],
    //                           this->n512_per_quartet,
    //                           _tmp_integral_buffer
    //                           );
    //        std::vector<float> _fpga_sph_integrals = permute_sph_quartet_scalar_single[this->fpga_perm](_tmp_integral_buffer, am);
    //        std::vector<double> fpga_sph_integrals;
    //        std::transform(_fpga_sph_integrals.begin(), _fpga_sph_integrals.end(), std::back_inserter(fpga_sph_integrals), [](float elem){return double(elem);});
    //        //std::vector<double> lmref = libint_maxl_ref.front();
    //        //std::vector<double> luref = libint_unit_ref.front();
    //        //std::vector<double> l2ref = libint2_ref.front();
    //        std::vector<double> rref = rys_ref.front();
    //        //libint_maxl_ref.pop();
    //        //libint_unit_ref.pop();
    //        //libint2_ref.pop();
    //        rys_ref.pop();

    //        std::cout << "Primitive #" << c << ": ( fpga_eps = " << this->host_eps[consume_db][c] << '\n';
    //        //std::cout << "libint-maxl-libint-unit-error: ";
    //        //display_quartet_error<false>(this->am, &luref[0], &lmref[0]); 
    //        //std::cout << "libint-maxl-libint2-error: ";
    //        //display_quartet_error<false>(this->am, &l2ref[0], &lmref[0]); 
    //        //std::cout << "libint-maxl-rys-cpu-error: ";
    //        //display_quartet_error<false>(this->am, &rref[0], &lmref[0]); 
    //        std::cout << "rys_cpu--rys-fpga-error: ";
    //        display_quartet_error<false>(this->am, &fpga_sph_integrals[0], &rref[0], 9.999e-5); 
    //        std::cout << '\n';
    //    }
    //    delete[] _tmp_integral_buffer;

    //    for(uint32_t q = 0;q < this->am.sph_size();q++)integral_buffer[q] = 0.0;
    //    
    //    ceri_ctr++;
    //}
#else
    oneapi_acc_context::oneapi_acc_context(const acc_device& acc_dev, 
                                           const simulation_box& sim_box,
                                           uint32_t num_contexts,
                                           uint64_t mem_available,
                                           bool compute_raw_cartesian,
                                           abstract_acc_context* master_ctx)
        : abstract_acc_context(acc_dev, sim_box, num_contexts, mem_available, compute_raw_cartesian, master_ctx)
    {
    }

    oneapi_acc_context::~oneapi_acc_context(){}
    void oneapi_acc_context::set_B_offset(int off_x, int off_y, int off_z)
    {
    }
    
    void oneapi_acc_context::set_D_offset(int off_x, int off_y, int off_z)
    {
    }

    void oneapi_acc_context::set_exponent_ids(int exp_a, int exp_b, int exp_c, int exp_d) 
    {
    }
    
    void oneapi_acc_context::set_Q_offset(int off_x, int off_y, int off_z)
    {
    }

    void oneapi_acc_context::submit_peri()
    {
    }
#endif
}
