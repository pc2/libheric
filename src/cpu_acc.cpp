#include <cpu_acc.h>

#include <constants.h>
#include <math_util.h>
#include <transform.h>
#include <util.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include <omp.h>

//#include <libint2.hpp>
//#include <libint2/boys.h>

namespace libheric 
{

    std::vector<acc_device> discover_cpu_accelerator(const acc_kind& acc_query)
    {
        std::vector<acc_device> found_cpu;
        std::ifstream f_cpuinfo("/proc/cpuinfo");
        if(f_cpuinfo.is_open())
        {
            std::optional<std::string> cpu_name;
            for(std::string line; std::getline(f_cpuinfo,line);)
            {
                if(line.find("model name") != std::string::npos)
                {
                    auto delim_pos = line.find(":");
                    if(delim_pos != std::string::npos)
                    {
                        cpu_name = line.substr(delim_pos + 1);
                        // Trim leading whitespace
                        cpu_name = cpu_name->substr(
                            std::find_if(cpu_name->begin(), cpu_name->end(), [](char c){return !std::isspace(c);}) - cpu_name->begin()
                        );
                        break;
                    }
                }
            }
            if(cpu_name.has_value())
            {
                int num_cores = omp_get_max_threads();
                found_cpu.push_back({cpu_name.value(), "FFFF:FF:FF.F", cpu_name.value(), num_cores, acc_query});
            }
            else
            {
                std::cout << MESSAGE_TYPE::WARNING << " Found /proc/cpuinfo but unable to query model name!\n";
            }
        }
        else
        {
            std::cout << MESSAGE_TYPE::WARNING << " Unable to query cpuinfo!\n";
        }
        return found_cpu;
    }

    cpu_acc_context::cpu_acc_context(const acc_device& acc_dev, 
                                     const simulation_box& sim_box,
                                     uint32_t ctx_id,
                                     uint64_t mem_available, 
                                     bool compute_raw_cartesian,
                                     abstract_acc_context* master_ctx)
       : abstract_acc_context(
               acc_dev, sim_box, 
               ctx_id,
               mem_available - cpu_acc_context::libint_size() > mem_available ? 1 : mem_available - cpu_acc_context::libint_size(),
               compute_raw_cartesian,
               master_ctx),
         boys_eval(4 * MAX_AM),
         libint_params({lparams0}),
         curr_libint_params(&libint_params[0])
    {
        if(!this->is_initialized)return;
        this->is_initialized = false;
        if constexpr (LOG_PRINT)
        {
            
            *logger << "libint_size = " << cpu_acc_context::libint_size()
                    << " for N_ENGINES = " << N_ENGINES << " and MAX_AM = " << MAX_AM << '\n';
            *logger << "sizeof(Libint_eri_t) = " << sizeof(Libint_eri_t) << '\n';
            *logger << "sizeof(libint_parameters) = " << sizeof(libint_parameters) << '\n';
        }
        uint32_t eng_ctr = 0;
        for(uint64_t eng = 0;eng < N_ENGINES;eng++)
        {
            eng_descs[eng].slots_used = 0;
            eng_descs[eng].first_engine = eng_ctr;
            for(uint32_t v = 0;v < eng_descs[eng].slots.size();v++)
            {
                eng_descs[eng].slots[v].acc_ptr = nullptr;
                eng_descs[eng].slots[v].staging_atom_ptr = nullptr;
            }
            auto n_contr = ipow2(eng);
            libheric_libint2_init_eri(&engines[eng_ctr], MAX_AM, nullptr);
            engines[eng_ctr].contrdepth = n_contr;
            eng_ctr += n_contr;
        }
        if(eng_ctr != N_CONTR_ENGINES)
        {
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << " Initialized " << eng_ctr << " libint2 engines, but there are " << N_CONTR_ENGINES << " engines\n";
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
            return;
        }
        this->is_initialized = true;

    }
    
    cpu_acc_context::~cpu_acc_context()
    {
        for(uint32_t eng = 0;eng < N_ENGINES;eng++)
        {
            libheric_libint2_cleanup_eri(&engines[eng_descs[eng].first_engine]);
        }
    }

    uint64_t cpu_acc_context::libint_size()
    {
        return sizeof(Libint_eri_t) * N_CONTR_ENGINES
               + sizeof(LIBINT2_REALTYPE) * libheric_libint2_need_memory_eri(MAX_AM) * cpu_acc_context::N_ENGINES
               + sizeof(libint_parameters) * N_LPARAMS;
    }

    void cpu_acc_context::set_potential_operator(potential_operator op, double op_param)
    {
        this->abstract_acc_context::set_potential_operator(op, op_param);
    }

    void cpu_acc_context::set_quartet_class(int la, int lb, int lc, int ld)
    {
        this->abstract_acc_context::set_quartet_class(la, lb, lc, ld);
        if(this->is_functional)
        {
            this->libint_input_am = this->am;
            this->libint_input_am.make_libint_canonical();
            engine_eval = libheric_libint2_build_eri[libint_input_am.a()][libint_input_am.b()][libint_input_am.c()][libint_input_am.d()];
            this->input_perm = libint_input_am.get_perm();
            
            this->libint_output_am = this->libint_input_am;
            this->libint_output_am.reverse();
            transform.set_quartet_class(this->libint_output_am); 
            
            if(!engine_eval)
            {
                std::stringstream ss_err;
                ss_err << MESSAGE_TYPE::ERROR << " cpu_acc_context: requested quartet_class " << this->am 
                       << " (libint_permuted -> " << this->libint_input_am << ") " << " is not available\n";
                *logger << ss_err.str() << std::flush;
                if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                this->is_functional = false;
                return;
            }
            if(curr_libint_params != &libint_params[0])
            {
                std::stringstream ss_err;
                ss_err << MESSAGE_TYPE::ERROR << " cpu_acc_context: " << curr_libint_params - &libint_params[0] << " elements left in libint_params\n";
                *logger << ss_err.str() << std::flush;
                if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                this->is_functional = false;
                return;
            }
            for(uint64_t eng = 0;eng < N_ENGINES;eng++)
            {
                if(eng_descs[eng].slots_used != 0)
                {
                    std::stringstream ss_err;
                    ss_err << MESSAGE_TYPE::ERROR << " cpu_acc_context: engine #" << eng << " not empty while setting new quartet class!\n";
                    *logger << ss_err.str() << std::flush;
                    if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                    this->is_functional = false;
                    return;
                }
                eng_descs[eng].slots_used = 0;
                for(uint32_t v = 0;v < eng_descs[eng].slots.size();v++)
                {
                    eng_descs[eng].slots[v].acc_ptr = nullptr;
                    eng_descs[eng].slots[v].staging_atom_ptr = nullptr;
                }
            }
            if constexpr(LOG_PRINT)
            {
                *logger << "Libint input am = " << this->libint_input_am << '\n';
                *logger << "Libint output am = " << this->libint_output_am << '\n';
            }

            this->curr_atom_quartet = nullptr;
            this->curr_shellset_ptr = nullptr;
            this->curr_pmax_ptr = nullptr;
            this->curr_shellset_data_ptr = nullptr;
            this->curr_shellset_count = 0;
            this->t_libint = 0;
            
        }
    }
    
    void cpu_acc_context::set_atom_ids(int atom_a, int atom_b, int atom_c, int atom_d)
    {
        this->abstract_acc_context::set_atom_ids(atom_a, atom_b, atom_c, atom_d);
        this->A = this->sim_box.positions(Eigen::all, atom_a);  
        this->B = this->sim_box.positions(Eigen::all, atom_b);  
        this->C = this->sim_box.positions(Eigen::all, atom_c);  
        this->D = this->sim_box.positions(Eigen::all, atom_d); 
        if constexpr (LOG_PRINT)
        {
            *logger << "Last t_libint time = " << t_libint << '\n';
            this->t_libint = 0;
            *logger << MESSAGE_TYPE::TIMESTAMP;
            *logger << "A = (" << A(0) << ", " << A(1) << ", " << A(2) << ")\n";
            *logger << "B = (" << B(0) << ", " << B(1) << ", " << B(2) << ")\n";
            *logger << "C = (" << C(0) << ", " << C(1) << ", " << C(2) << ")\n";
            *logger << "D = (" << D(0) << ", " << D(1) << ", " << D(2) << ")\n";
        }
        // Accept/Reject last shellset_quartet
        if(this->curr_atom_quartet && this->curr_shellset_count > 0)
        {
            this->submit_contracted(curr_libint_params - &libint_params[0]);
            this->curr_atom_quartet->value.n_shellset_quartets++;
        }

        this->curr_atom_quartet = this->staging.reserve(atom_a, atom_b, atom_c, atom_d);
        if(!this->curr_atom_quartet)
        {
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << " no space left in quartet staging. Forgot to call get_next_staged_quartet()?\n";
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
            this->is_functional = false;
            return;
        }
        this->curr_atom_quartet->prims_requested = 0;
        this->curr_atom_quartet->prims_computed = 0;
        this->curr_atom_quartet->is_transformed = false;
        this->curr_shellset_ptr = curr_atom_quartet->value.first_shellset_ptr;
        this->curr_pmax_ptr = (double*)byte_add(curr_atom_quartet->value.first_shellset_ptr, sizeof(uint64_t) * this->max_shellset_quartets);
        this->curr_shellset_data_ptr = curr_atom_quartet->value.first_shellset_data_ptr;
        this->curr_shellset_count = 0;
    }

    void cpu_acc_context::set_shell_ids(int shl_a, int shl_b, int shl_c, int shl_d)
    {
        this->abstract_acc_context::set_shell_ids(shl_a, shl_b, shl_c, shl_d);
        this->norm_prod = this->sim_box.normalizations[this->norm_off(0) + shl_a]
                        * this->sim_box.normalizations[this->norm_off(1) + shl_b]
                        * this->sim_box.normalizations[this->norm_off(2) + shl_c]
                        * this->sim_box.normalizations[this->norm_off(3) + shl_d];

        // Check if last shellset has not been screened
        if(this->curr_shellset_count > 0)
        {
            this->submit_contracted(curr_libint_params - &libint_params[0]);
            this->curr_shellset_ptr++;
            this->curr_pmax_ptr++;
            this->curr_shellset_data_ptr = byte_add(this->curr_shellset_data_ptr, staging.eris_allocated_per_shellset * sizeof(double));
            this->curr_atom_quartet->value.n_shellset_quartets++;
        }
        *(this->curr_shellset_ptr) = shellset_id;
        *(this->curr_pmax_ptr) = this->pmax;
        this->curr_shellset_count = 0;
        if constexpr (DEBUG_PRINT)
        {
            *logger << "Setting shellset = " << *curr_shellset_ptr << " with norm_prod = " << norm_prod << '\n';
        }
    }
    
    void cpu_acc_context::set_B_offset(int off_x, int off_y, int off_z)
    {
        auto B_off = Eigen::Vector3d(double(off_x), double(off_y), double(off_z));
        this->BB = this->B + this->sim_box.cell_mat * B_off;
        auto ABB = this->A - this->BB;
        this->ABB2 = ABB.dot(ABB);
        if constexpr (DEBUG_PRINT)
            *logger << "B_off = (" << off_x << ", " << off_y << ", " << off_z << ")\n";
        
        this->curr_libint_params->AB_x = ABB(0);
        this->curr_libint_params->AB_y = ABB(1);
        this->curr_libint_params->AB_z = ABB(2);
    }
    
    void cpu_acc_context::set_D_offset(int off_x, int off_y, int off_z)
    {
        // has no effect if difference is 0
        this->submit_contracted(curr_libint_params - &libint_params[0]);

        auto D_off = Eigen::Vector3d(double(off_x), double(off_y), double(off_z));
        this->DD = this->D + this->sim_box.cell_mat * D_off;
        auto CDD = this->C - this->DD;
        this->CDD2 = CDD.dot(CDD);
        if constexpr (DEBUG_PRINT)
            *logger << "D_off = (" << off_x << ", " << off_y << ", " << off_z << ")\n";
        
        this->curr_libint_params->CD_x = CDD(0);
        this->curr_libint_params->CD_y = CDD(1);
        this->curr_libint_params->CD_z = CDD(2);
    }

    void cpu_acc_context::set_exponent_ids(int exp_a, int exp_b, int exp_c, int exp_d) 
    {
        this->coeff_prod = this->sim_box.coefficients[this->coeff_off(0) + exp_a]
                         * this->sim_box.coefficients[this->coeff_off(1) + exp_b]
                         * this->sim_box.coefficients[this->coeff_off(2) + exp_c]
                         * this->sim_box.coefficients[this->coeff_off(3) + exp_d];
        double alpha = this->sim_box.exponents[this->exp_off(0) + exp_a];
        double beta  = this->sim_box.exponents[this->exp_off(1) + exp_b];
        double gamma = this->sim_box.exponents[this->exp_off(2) + exp_c];
        double delta = this->sim_box.exponents[this->exp_off(3) + exp_d];

        this->zeta = alpha + beta;
        this->eta  = gamma + delta;
        this->chi  = this->zeta  + this->eta;
        double ooz = 1.0 / this->zeta;
        double ooe = 1.0 / this->eta;
        this->ooc = 1.0 / this->chi;
        this->rho  = (this->zeta * this->eta) * this->ooc;
        this->P = (alpha * this->A + beta  * this->BB) * ooz;
        this->Q = (gamma * this->C + delta * this->DD) * ooe;
        auto PA = this->P - this->A;
        auto QC = this->Q - this->C; //Q_off does not affect this vector
        double pi_over_chi = std::numbers::pi * this->ooc;
        this->prefactor = std::exp(-((alpha * beta  * this->ABB2) * ooz
                                  +(gamma * delta * this->CDD2) * ooe))
                        * std::sqrt(pi_over_chi * pi_over_chi * pi_over_chi)
                        * (this->compute_raw_cartesian ? 1.0 : this->norm_prod * this->coeff_prod);
        
        this->curr_libint_params->PA_x = PA(0);
        this->curr_libint_params->PA_y = PA(1);
        this->curr_libint_params->PA_z = PA(2);
        
        this->curr_libint_params->QC_x = QC(0);
        this->curr_libint_params->QC_y = QC(1);
        this->curr_libint_params->QC_z = QC(2);

        this->curr_libint_params->oo2z = 0.5 * ooz;
        this->curr_libint_params->oo2e = 0.5 * ooe;
        this->curr_libint_params->oo2ze = 0.5 * this->ooc;
        this->curr_libint_params->roz = this->rho * ooz;
        this->curr_libint_params->roe = this->rho * ooe;
        this->curr_libint_params->oor = 1.0 / rho;
        if(op == potential_operator::short_range)
            this->curr_libint_params->op_param = this->op_param / std::sqrt(this->rho + this->op_param * this->op_param);
        if constexpr (DEBUG_PRINT)
        {
            double c1 = this->sim_box.coefficients[this->coeff_off(0) + exp_a];
            double c2 = this->sim_box.coefficients[this->coeff_off(1) + exp_b];
            double c3 = this->sim_box.coefficients[this->coeff_off(2) + exp_c];
            double c4 = this->sim_box.coefficients[this->coeff_off(3) + exp_d];
            *logger << "Setting exps = (" << exp_a << "," << exp_b << "|" << exp_c << "," << exp_d
                    << ")\nwith exp_off = { " << this->exp_off(0) << ", " << this->exp_off(1) << ", " << this->exp_off(2) << ", " << this->exp_off(3)
                    << ")\nand coeff_off = { " << this->coeff_off(0) << ", " << this->coeff_off(1) << ", " << this->coeff_off(2) << ", " << this->coeff_off(3)
                    << " }\nand values = { " << alpha << ", " << beta << ", " << gamma << ", " << delta
                    << " }\nand coeffs = { " << c1 << ", " << c2 << ", " << c3 << ", " << c4   
                    << " }\nwith P = (" << P(0) << ", " << P(1) << ", " << P(2) << ")"
                    << ",\nQ = (" << Q(0) << ", " << Q(1) << ", " << Q(2) << ")"
                    << ",\nzeta = " << zeta << ", eta = " << eta << ", rho = " << rho
                    << ",\ncoeff_prod = " << coeff_prod << ", prefactor = " << prefactor << '\n';
        }
    }
    
    void cpu_acc_context::set_Q_offset(int off_x, int off_y, int off_z)
    {
        auto Q_off = Eigen::Vector3d(double(off_x), double(off_y), double(off_z));
        auto QQ = this->Q + this->sim_box.cell_mat * Q_off;
        auto PQQ = P - QQ;
        auto W = (this->zeta * this->P + this->eta * QQ) * this->ooc;
        auto WP = W - this->P;
        auto WQ = W - QQ;
        auto PQQ2 = PQQ.dot(PQQ);

        this->curr_libint_params->WP_x = WP(0);
        this->curr_libint_params->WP_y = WP(1);
        this->curr_libint_params->WP_z = WP(2);
        
        this->curr_libint_params->WQ_x = WQ(0);
        this->curr_libint_params->WQ_y = WQ(1);
        this->curr_libint_params->WQ_z = WQ(2);

        this->curr_libint_params->rhopq2 = this->rho * PQQ2;
        if constexpr (DEBUG_PRINT)
        {
            *logger << "Q_off = (" << off_x << ", " << off_y << ", " << off_z
                    << ")\nwith QQ = (" << QQ(0) << ", " << QQ(1) << ", " << QQ(2) << ")"
                    << ",\nW = (" << W(0) << ", " << W(1) << ", " << W(2) << ")"
                    << ",\nPQQ2 = " << PQQ2 << '\n';
        }
    }
    

    void cpu_acc_context::submit_peri()
    {
        this->curr_atom_quartet->prims_requested++;
        this->curr_shellset_count++;
        if(op == potential_operator::coulomb)
        {
            this->boys_eval.eval(&(this->curr_libint_params->G[0]), this->curr_libint_params->rhopq2, this->total_am);
            for(uint32_t am_idx = 0;am_idx <= this->total_am;am_idx++)
                this->curr_libint_params->G[am_idx] *= this->prefactor * two_pi * this->curr_libint_params->oor;
        }
        else if(op == potential_operator::short_range)
        {
            this->boys_eval.eval(&(this->curr_libint_params->G[0]), this->curr_libint_params->rhopq2, this->total_am);
            double theta = this->curr_libint_params->op_param;  
            double theta2 = theta * theta;
            boys_eval.eval(&(G_buffer[0]), theta2 * this->curr_libint_params->rhopq2, this->total_am);
            double scale = theta;
            for(uint32_t am_idx = 0;am_idx <= this->total_am;am_idx++)
            {
                this->curr_libint_params->G[am_idx] -= scale * this->G_buffer[am_idx];
                this->curr_libint_params->G[am_idx] *= this->prefactor * two_pi * this->curr_libint_params->oor;
                scale *= theta2;
            }
             
        }
        if constexpr (DEBUG_PRINT)
        {
            *logger << *curr_libint_params << "\n\n";
        }
        if(curr_libint_params == &libint_params[0] + libint_params.size() - 1)
        {
            // Submit to last engine
            submit_contracted(libint_params.size());
        }
        else
        {
            curr_libint_params++;
            memcpy(curr_libint_params, curr_libint_params - 1, cpu_acc_context::LIBINT_PARAMETERS_COPY_BYTES);
        }
        

        //if(this->qq.full())
        //{
        //    // Use complete batch -> there can only be 1 atom quartet in it!
        //}
        //prim_tmpl.diff = this->quartet_diff;
        //switch(this->quartet_diff)
        //{
        //    case 0:
        //        this->qq.push(atom_tmpl);
        //    case 1:
        //    case 2:
        //    case 3:
        //    case 4:
        //        this->qq.push(shellset_tmpl.id);
        //    default:
        //        this->qq.push(prim_tmpl);
        //}
        //quartet_diff = 6;
        //if(curr_task_it == quartet_tasks.end())
        //{
        //    std::unique_lock lck(task_consumer_callback_mtx);
        //    curr_task_it = std::find_if(quartet_tasks.begin(), quartet_tasks.end(), [](const quartet_tasks& t){return !t.is_used()});
        //    while(curr_task_it == quartet_tasks.end())
        //    {
        //        task_consumer_callback.wait(lck); 
        //        curr_task_it = std::find_if(quartet_tasks.begin(), quartet_tasks.end(), [](const quartet_tasks& t){return !t.is_used()});
        //    }
        //}

        //if(curr_task_it->push_back(quartet_tmpl))
        //{
        //    auto curr_task = std::distance(quartet_tasks.begin(), curr_task_it);
        //    if(accelerator_queue)
        //    {
        //        if(!accelerator_queue->try_push(&quartet_task[curr_task]))
        //        {
        //            default_queue->push(&quartet_task[curr_task]);
        //        }
        //    }
        //    else
        //    {
        //        default_queue->push(&quartet_task[curr_task]);
        //    }
        //    curr_task_it = std::find_if(quartet_tasks.begin(), quartet_tasks.end(), [](const quartet_tasks& t){return !t.is_used()});
        //}
        //quartet_tasks.push(quartet_tmpl);
        //quartet_tmpl.diff = quartet_diff;
        //quartet_idx q = quartet_tmpl;
        //std::cout << "Processing quartet:"
        //          << " atom = (" << q.atom[0] << ", " << q.atom[1] << ", " << q.atom[2] << ", " << q.atom[3] << "),"
        //          << " set = (" << int32_t(q.set[0]) << ", " << int32_t(q.set[1]) << ", " << int32_t(q.set[2]) << ", " << int32_t(q.set[3]) << "),"
        //          << " shell = (" << int32_t(q.shl[0]) << ", " << int32_t(q.shl[1]) << ", " << int32_t(q.shl[2]) << ", " << int32_t(q.shl[3]) << "),"
        //          << " exp = (" << int32_t(q.exp[0]) << ", " << int32_t(q.exp[1]) << ", " << int32_t(q.exp[2]) << ", " << int32_t(q.exp[3]) << "),"
        //          << " B_off = (" << int32_t(q.off[0]) << ", " << int32_t(q.off[1]) << ", " << int32_t(q.off[2]) << "),"
        //          << " D_off = (" << int32_t(q.off[3]) << ", " << int32_t(q.off[4]) << ", " << int32_t(q.off[5]) << "),"
        //          << " Q_off = (" << int32_t(q.off[6]) << ", " << int32_t(q.off[7]) << ", " << int32_t(q.off[8]) << "),"
        //          << " diff = " << int32_t(q.diff) << '\n';
        //qq.push(quartet_tmpl);
        //quartet_diff = 6;
    }

    void cpu_acc_context::insert_params_into_engine_slot(Libint_eri_t& engine, const libint_parameters& lparams, uint32_t slot)
    {
        // Input permutations are executed "left to right", i.e the braket-swap is executed before the bra/ket-swaps
        bool ket_swap = (this->input_perm & 0x1) == 0x1;
        bool bra_swap = (this->input_perm & 0x2) == 0x2;
        bool braket_swap = (this->input_perm & 0x4) == 0x4;
        if(braket_swap)
        {

            // for a braket-swap, the new AB is DC if there is also a bra-swap and CD otherwise
            *(((double*)&(engine.AB_x[0])) + slot) = bra_swap ? -lparams.CD_x : lparams.CD_x;
            *(((double*)&(engine.AB_y[0])) + slot) = bra_swap ? -lparams.CD_y : lparams.CD_y;
            *(((double*)&(engine.AB_z[0])) + slot) = bra_swap ? -lparams.CD_z : lparams.CD_z;
            
            // for a braket-swap, the new CD is BA if there is also a ket-swap and AB otherwise
            *(((double*)&(engine.CD_x[0])) + slot) = ket_swap ? -lparams.AB_x : lparams.AB_x;
            *(((double*)&(engine.CD_y[0])) + slot) = ket_swap ? -lparams.AB_y : lparams.AB_y;
            *(((double*)&(engine.CD_z[0])) + slot) = ket_swap ? -lparams.AB_z : lparams.AB_z;
            
            // for a braket-swap, the new PA is QD (QC + CD) if there is also a bra-swap and QC otherwise
            *(((double*)&(engine.PA_x[0])) + slot) = bra_swap ? lparams.QC_x + lparams.CD_x : lparams.QC_x;
            *(((double*)&(engine.PA_y[0])) + slot) = bra_swap ? lparams.QC_y + lparams.CD_y : lparams.QC_y;
            *(((double*)&(engine.PA_z[0])) + slot) = bra_swap ? lparams.QC_z + lparams.CD_z : lparams.QC_z;
            
            // for a braket-swap, the new QC is PB (PA + AB) if there is also a ket-swap and PA otherwise
            *(((double*)&(engine.QC_x[0])) + slot) = ket_swap ? lparams.PA_x + lparams.AB_x : lparams.PA_x;
            *(((double*)&(engine.QC_y[0])) + slot) = ket_swap ? lparams.PA_y + lparams.AB_y : lparams.PA_y;
            *(((double*)&(engine.QC_z[0])) + slot) = ket_swap ? lparams.PA_z + lparams.AB_z : lparams.PA_z;
           
            // for a braket-swap, WP and WQ are swapped
            *(((double*)&(engine.WP_x[0])) + slot) = lparams.WQ_x;
            *(((double*)&(engine.WP_y[0])) + slot) = lparams.WQ_y;
            *(((double*)&(engine.WP_z[0])) + slot) = lparams.WQ_z;
            
            *(((double*)&(engine.WQ_x[0])) + slot) = lparams.WP_x;
            *(((double*)&(engine.WQ_y[0])) + slot) = lparams.WP_y;
            *(((double*)&(engine.WQ_z[0])) + slot) = lparams.WP_z;
            
            // for a braket-swap, zeta and eta are swapped
            *(((double*)&(engine.oo2z[0]))  + slot) = lparams.oo2e;
            *(((double*)&(engine.oo2e[0]))  + slot) = lparams.oo2z;
            *(((double*)&(engine.oo2ze[0])) + slot) = lparams.oo2ze;
            *(((double*)&(engine.roz[0]))   + slot) = lparams.roe;
            *(((double*)&(engine.roe[0]))   + slot) = lparams.roz;
        }
        else
        {
            // for a bra-swap, BA is needed instead of AB
            *(((double*)&(engine.AB_x[0])) + slot) = bra_swap ? -lparams.AB_x : lparams.AB_x;
            *(((double*)&(engine.AB_y[0])) + slot) = bra_swap ? -lparams.AB_y : lparams.AB_y;
            *(((double*)&(engine.AB_z[0])) + slot) = bra_swap ? -lparams.AB_z : lparams.AB_z;
            
            // for a ket-swap, DC is needed instead of CD
            *(((double*)&(engine.CD_x[0])) + slot) = ket_swap ? -lparams.CD_x : lparams.CD_x;
            *(((double*)&(engine.CD_y[0])) + slot) = ket_swap ? -lparams.CD_y : lparams.CD_y;
            *(((double*)&(engine.CD_z[0])) + slot) = ket_swap ? -lparams.CD_z : lparams.CD_z;
            
            // for a bra-swap, PB (PA + AB) is needed instead of PA
            *(((double*)&(engine.PA_x[0])) + slot) = bra_swap ? lparams.PA_x + lparams.AB_x : lparams.PA_x;
            *(((double*)&(engine.PA_y[0])) + slot) = bra_swap ? lparams.PA_y + lparams.AB_y : lparams.PA_y;
            *(((double*)&(engine.PA_z[0])) + slot) = bra_swap ? lparams.PA_z + lparams.AB_z : lparams.PA_z;
            
            // for a ket-swap, QD (QC + CD) is needed instead if QC
            *(((double*)&(engine.QC_x[0])) + slot) = ket_swap ? lparams.QC_x + lparams.CD_x : lparams.QC_x;
            *(((double*)&(engine.QC_y[0])) + slot) = ket_swap ? lparams.QC_y + lparams.CD_y : lparams.QC_y;
            *(((double*)&(engine.QC_z[0])) + slot) = ket_swap ? lparams.QC_z + lparams.CD_z : lparams.QC_z;
           
            // W, Q, P and exponent parameters remain unchanged without braket-swap
            *(((double*)&(engine.WP_x[0])) + slot) = lparams.WP_x;
            *(((double*)&(engine.WP_y[0])) + slot) = lparams.WP_y;
            *(((double*)&(engine.WP_z[0])) + slot) = lparams.WP_z;
            
            *(((double*)&(engine.WQ_x[0])) + slot) = lparams.WQ_x;
            *(((double*)&(engine.WQ_y[0])) + slot) = lparams.WQ_y;
            *(((double*)&(engine.WQ_z[0])) + slot) = lparams.WQ_z;
            
            *(((double*)&(engine.oo2z[0]))  + slot) = lparams.oo2z;
            *(((double*)&(engine.oo2e[0]))  + slot) = lparams.oo2e;
            *(((double*)&(engine.oo2ze[0])) + slot) = lparams.oo2ze;
            *(((double*)&(engine.roz[0]))   + slot) = lparams.roz;
            *(((double*)&(engine.roe[0]))   + slot) = lparams.roe;
        }

        double* G_ptr = ((double*)&(engine.LIBINT_T_SS_EREP_SS(0)[0])) + slot;
        for(uint32_t am_idx = 0;am_idx <= this->total_am;am_idx++,G_ptr += LIBINT2_REALTYPE::size())
        {
            *G_ptr = lparams.G[am_idx];
        }

    }

    void cpu_acc_context::submit_contracted(uint64_t n_contr)
    {
        uint32_t eng_id = eng_descs.size() - 1;
        uint32_t eng_contr = libint_params.size();
        uint32_t c = 0;
        while(n_contr > 0)
        {
            if(n_contr >= eng_contr)
            {
                auto& curr_eng = this->eng_descs[eng_id];
                if(curr_eng.slots_used == curr_eng.slots.size())this->flush_engine(eng_id);
                for(uint32_t eng_contr_idx = 0;eng_contr_idx < eng_contr;eng_contr_idx++,c++)
                {
                    auto& libint_eng = this->engines[curr_eng.first_engine + eng_contr_idx];
                    const auto& lparams = libint_params[c];
                    this->insert_params_into_engine_slot(libint_eng, lparams, curr_eng.slots_used);
                }
                curr_eng.slots[curr_eng.slots_used].acc_ptr = this->curr_shellset_data_ptr;
                curr_eng.slots[curr_eng.slots_used].staging_atom_ptr = this->curr_atom_quartet;
                curr_eng.slots_used++; 
                n_contr -= eng_contr;
            }
            eng_contr /= 2;
            eng_id--;
        }
        // Reset parameter pointer
        if(curr_libint_params != &libint_params[0])
            memcpy(&libint_params[0], curr_libint_params, cpu_acc_context::LIBINT_PARAMETERS_COPY_BYTES);
        curr_libint_params = &libint_params[0];
    }
    
    void cpu_acc_context::flush_engine(uint32_t eng_id)
    {
        auto& curr_eng = this->eng_descs[eng_id];
        if(curr_eng.slots_used == 0)return;
        bool single_accum = true;
        void* first_slot_acc = curr_eng.slots[0].acc_ptr;
        for(uint32_t slot = 1;slot < curr_eng.slots_used;slot++)
        {
            single_accum = single_accum && (first_slot_acc == curr_eng.slots[slot].acc_ptr);
            if(!single_accum)break;
        }
        if(curr_eng.slots_used < curr_eng.slots.size())
        {
            //Fill unused slots with 0s to force 0s in the output and set accumulate to first slot
            uint64_t eng_contr = ipow2(eng_id);
            for(uint32_t slot = curr_eng.slots_used;slot < curr_eng.slots.size();slot++)
            {
                curr_eng.slots[slot].acc_ptr = curr_eng.slots[0].acc_ptr;
                for(uint32_t eng_contr_idx = 0;eng_contr_idx < eng_contr;eng_contr_idx++)
                {
                    auto& libint_eng = this->engines[curr_eng.first_engine + eng_contr_idx];
                    this->insert_params_into_engine_slot(libint_eng, cpu_acc_context::lparams0, slot);
                }

            }
        }

        // Hand off to libint
        double t_start = omp_get_wtime();
        engine_eval(&this->engines[curr_eng.first_engine]);
        double t_stop = omp_get_wtime();
        t_libint += t_stop - t_start;

        double* eri_ptr = (double*)this->engines[curr_eng.first_engine].targets[0];
        if(single_accum)
        {
            double* acc_ptr = (double*)curr_eng.slots[0].acc_ptr;
            if constexpr (DEBUG_PRINT)
                *logger << "\tEngine #" << eng_id <<  " -> single accum into " << acc_ptr << '\n';
            for(uint64_t eri = 0;eri < this->cart_size;eri++,acc_ptr++)
            {
                if constexpr (DEBUG_PRINT)
                    *logger << "\teri[" << eri << "] = ";
                for(uint64_t slot = 0;slot < curr_eng.slots.size();slot++,eri_ptr++)
                {
                    if constexpr (DEBUG_PRINT)
                        *logger << *eri_ptr << (slot == curr_eng.slots.size() - 1 ? "\n" : " ");
                    *acc_ptr += *eri_ptr;
                }
            }
            curr_eng.slots[0].staging_atom_ptr->prims_computed += curr_eng.slots_used * ipow2(eng_id);
        }
        else
        {
            double* acc_ptr[curr_eng.slots.size()];
            if constexpr (DEBUG_PRINT)
                *logger << "\tEngine #" << eng_id << " -> accum into ";
            for(uint64_t slot = 0;slot < curr_eng.slots.size();slot++)
            {
                acc_ptr[slot] = (double*)curr_eng.slots[slot].acc_ptr;
                if constexpr (DEBUG_PRINT)
                    *logger << '\t' << acc_ptr[slot] << (slot == curr_eng.slots.size() - 1 ? "\n" : ", ");
            }

            for(uint64_t eri = 0;eri < this->cart_size;eri++)
            {
                if constexpr (DEBUG_PRINT)
                    *logger << "\teri[" << eri << "] = ";
                for(uint64_t slot = 0;slot < curr_eng.slots.size();slot++,eri_ptr++)
                {
                    if constexpr (DEBUG_PRINT)
                        *logger << *eri_ptr << (slot == curr_eng.slots.size() - 1 ? "\n" : " ");
                    *(acc_ptr[slot]) += *eri_ptr;
                }
                for(uint64_t slot = 0;slot < curr_eng.slots.size();slot++)
                    acc_ptr[slot]++;
            }
            for(uint64_t slot = 0;slot < curr_eng.slots_used;slot++)
                curr_eng.slots[slot].staging_atom_ptr->prims_computed += ipow2(eng_id); 
        }
        curr_eng.slots_used = 0;
    }
    
    void cpu_acc_context::finish_quartet_class()
    {
        abstract_acc_context::finish_quartet_class();
        if constexpr (DEBUG_PRINT)
        {
            for(uint32_t eng_id = 0;eng_id < N_ENGINES;eng_id++)
                *logger << "Engine #" << eng_id << '\n' << this->eng_descs[eng_id] << '\n';
            *logger << "Submitting " << curr_libint_params - &libint_params[0] << " primitives\n";
        }
        this->submit_contracted(curr_libint_params - &libint_params[0]);
        if constexpr (DEBUG_PRINT)
        {
            for(uint32_t eng_id = 0;eng_id < N_ENGINES;eng_id++)
                *logger << "Engine #" << eng_id << '\n' << this->eng_descs[eng_id] << '\n';
            *logger << "Flushing engines\n";
        }
        for(uint32_t eng_id = 0;eng_id < N_ENGINES;eng_id++)this->flush_engine(eng_id);
        if constexpr (DEBUG_PRINT)
        {
            for(uint32_t eng_id = 0;eng_id < N_ENGINES;eng_id++)
            *logger << "Engine #" << eng_id << '\n' << this->eng_descs[eng_id] << '\n';
        }
        if(this->curr_atom_quartet && this->curr_shellset_count > 0)
            this->curr_atom_quartet->value.n_shellset_quartets++;
        
    }
    
    abstract_acc_context::staging_index* cpu_acc_context::get_next_staged_atom_quartet()
    {
        while(!this->staging.empty())
        {
            staging_index* next_atom_quartet = this->staging.get_next_atom_quartet();
            if constexpr (DEBUG_PRINT) *logger << "next_atom_quartet = " << next_atom_quartet << '\n'; 
            if(!next_atom_quartet)
            {
                if(this->staging.full())
                {
                    // Flushing all engines guarantees that at least one new atom quartet finishes
                    for(uint32_t eng_id = 0;eng_id < N_ENGINES;eng_id++)this->flush_engine(eng_id);
                    next_atom_quartet = this->staging.get_next_atom_quartet(); 
                }
                else
                {
                    // No need to flush there is still space in the quartet staging buffer
                    if constexpr (LOG_PRINT) *logger << "Request for next staged atom quartet postponed, because atom quartet is not yet ready and staging still has space\n";
                    return nullptr;
                }
            } 
            
            // Sanity_check
            if(!next_atom_quartet)
            {
                std::stringstream ss_err;
                ss_err << MESSAGE_TYPE::ERROR << " Request for next staged atom failed, even after flushing!\n";
                *logger << ss_err.str() << std::flush;
                if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                return nullptr;
            }

            if(!next_atom_quartet->is_transformed)
            {
                double t_start = omp_get_wtime();
                double* cart_eris = (double*) next_atom_quartet->value.first_shellset_data_ptr; 
                double* sph_eris = (double*) next_atom_quartet->value.first_shellset_data_ptr;
                for(uint64_t q = 0;q < next_atom_quartet->value.n_shellset_quartets;q++,
                    cart_eris += this->staging.eris_allocated_per_shellset, 
                    sph_eris += this->sph_size)
                {
                    transform.apply_transformation(this->transform, cart_eris, sph_eris);
                }
                double t_stop = omp_get_wtime();
                if constexpr (LOG_PRINT) *logger << "Transformation took: " << t_stop - t_start << " seconds\n";
            }
            
            next_atom_quartet->eris_allocated_per_shellset = this->compute_raw_cartesian ? this->cart_size : this->sph_size; // A bit hacky 
            if constexpr (LOG_PRINT)
            {
                *logger << MESSAGE_TYPE::TIMESTAMP 
                        << " -> Got staged quartet : " << *next_atom_quartet << '\n'
                        << " -> staging.read_idx = "  << this->staging.read_idx
                        << ", staging.write_idx = " << this->staging.write_idx
                        << ", staging.n_staging = "  << this->staging.n_staging
                        << ", staging.mem_pool = "  << this->staging.mem_pool
                        << ", staging.index_ptr[write_idx].first_shellset_data_ptr = " 
                        << this->staging.index_ptr[this->staging.write_idx].value.first_shellset_data_ptr 
                        << ", staging.index_ptr = "  << this->staging.index_ptr << '\n';
            }

            //std::array<uint32_t, 4> ref_atom_request;
            //for(uint32_t i = 0;i < 4;i++)ref_atom_request[i] = 1 + next_atom_quartet->key[i];
            //auto ref_atom_quartet = this->get_reference_atom_quartet(ref_atom_request, this->am); 
            //if(ref_atom_quartet.first_shellset_data_ptr != nullptr)
            //{
            //    compare_atom_quartets<false>(this->am, next_atom_quartet->value, ref_atom_quartet, 1e-9, next_atom_quartet->eris_allocated_per_shellset);
            //    // Ugly!
            //    delete[] (double*)ref_atom_quartet.first_shellset_data_ptr;
            //    delete[] ref_atom_quartet.first_shellset_ptr;
            //    *logger << "\n";
            //}
            
            // Not screened
            if(next_atom_quartet->value.n_shellset_quartets != 0)return next_atom_quartet;
        }

        return nullptr;
    }

     
    //bool cpu_acc_context::compute()
    //{
        //quartet_idx* batch = nullptr;
        //uint64_t batch_size = 0;
        //return false;
        //do
        //{  
        //    if(this->quartet_class_finalizing)
        //    {
        //        auto batch_wide_ptr = qq.incomplete_front_batch();
        //        batch = batch_wide_ptr.first;
        //        batch_size = batch_wide_ptr.second;
        //    }
        //    else
        //    {
        //        batch = qq.front_batch();
        //        batch_size = batch == nullptr ? 0 : qq.batch_size;
        //    }
        //    if(batch != nullptr && batch_size != 0)
        //    {
        //        uint64_t shellset_id;
        //        uint32_t set_id, shell_id;
        //        uint32_t curr_eng_idx, curr_eng_triag_idx, curr_slot;
        //        engine_descriptor* curr_eng_desc;
        //        Libint_eri_t* curr_eng;
        //        double* eng_G_ptr;
        //        double* curr_acc_ptr;
        //        double* res_acc;
        //        uint64_t q_idx;
        //        atom_quartet_descriptor* curr_atom_quartet = staging.get_current_descriptor();
        //        if(!curr_atom_quartet)curr_atom_quartet = this->staging.reserve(batch[0].atom[0], batch[0].atom[1], batch[0].atom[2], batch[0].atom[3]);
        //        uint64_t atom_quartet_high = prev_q_idx;
        //        do 
        //        {
        //            uint64_t atom_quartet_low = atom_quartet_high;
        //            uint64_t atom_quartet_high++;
        //            curr_shell_ptr = curr_atom_quartet->first_shellset_ptr;
        //            // Initialize shellset id array
        //            for(;atom_quartet_high < batch_size && batch[atom_quartet_high].diff > 2;atom_quartet_high++)
        //            {
        //                switch(batch[atom_quartet_high].diff)
        //                {
        //                    case 3:
        //                    case 4:
        //                        curr_shellset_ptr++;
        //                        curr_shellset_ptr->shellset_id = batch[atom_quartet_high].shellset_id;
        //                        curr_shellset_ptr->count = 0;
        //                        curr_atom_quartet->n_shellset_quartets++;
        //                        [[fallthrough]];
        //                    default:
        //                        curr_shellset_ptr->count++;

        //                }
        //            }
        //            // Count the number of primitives for each shellset
        //            shellset_quartet_descriptor* curr_atom_quartet_shellset_end = curr_atom_quartet->first_shellset_ptr + curr_atom_quartet->n_shellset_quartets;
        //            for(;atom_quartet_high < batch_size && batch[atom_quartet_high].diff != 0;atom_quartet_high++)
        //            {
        //                switch(batch[atom_quartet_high].diff)
        //                {
        //                    case 1:
        //                    case 2:
        //                        curr_shellset_ptr = curr_atom_quartet->first_shellset_ptr;
        //                        [[fallthrough]];
        //                    case 3:
        //                    case 4:
        //                        curr_shellset_ptr = std::lower_bound(curr_shellset_ptr, curr_atom_quartet_shellset_end, batch[atom_quartet_high].shellset_id);
        //                        if(curr_shellset_ptr == curr_shellset_ptr + curr_atom_quartet.n_shellset_quartets)
        //                        {
        //                            std::cout << "[ERROR] shellset_id = " << batch[atom_quartet_high].shellset_id << " not in shellset_id_list\n";
        //                            return false; // ???
        //                        }
        //                        [[fallthrough]];
        //                    default
        //                        curr_shellset_ptr->count++;

        //                }
        //            }

        //            std::cout << "Input order for atom quartet:\n";
        //            for(uint64_t i = atom_quartet_low;i < atom_quartet_high;i++)
        //            {
        //                std::cout << batch[i] << '\n';
        //            }
        //            

        //            //Set up pointers to reorder buffer
        //            uint64_t cum_count = 0;
        //            for(curr_shellset_ptr != curr_atom_quartet_shellset_end;curr_shellset_ptr++)
        //            {
        //                uint64_t tmp_count = curr_shellset_ptr->count;
        //                curr_shellset_ptr->quartet_reorder_ptr = quartet_reorder_buffer + cum_count;
        //                cum_count += tmp_count;
        //            }

        //            for(uint64_t i = atom_quartet_low;i < atom_quartet_high;i++)
        //            {
        //                switch(batch[atom_quartet_high].diff)
        //                {
        //                    case 1:
        //                    case 2:
        //                        curr_shellset_ptr = curr_atom_quartet->first_shellset_ptr;
        //                    case 3:
        //                    case 4:
        //                        curr_shellset_ptr = std::lower_bound(curr_shellset_ptr, curr_atom_quartet_shellset_end, batch[i].shellset_id);
        //                        if(curr_shellset_ptr == curr_shellset_ptr + curr_atom_quartet.n_shellset_quartets)
        //                        {
        //                            std::cout << "[ERROR] shellset_id = " << batch[i].shellset_id << " not in shellset_id_list\n";
        //                            return false; // ???
        //                        }
        //                    case 5:
        //                        *curr_shellset->quartet_reorder_ptr = i; 
        //                        curr_shellset->quartet_reorder_ptr++;
        //                }
        //                
        //            }

        //            std::memset(curr_atom_quartet->first_shellset_data_ptr, 0, curr_atom_quartet->n_shellset_quartets * this->staging.data_inc * sizeof(double));

        //            std::cout << "Reordered batch for atom quartet:\n";
        //            for(int i = 0;i < cum_count;i++)
        //            {
        //                std::cout << batch[quartet_reorder_buffer[i]] << '\n';
        //            }
        //        } while(curr_atom_quartet)
        //        return false;

        //        
        //        
        //        //for(q_idx = prev_q_idx;q_idx < batch_size;q_idx++)
        //        //{
        //        //    const quartet_idx& q = batch[q_idx]; 
        //        //    //std::cout << q << std::endl;
        //        //    switch(q.diff)
        //        //    {
        //        //    case 0: // New atom quartet
        //        //        this->A = this->sim_box.positions(Eigen::all, q.atom[0]);  
        //        //        this->B = this->sim_box.positions(Eigen::all, q.atom[1]);  
        //        //        this->C = this->sim_box.positions(Eigen::all, q.atom[2]);  
        //        //        this->D = this->sim_box.positions(Eigen::all, q.atom[3]); 

        //        //        this->kind_set_off(0) = this->sim_box.kind_set_offsets(q.atom[0]);
        //        //        this->kind_set_off(1) = this->sim_box.kind_set_offsets(q.atom[1]);
        //        //        this->kind_set_off(2) = this->sim_box.kind_set_offsets(q.atom[2]);
        //        //        this->kind_set_off(3) = this->sim_box.kind_set_offsets(q.atom[3]);
        //        //        
        //        //        this->kind_exp_off(0) = this->sim_box.kind_exp_offsets(q.atom[0]);
        //        //        this->kind_exp_off(1) = this->sim_box.kind_exp_offsets(q.atom[1]);
        //        //        this->kind_exp_off(2) = this->sim_box.kind_exp_offsets(q.atom[2]);
        //        //        this->kind_exp_off(3) = this->sim_box.kind_exp_offsets(q.atom[3]);
        //        //       
        //        //        //Insert descriptor into staging index
        //        //        if(curr_atom_quartet)curr_atom_quartet.finish();
        //        //        curr_atom_quartet = this->staging.reserve(q.atom[0], q.atom[1], q.atom[2], q.atom[3]);
        //        //        if(!curr_atom_quartet)break;
        //        //        this->shellset_id_to_acc_ptr.clear(); 

        //        //        //std::cout << "kind_set_off = (" << kind_set_off(0) << ", " << kind_set_off(1) << ", " << kind_set_off(2) << ", " << kind_set_off(3) << ")" << std::endl;
        //        //        //std::cout << "kind_exp_off = (" << kind_exp_off(0) << ", " << kind_exp_off(1) << ", " << kind_exp_off(2) << ", " << kind_exp_off(3) << ")" << std::endl;
        //        //        [[fallthrough]];
        //        //    case 1: // New B offset
        //        //        B_off = Eigen::Vector3d(double(q.off[0]), double(q.off[1]), double(q.off[2]));
        //        //        this->BB = this->B + this->sim_box.cell_mat * B_off;
        //        //        this->ABB = this->A - this->BB;
        //        //        this->ABB2 = ABB.dot(this->ABB);
        //        //        [[fallthrough]];
        //        //    case 2: // New D offset 
        //        //        D_off = Eigen::Vector3d(double(q.off[3]), double(q.off[4]), double(q.off[5]));
        //        //        this->DD = this->D + this->sim_box.cell_mat * D_off;
        //        //        this->CDD = this->C - this->DD;
        //        //        this->CDD2 = this->CDD.dot(this->CDD); 
        //        //        [[fallthrough]];
        //        //    case 3: // New set quartet
        //        //        this->set_off(0) = this->kind_set_off(0) + q.set[0];
        //        //        this->set_off(1) = this->kind_set_off(1) + q.set[1];
        //        //        this->set_off(2) = this->kind_set_off(2) + q.set[2];
        //        //        this->set_off(3) = this->kind_set_off(3) + q.set[3];
        //        //        
        //        //        this->exp_off(0) = this->kind_exp_off(0) + this->sim_box.set_offsets(q.set[0]);
        //        //        this->exp_off(1) = this->kind_exp_off(1) + this->sim_box.set_offsets(q.set[1]);
        //        //        this->exp_off(2) = this->kind_exp_off(2) + this->sim_box.set_offsets(q.set[2]);
        //        //        this->exp_off(3) = this->kind_exp_off(3) + this->sim_box.set_offsets(q.set[3]);
        //        //        //std::cout << "set_off = (" << set_off(0) << ", " << set_off(1) << ", " << set_off(2) << ", " << set_off(3) << ")" << std::endl;
        //        //        //std::cout << "exp_off = (" << exp_off(0) << ", " << exp_off(1) << ", " << exp_off(2) << ", " << exp_off(3) << ")" << std::endl;
        //        //        set_id = uint32_t(q.set[0])
        //        //               | (uint32_t(q.set[1]) << 8)
        //        //               | (uint32_t(q.set[2]) << 16)
        //        //               | (uint32_t(q.set[3]) << 24);
        //        //        [[fallthrough]];
        //        //    case 4: // New shell quartet
        //        //        this->shl_off(0) = this->lshl_off(0) + q.shl[0];
        //        //        this->shl_off(1) = this->lshl_off(1) + q.shl[1];
        //        //        this->shl_off(2) = this->lshl_off(2) + q.shl[2];
        //        //        this->shl_off(3) = this->lshl_off(3) + q.shl[3];
        //        //        //std::cout << "shl_off = (" << shl_off(0) << ", " << shl_off(1) << ", " << shl_off(2) << ", " << shl_off(3) << ")" << std::endl;
        //        //        this->norm_prod = this->sim_box.normalizations(this->set_off(0), this->shl_off(0))
        //        //                        * this->sim_box.normalizations(this->set_off(1), this->shl_off(1))
        //        //                        * this->sim_box.normalizations(this->set_off(2), this->shl_off(2))
        //        //                        * this->sim_box.normalizations(this->set_off(3), this->shl_off(3));
        //        //        shell_id = uint32_t(q.shl[0])
        //        //                 | (uint32_t(q.shl[1]) << 8)
        //        //                 | (uint32_t(q.shl[2]) << 16)
        //        //                 | (uint32_t(q.shl[3]) << 24);
        //        //        shellset_id = uint64_t(set_id) | (uint64_t(shell_id) << 32);
        //        //        shellset_id_it = shellset_id_to_acc_ptr.find(shellset_id);
        //        //        if(shellset_id_it == shellset_id_to_acc_ptr.end())
        //        //        {
        //        //            curr_acc_ptr = (double*)byte_add(curr_atom_quartet.first_shellset_data_ptr, curr_atom_quartet.n_shellset_quartets * shellset_quartet_size);
        //        //            curr_atom_quartet.n_shellset_quartets++;
        //        //            curr_atom_quartet.first_shellset_ptr[-int64_t(curr_atom_quartet.n_shellset_quartets)].shellset_id = shellset_id;
        //        //            shellset_id_to_acc_ptr.emplace(shellset_id, curr_acc_ptr);
        //        //            std::memset(curr_acc_ptr, 0, shellset_quartet_size);
        //        //        }
        //        //        else
        //        //        {
        //        //            curr_acc_ptr = shellset_id_it->second;
        //        //        } 
        //        //        [[fallthrough]];
        //        //    case 5: // New exp quartet 
        //        //        this->coeff_prod = this->sim_box.coefficients(this->exp_off(0) + q.exp[0], this->shl_off(0))
        //        //                         * this->sim_box.coefficients(this->exp_off(1) + q.exp[1], this->shl_off(1))
        //        //                         * this->sim_box.coefficients(this->exp_off(2) + q.exp[2], this->shl_off(2))
        //        //                         * this->sim_box.coefficients(this->exp_off(3) + q.exp[3], this->shl_off(3));
        //        //        this->alpha = this->sim_box.exponents(this->exp_off(0) + q.exp[0]);
        //        //        this->beta  = this->sim_box.exponents(this->exp_off(1) + q.exp[1]);
        //        //        this->gamma = this->sim_box.exponents(this->exp_off(2) + q.exp[2]);
        //        //        this->delta = this->sim_box.exponents(this->exp_off(3) + q.exp[3]);

        //        //        this->zeta = this->alpha + this->beta;
        //        //        this->eta  = this->gamma + this->delta;
        //        //        this->chi  = this->zeta  + this->eta;
        //        //        this->pi_over_chi = std::numbers::pi / chi;
        //        //        this->rho  = (this->alpha * this->beta) / this->chi;
        //        //        this->P = (this->alpha * A + this->beta  * BB) / this->zeta;
        //        //        this->Q = (this->gamma * C + this->delta * DD) / this->eta;
        //        //        this->PA = this->P - this->A;
        //        //        this->QC = this->Q - this->C; //Q_off does not affect this vector
        //        //        [[fallthrough]];
        //        //    case 6:
        //        //        Q_off = Eigen::Vector3d(double(q.off[6]), double(q.off[7]), double(q.off[8]));
        //        //        this->QQ = this->Q + this->sim_box.cell_mat * Q_off;
        //        //        this->PQQ = P - QQ;
        //        //        this->W = (this->zeta * this->P + this->eta * this->QQ) / this->chi;
        //        //        this->WP = this->W - this->P;
        //        //        this->WQ = this->W - this->Q;
        //        //        this->PQQ2 = this->PQQ.dot(this->PQQ);
        //        //        this->rhopq2 = this->rho * this->PQQ2;
        //        //        this->prefactor = std::exp((this->alpha * this->beta  * this->ABB2) / this->zeta 
        //        //                                  +(this->gamma * this->delta * this->CDD2) / this->eta)
        //        //                        * std::sqrt(pi_over_chi * pi_over_chi * pi_over_chi)
        //        //                        * norm_prod * coeff_prod;

        //        //        this->curr_eng_idx = 0; // No early contraction for now
        //        //        this->curr_eng_desc = &(this->eng_descs[this->curr_eng_idx]);
        //        //        this->curr_slot = this->curr_eng_desc->slots_used;
        //        //        this->curr_eng_desc->slots[this->curr_slot].acc_ptr = curr_acc_ptr;
        //        //        this->curr_eng_triag_idx = this->curr_eng_desc->first_engine;
        //        //        this->curr_eng = &(this->engines[this->curr_eng_triag_idx]);

        //        //        *(((double*)&(this->curr_eng->AB_x[0])) + this->curr_slot) = this->ABB[0];
        //        //        *(((double*)&(this->curr_eng->AB_y[0])) + this->curr_slot) = this->ABB[1];
        //        //        *(((double*)&(this->curr_eng->AB_z[0])) + this->curr_slot) = this->ABB[2];
        //        //        
        //        //        *(((double*)&(this->curr_eng->CD_x[0])) + this->curr_slot) = this->CDD[0];
        //        //        *(((double*)&(this->curr_eng->CD_y[0])) + this->curr_slot) = this->CDD[1];
        //        //        *(((double*)&(this->curr_eng->CD_z[0])) + this->curr_slot) = this->CDD[2];
        //        //        
        //        //        *(((double*)&(this->curr_eng->PA_x[0])) + this->curr_slot) = this->PA[0];
        //        //        *(((double*)&(this->curr_eng->PA_y[0])) + this->curr_slot) = this->PA[1];
        //        //        *(((double*)&(this->curr_eng->PA_z[0])) + this->curr_slot) = this->PA[2];
        //        //        
        //        //        *(((double*)&(this->curr_eng->QC_x[0])) + this->curr_slot) = this->QC[0];
        //        //        *(((double*)&(this->curr_eng->QC_y[0])) + this->curr_slot) = this->QC[1];
        //        //        *(((double*)&(this->curr_eng->QC_z[0])) + this->curr_slot) = this->QC[2];
        //        //        
        //        //        *(((double*)&(this->curr_eng->WP_x[0])) + this->curr_slot) = this->WP[0];
        //        //        *(((double*)&(this->curr_eng->WP_y[0])) + this->curr_slot) = this->WP[1];
        //        //        *(((double*)&(this->curr_eng->WP_z[0])) + this->curr_slot) = this->WP[2];
        //        //        
        //        //        *(((double*)&(this->curr_eng->WQ_x[0])) + this->curr_slot) = this->WQ[0];
        //        //        *(((double*)&(this->curr_eng->WQ_y[0])) + this->curr_slot) = this->WQ[1];
        //        //        *(((double*)&(this->curr_eng->WQ_z[0])) + this->curr_slot) = this->WQ[2];
        //        //        
        //        //        *(((double*)&(this->curr_eng->oo2z[0]))  + this->curr_slot) = 0.5 / this->zeta;
        //        //        *(((double*)&(this->curr_eng->oo2e[0]))  + this->curr_slot) = 0.5 / this->eta;
        //        //        *(((double*)&(this->curr_eng->oo2ze[0])) + this->curr_slot) = 0.5 / this->chi;
        //        //        *(((double*)&(this->curr_eng->roz[0]))   + this->curr_slot) = this->rho / this->zeta;
        //        //        *(((double*)&(this->curr_eng->roe[0]))   + this->curr_slot) = this->rho / this->eta;

        //        //        this->eng_G_ptr = ((double*)&(this->curr_eng->LIBINT_T_SS_EREP_SS(0)[0])) + this->curr_slot;
        //        //        if(op == potential_operator::coulomb)
        //        //        {
        //        //            this->boys_eval.eval(&(this->G_buffer[0][0]), this->rhopq2, am.max());
        //        //            for(int32_t am_idx = 0;am_idx <= am.max();am_idx++, eng_G_ptr += LIBINT2_REALTYPE::size())
        //        //                *eng_G_ptr = this->prefactor * (two_pi / this->rho) * this->G_buffer[0][am_idx];
        //        //        }
        //        //        else if(op == potential_operator::short_range)
        //        //        {
        //        //            boys_eval.eval(&(G_buffer[0][0]), this->rhopq2, am.max());
        //        //            this->theta = this->op_param / std::sqrt(this->rho + this->op_param * this->op_param);
        //        //            this->theta2 = this->theta * this->theta;
        //        //            boys_eval.eval(&(G_buffer[1][0]), this->theta2 * this->rhopq2, am.max());
        //        //            this->scale = theta;
        //        //            for(int32_t am_idx = 0;am_idx <= am.max();am_idx++, eng_G_ptr += LIBINT2_REALTYPE::size())
        //        //            {
        //        //                *eng_G_ptr = this->prefactor * (two_pi / this->rho) * (this->G_buffer[0][am_idx] - this->scale * this->G_buffer[1][am_idx]);
        //        //                this->scale *= this->theta2;
        //        //            }
        //        //             
        //        //        }
        //        //        else
        //        //        {
        //        //            
        //        //        }
        //        //        this->curr_eng_desc->slots_used++;
        //        //        //std::cout << "Submitting quartet " << q << '\n';
        //        //        if(this->curr_eng_desc->slots_used == LIBINT2_REALTYPE::size())
        //        //        {
        //        //            // Compute
        //        //            //std::cout << "Computing with acc_ptrs = " 
        //        //            //          << this->curr_eng_desc->slots[0].acc_ptr << ", "
        //        //            //          << this->curr_eng_desc->slots[1].acc_ptr << ", "
        //        //            //          << this->curr_eng_desc->slots[2].acc_ptr << ", "
        //        //            //          << this->curr_eng_desc->slots[3].acc_ptr << "; "
        //        //            //          << "curr_atom_quartet = {" 
        //        //            //          << curr_atom_quartet.first_shellset_data_ptr << ", "
        //        //            //          << curr_atom_quartet.first_shellset_ptr << ", "
        //        //            //          << curr_atom_quartet.n_shellset_quartets << "}; "
        //        //            //          << "shellset_id_to_acc_ptr.size() = " << shellset_id_to_acc_ptr.size() << '\n';
        //        //            libheric_libint2_build_eri[this->la][this->lb][this->lc][this->ld](this->curr_eng);
        //        //            for(uint32_t v = 0;v < LIBINT2_REALTYPE::size();v++)
        //        //            {
        //        //                this->res_acc = (double*)this->curr_eng_desc->slots[v].acc_ptr;
        //        //                for(int32_t i = 0;i < this->am.cart_size();i++)
        //        //                {
        //        //                    this->res_acc[i] += this->curr_eng->targets[0][i][v];
        //        //                }
        //        //            }
        //        //            this->curr_eng_desc->slots_used = 0;
        //        //        }
        //        //        //std::cout << "Submitting quartet #" << q_idx << " for shellset_idx = " << this->shellset_id << ": \n";
        //        //        //std::cout << "A = " << this->A.transpose() << " " << this->alpha << '\n';
        //        //        //std::cout << "B = " << this->BB.transpose() << " " << this->beta << '\n';
        //        //        //std::cout << "C = " << (this->C + this->sim_box.cell_mat * this->Q_off).transpose() << " " << this->gamma << '\n';
        //        //        //std::cout << "D = " << (this->DD + this->sim_box.cell_mat * this->Q_off).transpose() << " " << this->delta << '\n';
        //        //    }
        //        //}
        //        if(q_idx == batch_size)
        //        {
        //            if(this->quartet_class_finalizing)
        //            {
        //                std::cout << "Popping final batch of size = " << batch_size << "\n";
        //                if(batch_size == qq.batch_size)this->qq.pop_batch();
        //                else this->qq.reset();
        //                if(this->qq.size() == 0 && this->curr_atom_quartet.n_shellset_quartets != 0)
        //                {
        //                    // Commit final atom_quartet
        //                    this->curr_atom_quartet.done = true;
        //                    staging.commit(this->curr_staging_index, this->curr_atom_quartet, this->shellset_quartet_size);
        //                }
        //            }
        //            else
        //            {
        //                std::cout << "Popping batch\n";
        //                this->qq.pop_batch();
        //            }
        //            this->prev_q_idx = 0;
        //        }
        //        else
        //        {
        //            this->prev_q_idx = q_idx;
        //        }
        //    }
        //    std::cout << "condition = " << (batch != nullptr && batch_size != 0) << ", prev_q_idx = " << prev_q_idx << '\n';
        //} while (batch != nullptr && batch_size != 0);
        //return this->quartet_class_finalizing ? qq.empty() : qq.num_batches_ready() == 0;
        //Eigen::Vector3d A;
        //Eigen::Vector3d B, BB;
        //Eigen::Vector3d C, CC;
        //Eigen::Vector3d D, DD;
        //double alpha, beta, gamma, delta;
        //double zeta, eta, chi;
        //double kappa;

        //double norm_coeff_prod[4];

        //Eigen::Vector3d ABB;
        //Eigen::Vector3d CCDD;
        //Eigen::Vector3d P, Q, W;

        //curr_root = 0;
        //uint32_t ids_left = primitive_tasks.size();
        //uint32_t contr_ids[MAX_CONTR];
        //root_exp_idx = 0;
        //root_k_idx = 0;
        //curr_exp_idx = 0;
        //curr_k_idx = 0;
        //while(ids_left > 0)
        //{
        //    while(num_found < MAX_CONTR)
        //    {
        //        contr_ids[num_found] = curr_k_idx;
        //        num_found++;
        //        curr_exp_idx++;

        //        curr_k_idx = exp_offsets[curr_exp_idx];
        //        while(primitive_tasks[curr_k_idx].done && curr_k_idx < exp_offsets[curr_exp_offsets + 1])curr_k_idx++;
        //        exp_offsets[curr_exp_idx] = curr_k_idx;
        //        while(curr_k_idx < exp_offsets[curr_exp_offsets + 1])
        //        {
        //            if(!primitive_tasks[curr_k_idx].done && primitive_tasks[root_k_idx].off_equal(primitive_tasks[curr_k_idx]))
        //            {
        //                break;
        //            }
        //            curr_k_idx++;
        //        }
        //    }

        //    quartet_input<double> qi;
        //    qi.A() = this->sim_box.positions(Eigen::all, q_idx.pos[0]);       
        //    qi.B() = this->sim_box.positions(Eigen::all, q_idx.pos[0]);       
        //    qi.C() = this->sim_box.positions(Eigen::all, q_idx.pos[0]);       
        //    qi.D() = this->sim_box.positions(Eigen::all, q_idx.pos[0]);       
        //    

        //    for(uint32_t i = 0;i < num_found;i++)
        //    {
        //        
        //    }
        //    uint32_t num_contr = collect_contr()
        //    if(next.diff < 12) // Exp has changed
        //    curr_contr = ...

        //}
        //for(quartet_idx in primitive_tasks)
        //{
        //    switch(quartet_idx.diff)
        //    {
        //    case 0:
        //        A = this->sim_box.positions(Eigen::all, q_idx.pos[0]);       
        //    case 1:
        //        B = this->sim_box.positions(Eigen::all, q_idx.pos[1]);       
        //    case 2:
        //        C = this->sim_box.positions(Eigen::all, q_idx.pos[2]);       
        //    case 3:
        //        D = this->sim_box.positions(Eigen::all, q_idx.pos[3]);       
        //    case 4:
        //    case 5:
        //    case 6:
        //    case 7:
        //    case 8:
        //        alpha = this->sim_box.exponents(q_idx.exp[0]);
        //        norm_coeff_prod[0] = maxl_norm(am.a(), alpha) * sim_box.coefficients(q_idx.exp[0], q_idx.shl[0]);
        //    case 9:
        //        beta  = this->sim_box.exponents(q_idx.exp[1]);
        //        norm_coeff_prod[1] = norm_coeff_prod[0] * maxl_norm(am.b(), beta) * sim_box.coefficients(q_idx.exp[1], q_idx.shl[1]);
        //        zeta  = alpha + beta;
        //        ooz  = 1.0 / zeta;
        //        oo2z = 0.5 * ooz;
        //    case 10:
        //        gamma = this->sim_box.exponents(q_idx.exp[2]);
        //        norm_coeff_prod[2] = norm_coeff_prod[1] * maxl_norm(am.c(), gamma) * sim_box.coefficients(q_idx.exp[2], q_idx.shl[2]);
        //    case 11:
        //        delta = this->sim_box.exponents(q_idx.exp[3]);
        //        norm_coeff_prod[3] = norm_coeff_prod[2] * maxl_norm(am.d(), delta) * sim_box.coefficients(q_idx.exp[3], q_idx.shl[3]);
        //        eta = gamma + delta;
        //        ooe = 1.0 / eta;
        //        oo2e = 0.5 * ooe;
        //        oo2ze = 1.0 / (2.0 * (zeta + eta));
        //        chi  = (zeta * eta) / (zeta + eta);
        //        chioz = chi * ooz;
        //        chioe = chi * ooe;
        //        prefactor = 2.0 * std::pow(std::numbers::pi, 2.5) * std::sqrt(chi / (zeta * eta)) / (zeta * eta);
        //    case 12:
        //        BB = B + sim_box.cell_mat * Eigen::Vector3d(double(q_idx.off[0]), double(q_idx.off[1]), double(q_idx.off[2]));  
        //        AB = (this->libint_perm & 2 == 0) ? A - BB : BB - A;
        //        AB2 = AB.squared_norm();
        //        P = (alpha * A + beta * BB) * ooz;
        //        PA = (this->libint_perm & 2 == 0) ? P - A : P - BB;
        //    case 13:
        //        CC = C + sim_box.cell_mat * Eigen::Vector3d(double(q_idx.off[3]), double(q_idx.off[4]), double(q_idx.off[5]));  
        //    case 14:
        //        DD = D + sim_box.cell_mat * Eigen::Vector3d(double(q_idx.off[6]), double(q_idx.off[7]), double(q_idx.off[8])); 
        //        CD = (this->libint_perm & 1 == 0) ? CC - DD : DD - CC;
        //        CD2 = CCDD.squared_norm();
        //        kappa = std::exp(-((alpha * beta * AB2) * ooz + (gamma * delta * CD2) * ooe));
        //        Q = (gamma * CC + delta * DD) * ooe;
        //        QC = (this->libint_perm & 1 == 0) ? Q - CC : Q - DD;
        //        PQ = P - Q;
        //        PQ2 = PQ.squared_norm();
        //        W = (zeta * P + gamma * Q) / (zeta + eta); 
        //        WP = W - P 
        //        WQ = W - Q
        //        
        //        if(this->libint_perm & 4 == 0)
        //        {
        //            lint_engine.AB_x[0] = AB[0]; 
        //            lint_engine.AB_y[0] = AB[1]; 
        //            lint_engine.AB_z[0] = AB[2]; 
        //            
        //            lint_engine.CD_x[0] = CD[0]; 
        //            lint_engine.CD_y[0] = CD[1]; 
        //            lint_engine.CD_z[0] = CD[2]; 

        //            lint_engine.PA_x[0] = PA[0]; 
        //            lint_engine.PA_y[0] = PA[1]; 
        //            lint_engine.PA_z[0] = PA[2]; 
        //                                              
        //            lint_engine.QC_x[0] = QC[0]; 
        //            lint_engine.QC_y[0] = QC[1]; 
        //            lint_engine.QC_z[0] = QC[2]; 

        //            lint_engine.WP_x[0] = WP[0]; 
        //            lint_engine.WP_y[0] = WP[1]; 
        //            lint_engine.WP_z[0] = WP[2]; 

        //            lint_engine.WQ_x[0] = WQ[0]; 
        //            lint_engine.WQ_y[0] = WQ[1]; 
        //            lint_engine.WQ_z[0] = WQ[2]; 
        //            lint_engine.oo2z [0] = oo2z;
        //            lint_engine.oo2e [0] = oo2e;
        //            lint_engine.roz  [0] = chioz; 
        //            lint_engine.roe  [0] = chioe;
        //        }
        //        else 
        //        {
        //            lint_engine.AB_x[0] = CD[0]; 
        //            lint_engine.AB_y[0] = CD[1]; 
        //            lint_engine.AB_z[0] = CD[2]; 
        //            
        //            lint_engine.CD_x[0] = AB[0]; 
        //            lint_engine.CD_y[0] = AB[1]; 
        //            lint_engine.CD_z[0] = AB[2]; 

        //            lint_engine.PA_x[0] = QC[0]; 
        //            lint_engine.PA_y[0] = QC[1]; 
        //            lint_engine.PA_z[0] = QC[2]; 
        //                                              
        //            lint_engine.QC_x[0] = PA[0]; 
        //            lint_engine.QC_y[0] = PA[1]; 
        //            lint_engine.QC_z[0] = PA[2]; 

        //            lint_engine.WP_x[0] = WQ[0]; 
        //            lint_engine.WP_y[0] = WQ[1]; 
        //            lint_engine.WP_z[0] = WQ[2]; 

        //            lint_engine.WQ_x[0] = WP[0]; 
        //            lint_engine.WQ_y[0] = WP[1]; 
        //            lint_engine.WQ_z[0] = WP[2]; 
        //            lint_engine.oo2z [0] = oo2e;
        //            lint_engine.oo2e [0] = oo2z;
        //            lint_engine.roz  [0] = chioe; 
        //            lint_engine.roe  [0] = chioz;
        //        }
        //        lint_engine.oo2ze[0] = oo2ze;

        //         
        //    }    
        //}
    //}

    std::ostream& operator<<(std::ostream& os, const cpu_acc_context::libint_parameters& lp)
    {
        os << "{ AB = (" << lp.AB_x << ", " << lp.AB_y << ", " << lp.AB_z 
           << "),\nCD = (" << lp.CD_x << ", " << lp.CD_y << ", " << lp.CD_z 
           << "),\nPA = (" << lp.PA_x << ", " << lp.PA_y << ", " << lp.PA_z 
           << "),\nQC = (" << lp.QC_x << ", " << lp.QC_y << ", " << lp.QC_z 
           << "),\nWP = (" << lp.WP_x << ", " << lp.WP_y << ", " << lp.WP_z 
           << "),\nWQ = (" << lp.WQ_x << ", " << lp.WQ_y << ", " << lp.WQ_z 
           << "),\noo2z = " << lp.oo2z
           << ", oo2e = " << lp.oo2e
           << ", oo2ze = " << lp.oo2ze
           << ",\nroz = " << lp.roz
           << ", roe = " << lp.roe
           << ", rho = " << lp.rho
           << ",\noor = " << lp.oor
           << ", op_param = " << lp.op_param
           << ", rhopq2 = " << lp.rhopq2
           << ",\nG = { ";
        for(uint32_t g = 0;g < 4 * MAX_AM + 1;g++)
            os << lp.G[g] << (g == 4 * MAX_AM ? " }}" : ", ");
        return os;
    }
    
    std::ostream& operator<<(std::ostream& os, const cpu_acc_context::engine_descriptor& eng_desc)
    {
        os << "first engine -> " << eng_desc.first_engine << '\n';
        os << "slots = ";
        for(uint64_t i = 0;i < eng_desc.slots.size();i++)
        {
            if(i < eng_desc.slots_used) os << std::setw(16) << i;
            else os << std::string(16, '-');
            os << (i == eng_desc.slots.size() - 1 ? "\n" : " ");
        }
        os << "        "; 
        for(uint64_t i = 0;i < eng_desc.slots.size();i++)
        {
            if(i < eng_desc.slots_used) os << std::setw(16) << eng_desc.slots[i].acc_ptr;
            else os << std::string(16, '-');
            os << (i == eng_desc.slots.size() - 1 ? "\n" : " ");
        }
        return os;
    }
}
