#ifndef __LIBHERIC_CPU_ACC_H__
#define __LIBHERIC_CPU_ACC_H__

#include <abstract_acc.h>
#include <math_util.h>

#include <queue>
#include <vector>

#include <libint2.h>
#include <libint2/boys.h>

namespace libheric
{
    std::vector<acc_device> discover_cpu_accelerator(const acc_kind& acc_query); 

    class cpu_acc_context : public abstract_acc_context
    {
    public:
        cpu_acc_context(const acc_device& acc_dev, 
                        const simulation_box& sim_box,
                        uint32_t ctx_id,
                        uint64_t mem_available,
                        bool compute_raw_cartesian,
                        abstract_acc_context* master_ctx);
        virtual ~cpu_acc_context();
       
        virtual void set_potential_operator(potential_operator op, double op_param) override; 
        virtual void set_quartet_class(int la, int lb, int lc, int ld) override;
        virtual void set_atom_ids(int atom_a, int atom_b, int atom_c, int atom_d) override;
        virtual void set_shell_ids(int shl_a, int shl_b, int shl_c, int shl_d) override;
        virtual void set_B_offset(int off_x, int off_y, int off_z) override;
        virtual void set_D_offset(int off_x, int off_y, int off_z) override;
        virtual void set_exponent_ids(int exp_a, int exp_b, int exp_c, int exp_d) override;
        virtual void set_Q_offset(int off_x, int off_y, int off_z) override;
        virtual void submit_peri() override;
      
        virtual void finish_quartet_class(); 
        virtual staging_index* get_next_staged_atom_quartet();

    public:
        struct engine_descriptor
        {
            struct engine_slot
            {
                void* acc_ptr;
                staging_index* staging_atom_ptr;  
            };

            std::array<engine_slot, LIBINT2_REALTYPE::size()> slots;
            uint32_t slots_used;
            uint32_t first_engine;
        };
        friend std::ostream& operator<<(std::ostream& os, const engine_descriptor& eng_desc); 

        struct libint_parameters
        {
            double AB_x, AB_y, AB_z;
            double CD_x, CD_y, CD_z;
            double PA_x, PA_y, PA_z;
            double QC_x, QC_y, QC_z;
            double oo2z, oo2e, oo2ze;
            double roz, roe, rho;
            double oor, op_param;
            // These parameters change in every invocation of submit_peri, 
            // so they do not need to be copied
            double rhopq2;
            double WP_x, WP_y, WP_z;
            double WQ_x, WQ_y, WQ_z;
            double G[4 * MAX_AM + 1];
            friend std::ostream& operator<<(std::ostream& os, const libint_parameters& lp);
        };
    private:
        static uint64_t libint_size();
        void insert_params_into_engine_slot(Libint_eri_t& engine, const libint_parameters& lparams, uint32_t slot);
        void submit_contracted(uint64_t n_contr);
        void flush_engine(uint32_t eng_id);
    private:
        libint2::FmEval_Chebyshev7<double> boys_eval;
        std::function<void(Libint_eri_t*)> engine_eval;
        am_container libint_input_am, libint_output_am;

        static constexpr libint_parameters lparams0 = { 
            .AB_x=0.0, .AB_y=0.0, .AB_z = 0.0, 
            .CD_x=0.0, .CD_y=0.0, .CD_z = 0.0, 
            .PA_x=0.0, .PA_y=0.0, .PA_z = 0.0, 
            .QC_x=0.0, .QC_y=0.0, .QC_z = 0.0, 
            .oo2z=0.0, .oo2e=0.0, .oo2ze = 0.0, 
            .roz=0.0,  .roe=0.0,  .rho = 0.0, 
            .oor=0.0,  .op_param = 0.0,
            .rhopq2 = 0.0,
            .WP_x=0.0, .WP_y=0.0, .WP_z = 0.0, 
            .WQ_x=0.0, .WQ_y=0.0, .WQ_z = 0.0, 
            .G = {0.0}
        };
        static constexpr uint32_t LIBINT_PARAMETERS_COPY_BYTES = sizeof(libint_parameters) - (1 + 3 + 3 + 4 * MAX_AM + 1) * sizeof(double);
      
        // Engine i has max_contr = 2^i
        static constexpr uint32_t N_ENGINES = 4;
        static constexpr uint64_t N_CONTR_ENGINES = ipow2(N_ENGINES) - 1;
        static constexpr uint64_t N_LPARAMS = ipow2(N_ENGINES - 1);
        std::array<engine_descriptor, N_ENGINES> eng_descs;
        std::array<Libint_eri_t, N_CONTR_ENGINES> engines;
        std::array<libint_parameters, N_LPARAMS> libint_params;
        libint_parameters* curr_libint_params; 

        Eigen::Vector3d A, B, C, D, BB, DD;
        Eigen::Vector3d P, Q;
        double zeta, eta, chi, ooc, rho;
        double ABB2, CDD2;
        double norm_prod, coeff_prod, prefactor;

        abstract_acc_context::staging_index* curr_atom_quartet;
        void* curr_shellset_data_ptr;
        std::array<uint8_t, 8>* curr_shellset_ptr;
        double* curr_pmax_ptr;
        uint64_t curr_shellset_count;

        double G_buffer[4 * MAX_AM + 1];
        double t_libint;


    };
}

#endif
