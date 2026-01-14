#ifndef __LIBHERIC_ONEAPI_ACC_H__
#define __LIBHERIC_ONEAPI_ACC_H__

#include "default_acc.h"
#include <optional>
#include <queue>
#include <semaphore>
#include <vector>
#include <sycl/CL/sycl.hpp>
#include <avx2/avx2_decompress.h>
#include <scalar/scalar_decompress.h>

namespace libheric
{
    std::vector<acc_device> discover_oneapi_accelerators(const acc_kind& acc_query); 

    class oneapi_acc_context : public acc_context
    {
    public:
        oneapi_acc_context(const acc_device& acc_dev, 
                           const simulation_box& sim_box); 
        virtual ~oneapi_acc_context();
        virtual void set_quartet(int la, int lb, int lc, int ld);
        virtual void set_quartet_permutation(int perm);
        virtual void set_atom_id(int quartet_pos, int atom_id);
        virtual void set_exponent_id(int quartet_pos, int exp_id);
        virtual void set_offset(int off_pos, int off_x, int off_y, int off_z);
        virtual void submit_peri();
        virtual void submit_ceri();
        virtual void finish_quartet_class();
        virtual void get_next_ceri(double* integral_buffer);

    private:
        std::optional<sycl::device> dev;
        std::optional<sycl::context> ctx;
        std::optional<sycl::queue> q;
	
        int fpga_perm;

		void* lib_handle;
		using Rys4Compute_kernel = 
            void (*)(sycl::queue &,
                     float* /* abcd_dd */, float* /* rtwt_dd  */,
                     uint8_t* /* cERI_dd */, float* /* epsi_dd  */,
                     const std::vector<sycl::event> & /* prev_evt */,
                     sycl::event & /* evt_rys4 */);
        Rys4Compute_kernel kernel;
        
		uint32_t batch_size;
        uint32_t cbits;
		uint32_t n_rtwt; 
		uint32_t n512_per_quartet;
		static constexpr uint64_t n8_per_n512 = 512 / 8;
		static constexpr uint64_t num_abcd_elems = 16;
		static constexpr uint64_t num_rtwt_elems = 16;
		float* host_eri_d[2];
        uint8_t* host_eri_c[2];
        float* host_eps[2];
        float* dev_abcd[2];
        float* dev_rtwt[2];
        uint8_t* dev_eri_c[2];
        float* dev_eps[2];

        double norm_prefactor;
        double norm_coeffs[4];


        rtwt_container<double> rtwt_tmpl;
        std::vector<quartet_input<float>> input_quartets[2];
        std::vector<rtwt_container<float>> input_rtwt[2];
        std::vector<double> quartet_coeffs[2];
        std::vector<uint32_t> ceri_offsets[2];

        sycl::event evt_abcd[2];
        sycl::event evt_rtwt[2];
        std::vector<sycl::event> comp_events;
        sycl::event evt_eri_c[2];
        sycl::event evt_eps[2];

        bool new_batch;
        uint32_t gen_db, consume_db;
        uint32_t ceri_ctr;
        bool new_quartet_class_for_gen, new_quartet_class_for_consume;
        std::counting_semaphore<2> overwrite_quartet_batch;
        std::binary_semaphore start_consuming;

        decompress::AVX2Decompressor<16> decomp;
        decompress::ScalarDecompressor<16> sdecomp;
        std::queue<std::vector<double>> libint_maxl_ref;
        std::queue<std::vector<double>> libint_unit_ref;
        std::queue<std::vector<double>> libint2_ref;
        std::queue<std::vector<double>> rys_ref;

    };
}

#endif
