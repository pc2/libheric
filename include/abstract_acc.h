#ifndef __LIBHERIC_DEFAULT_ACC_H__
#define __LIBHERIC_DEFAULT_ACC_H__

#include <constants.h>
#include <logging.h>
#include <quartet_input.h>
#include <quartet_queue.h>
#include <simulation_box.h>
#include <transform.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace libheric
{
    struct acc_kind
    {
        std::string name;
        std::string type; 
        std::string library;
        friend bool operator<(const acc_kind& lhs, const acc_kind& rhs);
        friend std::ostream& operator<<(std::ostream& os, const acc_kind& a); 
    };
    struct acc_device
    {
        std::string name;
        std::string bdf;
        std::string cpu_name;
        int num_cores;
        acc_kind kind;
    };
    
    class abstract_acc_context
    {
    public:
        struct shellset_quartet_descriptor
        {
            std::array<uint8_t, 8> shellset;
            union {
                double scaling; // In the cache it makes more sense to directly store the scaling factor
                double pmax;
            };

        };
        
        struct atom_quartet_descriptor
        {
            void* first_shellset_data_ptr;
            std::array<uint8_t, 8>* first_shellset_ptr; // points the first element of the shellset_quartet_descriptor struct in a SoA layout
                                                        // The pointer to the scaling/pmax union is implicit, i.e. first_scaling_ptr = first_shellset_ptr + max_shellset_quartets 
            uint64_t n_shellset_quartets;
        };


        struct staging_index
        {
            std::array<uint32_t, 4> key;
            atom_quartet_descriptor value;
            union {
                uint64_t prims_requested;
                uint64_t eris_allocated_per_shellset;
            };
            uint64_t prims_computed;
            bool is_transformed;
            bool is_finished() {return prims_requested == prims_computed;}
            friend std::ostream& operator<<(std::ostream& os, const staging_index& stg_idx);

        };
        
        static_assert(sizeof(shellset_quartet_descriptor) == 2 * sizeof(uint64_t), "shellset_quartet_descriptor has unexpected size");

        struct cache_index
        {
            std::array<uint32_t, 4> key;
            atom_quartet_descriptor value;
        };

        struct quartet_cache
        {
            quartet_cache() = default;
            quartet_cache(uint64_t cache_size);
            ~quartet_cache();
            void reset();
            int64_t size_remaining();
            cache_index* reserve(const std::array<uint32_t, 4>& idx, uint64_t n_shellset_quartets,
                                      uint64_t num_data_bytes, uint64_t data_align = 0);
            
            uint64_t mem_pool_size; 
            void* mem_pool;
            void* data_ptr;
            cache_index* index_ptr;
        };

        struct quartet_staging
        {
            quartet_staging();
            ~quartet_staging();
            bool full();
            bool empty();
            bool reset(const am_container& am, uint64_t max_shellset_quartets);
            staging_index* reserve(uint32_t A, uint32_t B, uint32_t C, uint32_t D);
            staging_index* get_current_index();
            atom_quartet_descriptor* get_transform_descriptor();
            staging_index* get_next_atom_quartet();
            void* mem_pool;
        	static constexpr uint64_t mem_pool_size = 64 * 1024 * 1024; // 64MiB
            uint64_t n_staging;
            uint64_t read_idx;
            uint64_t write_idx;
            uint64_t eris_allocated_per_shellset;
            uint32_t am_enc32;
            staging_index* index_ptr;
        };
    public:
        abstract_acc_context(const acc_device& acc_dev, 
                             const simulation_box& sim_box,
                             uint32_t ctx_id,
                             uint64_t mem_available,
                             bool compute_raw_cartesian,
                             abstract_acc_context* master_ctx);
        
        virtual ~abstract_acc_context();
		bool functional();

        virtual void set_potential_operator(potential_operator op, double op_param);
        virtual void set_quartet_class(int la, int lb, int lc, int ld);
        virtual void set_atom_ids(int atom_a, int atom_b, int atom_c, int atom_d);
        virtual void set_set_ids(int set_a, int set_b, int set_c, int set_d, double pmax);
        virtual void set_shell_ids(int shl_a, int shl_b, int shl_c, int shl_d);
        virtual void set_B_offset(int off_x, int off_y, int off_z) = 0;
        virtual void set_D_offset(int off_x, int off_y, int off_z) = 0;
        virtual void set_exponent_ids(int exp_a, int exp_b, int exp_c, int exp_d) = 0;
        virtual void set_Q_offset(int off_x, int off_y, int off_z) = 0;
        virtual void submit_peri() = 0;

        virtual void finish_quartet_class();

        // returns whether there is a staged quartet present in integral buffer
        virtual staging_index* get_next_staged_atom_quartet();
        // returns whether there is quartet (AB|CD) in the cache
        bool prefetch_cached_atom_quartet(int A, int B, int C, int D);

        void update_positions(Eigen::Matrix3Xd new_positions);
        void update_cell(Eigen::Matrix3d new_cell_mat);
        
    protected:
        atom_quartet_descriptor get_reference_atom_quartet(const std::array<uint32_t, 4>& atom_quartet, const am_container& am);
    protected:
        acc_device dev;
        simulation_box sim_box;
        uint32_t ctx_id;
        std::ostream* logger;

        potential_operator op;
        double op_param;
        am_container am;
        uint32_t total_am;
        uint8_t input_perm;
        uint8_t output_perm;
        Eigen::Vector4i lkind; 
        Eigen::Vector4i lkindset;
        Eigen::Vector4i exp_off;
        Eigen::Vector4i nexp;
        Eigen::Vector4i norm_off;
        Eigen::Vector4i coeff_off_base;
        Eigen::Vector4i coeff_off;
        std::array<uint8_t, 8> shellset_id;
        double pmax;
        
        std::string abcd;
        uint32_t max_shellset_quartets;
        uint32_t cart_size, sph_size;
        bool is_functional;
        bool is_initialized;
        bool quartet_class_finalizing;
        bool compute_raw_cartesian;

        //std::mutex task_consumer_callback_mtx;
        //std::condition_variable task_consumer_callback;
        quartet_transform transform;
        quartet_staging staging;
        quartet_cache cache;

        static constexpr std::string_view ref_dir{"/path/to/reference/eris"};
        std::vector<std::string> ref_data_filenames; 
        std::map<std::array<uint32_t, 4>, std::pair<uint64_t, uint64_t>> ref_index;
        uint32_t ref_l_max;
        
        abstract_acc_context* master_ctx;
    };
}
#endif
