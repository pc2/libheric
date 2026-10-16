#ifndef __LIBHERIC_CUDA_ACC_H__
#define __LIBHERIC_CUDA_ACC_H__

#include <abstract_acc.h>
#include <vector>
namespace libheric
{
    std::vector<acc_device> discover_cuda_accelerators(const acc_kind& acc_query); 

    class cuda_acc_context : public abstract_acc_context
    {
    public:
        cuda_acc_context(const acc_device& acc_dev, 
                         const simulation_box& sim_box,
                         uint32_t ctx_id,
                         uint64_t mem_available,
                         bool compute_raw_cartesian,
                         abstract_acc_context* master_ctx);
        virtual ~cuda_acc_context();
        
        virtual void set_B_offset(int off_x, int off_y, int off_z) override;
        virtual void set_D_offset(int off_x, int off_y, int off_z) override;
        virtual void set_exponent_ids(int exp_a, int exp_b, int exp_c, int exp_d) override;
        virtual void set_Q_offset(int off_x, int off_y, int off_z) override;
        virtual void submit_peri() override;
    };
}

#endif
