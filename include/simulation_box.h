#ifndef __LIBHERIC_SIMULATION_BOX_H__
#define __LIBHERIC_SIMULATION_BOX_H__

#include <constants.h>
#include <quartet_input.h>

#include <array>
#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace libheric
{
    struct simulation_box 
    {

        simulation_box(const Eigen::Matrix3d&  cell_mat, 
                       const Eigen::Matrix3Xd& positions,
                       const std::vector<int>&    kinds,
                       const std::vector<double>& exponents,
                       const std::vector<double>& coefficients,
                       const std::vector<int>&    kind_nset,
                       const std::vector<int>&    set_nexp,
                       const std::vector<int>&    set_nshl,
                       int max_set, int nkind, int l_max, int natom);

        uint32_t max_shellset_quartets(const am_container& am);  // Maximum number of contracted shellset quartets
        uint32_t max_shellprim_quartets(const am_container& am); // Maximum number of primitive quartets for any combination of sets
        uint64_t mem_usage() const;
        void update_positions(const Eigen::Matrix3Xd& new_positions);
        void update_cell(const Eigen::Matrix3d& new_cell_mat);

        Eigen::Matrix3d  cell_mat; 
        Eigen::Matrix3Xd positions;
        std::vector<int>    kinds;
        std::vector<double> exponents;
        std::vector<double> coefficients;
        std::vector<int>    kind_nset; // Number of sets in kind
        std::vector<int>    set_nexp;  // Number of exponents in set of specific kind
        std::vector<int>    set_nshl;  // Number of shells in set of specific kind
        
        std::vector<int>    coeff_offsets; 
        std::vector<int>    exp_offsets;   
        std::vector<int>    norm_offsets;  
        
        std::vector<double> normalizations;

        int32_t max_set, nkind, l_max, natom;
        
    private:
        std::array<uint32_t, MAX_AM + 1> max_shellsets_for_l; 
        std::array<uint32_t, MAX_AM + 1> max_shellprims_for_l; 

    };
}
#endif
