#ifndef __LIBHERIC_DEFAULT_ACC_H__
#define __LIBHERIC_DEFAULT_ACC_H__

#include "constants.h"
#include "quartet_input.h"
#include "simulation_box.h"

#include <cstdint>
#include <iostream>
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
    
    std::vector<acc_device> discover_cpu_accelerator(const acc_kind& acc_query); 

    class acc_context
    {
    public:
        acc_context(const acc_device& acc_dev, 
                    const simulation_box& sim_box);
        virtual ~acc_context();
		bool functional();

        virtual void set_quartet(int la, int lb, int lc, int ld);
        virtual void set_quartet_permutation(int perm);
        virtual void set_atom_id(int quartet_pos, int atom_id);
        virtual void set_set_id(int quartet_pos, int set_id);
        virtual void set_shell_id(int quartet_pos, int shell_id);
        virtual void set_exponent_id(int quartet_pos, int exp_id);
        virtual void set_offset(int off_pos, int off_x, int off_y, int off_z);
//
        virtual void submit_peri();
        virtual void submit_ceri();
        virtual void finish_quartet_class();
        virtual void get_next_ceri(double* integral_buffer);
         
    protected:
        acc_device dev;
        simulation_box sim_box;

        std::vector<Eigen::SparseMatrix<double>> unc2uns; // unit normalized cartesian to unit normalized spherical
        std::vector<Eigen::SparseMatrix<double>> mnc2uns; // max-L normalized cartesian to unit normalized spherical
        
        int la, lb, lc, ld;
        am_container am;
        Eigen::Vector4i kind_off;
        Eigen::MatrixX4i set_offs;
        Eigen::MatrixX4i set_sizes;
        Eigen::Vector4i set_off;
        Eigen::Vector4i set_size;
        Eigen::Vector4i shell_off;
        Eigen::Vector4i exp_off; 
        
        Eigen::Matrix<double, MAX_SET_SIZE, 4> coeffs;
        Eigen::Vector4d coeff;
        Eigen::Vector4d coeff_prod;

        Eigen::Matrix3i offsets;
        Eigen::Matrix3d offsets_d;

        int perm;
        int libint_perm;
        std::string abcd;
        bool is_functional;

        quartet_input<double>  quartet_tmpl;
        quartet_offset<double> quartet_off;
    };
}
#endif
