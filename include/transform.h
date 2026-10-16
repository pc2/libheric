#ifndef __LIBHERIC_TRANSFORM_H__
#define __LIBHERIC_TRANSFORM_H__

#include <cstdint>
#include <functional>
#include <vector>

#include <constants.h>
#include <math_util.h>
#include <quartet_input.h>

#include <Eigen/Sparse>

namespace libheric {
    template<typename FLOAT, int PERM, bool IS_CART>
    std::vector<FLOAT> permute_quartet_scalar(const FLOAT* results, const am_container& am);

    extern std::array<std::function<std::vector<double>(const double*, const am_container&)>, NUM_PERMS> permute_cart_quartet_scalar_double;
    extern std::array<std::function<std::vector<float>(const float*, const am_container&)>, NUM_PERMS> permute_cart_quartet_scalar_single;
    extern std::array<std::function<std::vector<double>(const double*, const am_container&)>, NUM_PERMS> permute_sph_quartet_scalar_double;
    extern std::array<std::function<std::vector<float>(const float*, const am_container&)>, NUM_PERMS> permute_sph_quartet_scalar_single;


    class quartet_transform
    {
    public:
       enum 
       {
           NONE=0, BRA = 1, KET = 2, BRAKET = 3, 
           TRANSPOSE = 4, BRA_TRANSPOSE = 5, KET_TRANSPOSE = 6, BRAKET_TRANSPOSE = 7
       }; 
    public:
        quartet_transform(bool only_permutation);
        ~quartet_transform();

        void set_quartet_class(const am_container& am);
        void apply_permutation(double* quartet);
        uint64_t mem_usage();
        std::vector<double> naive_cart2sph(const am_container& am, const std::vector<double>& cart_quartet);
    public:
        std::function<void(quartet_transform&, double*, double*)> apply_transformation;
    private:
        template<uint32_t KIND>
        friend void __transform(quartet_transform& qt, double* quartet_input, double* quartet_output);
        void naive_cart2sph_step(int turn, const am_container& am, std::vector<double>& in_quartet, std::vector<double>& out_quartet);

    private:
        std::array<Eigen::SparseMatrix<double>, MAX_AM + 1> c2s; // max-L normalized cartesian to unit normalized spherical
        std::array<std::array<Eigen::SparseMatrix<double>, MAX_AM + 1>, MAX_AM + 1> cc2ss; // max-L normalized cartesian bra to unit normalized spherical bra

        double* transform_buffer1, *transform_buffer2;
        Eigen::SparseMatrix<double> bra_transform;
        Eigen::SparseMatrix<double> ket_transform;
        uint64_t input_bra_dim, tmp_bra_dim, output_bra_dim;
        uint64_t input_ket_dim, tmp_ket_dim, output_ket_dim;
        uint64_t cart_size, sph_size;
        bool only_permutation;

        std::array<std::function<void(quartet_transform&, double*, double*)>, 8> transform_funcs;
    };
}

#endif 
