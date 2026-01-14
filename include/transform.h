#ifndef __LIBHERIC_TRANSFORM_H__
#define __LIBHERIC_TRANSFORM_H__

#include <cstdint>
#include <functional>
#include <vector>

#include <constants.h> 
#include <quartet_input.h>

#include <Eigen/Sparse>

namespace libheric {
    template<typename FLOAT, int PERM, bool IS_CART>
    std::vector<FLOAT> permute_quartet_scalar(const FLOAT* results, const am_container& am);

    extern std::array<std::function<std::vector<double>(const double*, const am_container&)>, NUM_PERMS> permute_cart_quartet_scalar_double;
    extern std::array<std::function<std::vector<float>(const float*, const am_container&)>, NUM_PERMS> permute_cart_quartet_scalar_single;
    extern std::array<std::function<std::vector<double>(const double*, const am_container&)>, NUM_PERMS> permute_sph_quartet_scalar_double;
    extern std::array<std::function<std::vector<float>(const float*, const am_container&)>, NUM_PERMS> permute_sph_quartet_scalar_single;

    std::vector<double> cart2sph(const am_container& am, const std::vector<Eigen::SparseMatrix<double>>& trans_mats, const std::vector<double>& cart_quartet);
}

#endif 
