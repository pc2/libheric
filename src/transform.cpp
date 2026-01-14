#include<transform.h>

namespace libheric {
    template<typename FLOAT, int PERM, bool IS_CART>
    std::vector<FLOAT> permute_quartet_scalar(const FLOAT* results, const am_container& am) // am has its values in the target order
    {
        static_assert(PERM < 8 && PERM >= 0, "PERM not in [0, 7]");
        uint32_t nga = IS_CART ? nco[am.a()] : nso[am.a()];
        uint32_t ngb = IS_CART ? nco[am.b()] : nso[am.b()];
        uint32_t ngc = IS_CART ? nco[am.c()] : nso[am.c()];
        uint32_t ngd = IS_CART ? nco[am.d()] : nso[am.d()];

        uint32_t quartet_size = nga * ngb * ngc * ngd;
        std::vector<FLOAT> perm_results;
        perm_results.reserve(quartet_size);

        for(uint32_t d = 0;d < ngd;d++)
        {
            for(uint32_t c = 0;c < ngc;c++)
            {
                for(uint32_t b = 0;b < ngb;b++)
                {
                    for(uint32_t a = 0;a < nga;a++)
                    {
                        if constexpr(PERM == 0) // (ab|cd) -> (ab|cd)
                            perm_results.push_back(results[(((d * ngc) + c) * ngb + b) * nga + a]);
                        else if constexpr(PERM == 1) // (ab|dc) -> (ab|cd)
                            perm_results.push_back(results[(((c * ngd) + d) * ngb + b) * nga + a]);
                        else if constexpr(PERM == 2) // (ba|cd) -> (ab|cd)
                            perm_results.push_back(results[(((d * ngc) + c) * nga + a) * ngb + b]);
                        else if constexpr(PERM == 3) // (ba|dc) -> (ab|cd)
                            perm_results.push_back(results[(((c * ngd) + d) * nga + a) * ngb + b]);
                        else if constexpr(PERM == 4) // (cd|ab) -> (ab|cd)
                            perm_results.push_back(results[(((b * nga) + a) * ngd + d) * ngc + c]);
                        else if constexpr(PERM == 5) // (dc|ab) -> (ab|cd)
                            perm_results.push_back(results[(((b * nga) + a) * ngc + c) * ngd + d]);
                        else if constexpr(PERM == 6) // (cd|ba) -> (ab|cd)
                            perm_results.push_back(results[(((a * ngb) + b) * ngd + d) * ngc + c]);
                        else if constexpr(PERM == 7) // (dc|ba) -> (ab|cd)
                            perm_results.push_back(results[(((a * ngb) + b) * ngc + c) * ngd + d]);
                    }
                }
            }
        }
        return perm_results;
    }

    std::array<std::function<std::vector<double>(const double*, const am_container&)>, NUM_PERMS> permute_cart_quartet_scalar_double = {
        permute_quartet_scalar<double, 0, true>,
        permute_quartet_scalar<double, 1, true>,
        permute_quartet_scalar<double, 2, true>,
        permute_quartet_scalar<double, 3, true>,
        permute_quartet_scalar<double, 4, true>,
        permute_quartet_scalar<double, 5, true>,
        permute_quartet_scalar<double, 6, true>,
        permute_quartet_scalar<double, 7, true>
    };

    std::array<std::function<std::vector<float>(const float*, const am_container&)>, NUM_PERMS> permute_cart_quartet_scalar_single = {
        permute_quartet_scalar<float, 0, true>,
        permute_quartet_scalar<float, 1, true>,
        permute_quartet_scalar<float, 2, true>,
        permute_quartet_scalar<float, 3, true>,
        permute_quartet_scalar<float, 4, true>,
        permute_quartet_scalar<float, 5, true>,
        permute_quartet_scalar<float, 6, true>,
        permute_quartet_scalar<float, 7, true>
    };

    std::array<std::function<std::vector<double>(const double*, const am_container&)>, NUM_PERMS> permute_sph_quartet_scalar_double = {
        permute_quartet_scalar<double, 0, false>,
        permute_quartet_scalar<double, 1, false>,
        permute_quartet_scalar<double, 2, false>,
        permute_quartet_scalar<double, 3, false>,
        permute_quartet_scalar<double, 4, false>,
        permute_quartet_scalar<double, 5, false>,
        permute_quartet_scalar<double, 6, false>,
        permute_quartet_scalar<double, 7, false>
    };

    std::array<std::function<std::vector<float>(const float*, const am_container&)>, NUM_PERMS> permute_sph_quartet_scalar_single = {
        permute_quartet_scalar<float, 0, false>,
        permute_quartet_scalar<float, 1, false>,
        permute_quartet_scalar<float, 2, false>,
        permute_quartet_scalar<float, 3, false>,
        permute_quartet_scalar<float, 4, false>,
        permute_quartet_scalar<float, 5, false>,
        permute_quartet_scalar<float, 6, false>,
        permute_quartet_scalar<float, 7, false>
    };

    void cart2sph_step(int turn, const am_container& am, std::vector<double>& in_quartet, const Eigen::SparseMatrix<double>& trans_mat, std::vector<double>& out_quartet)
    {
        int32_t nga = turn <= 3 ? nco[am.raw[(0 + turn) % 4]] : nso[am.raw[(0 + turn) % 4]];
        int32_t ngb = turn <= 2 ? nco[am.raw[(1 + turn) % 4]] : nso[am.raw[(1 + turn) % 4]];
        int32_t ngc = turn <= 1 ? nco[am.raw[(2 + turn) % 4]] : nso[am.raw[(2 + turn) % 4]];
        int32_t ngd = turn <= 0 ? nco[am.raw[(3 + turn) % 4]] : nso[am.raw[(3 + turn) % 4]];
        
        int32_t out_nga = nso[am.raw[(0 + turn) % 4]];

        for(int32_t d = 0;d < ngd;d++)
        {
            for(int32_t c = 0;c < ngc;c++)
            {
                for(int32_t b = 0;b < ngb;b++)
                {
                    int32_t dcb_idx = (d * ngc + c) * ngb + b;
                    Eigen::Map<Eigen::VectorXd> a_vec(&in_quartet[dcb_idx * nga], nga);
                    auto a_out = trans_mat * a_vec;
                    for(int32_t a = 0;a < out_nga;a++)
                    {
                        out_quartet[a * ngb * ngc * ngd + dcb_idx] = a_out[a];
                    }
                }
            }
        }

    }

    std::vector<double> cart2sph(const am_container& am, const std::vector<Eigen::SparseMatrix<double>>& trans_mats, const std::vector<double>& cart_quartet)
    {
        std::vector<double> tmp1 = cart_quartet, tmp2(am.cart_size());
        std::vector<double> sph_quartet(am.sph_size());
        cart2sph_step(0, am, tmp1, trans_mats[am.raw[0]], tmp2);
        cart2sph_step(1, am, tmp2, trans_mats[am.raw[1]], tmp1);
        cart2sph_step(2, am, tmp1, trans_mats[am.raw[2]], tmp2);
        cart2sph_step(3, am, tmp2, trans_mats[am.raw[3]], sph_quartet);
        return sph_quartet;
    }
}
