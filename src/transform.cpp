#include <transform.h>

#include <logging.h>

#include <cstdlib>
#include <iomanip>

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
                        else if constexpr(PERM == 5) // (cd|ba) -> (ab|cd)
                            perm_results.push_back(results[(((a * ngb) + b) * ngd + d) * ngc + c]);
                        else if constexpr(PERM == 6) // (dc|ab) -> (ab|cd)
                            perm_results.push_back(results[(((b * nga) + a) * ngc + c) * ngd + d]);
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

    void quartet_transform::naive_cart2sph_step(int turn, const am_container& am, std::vector<double>& in_quartet, std::vector<double>& out_quartet)
    {
        uint32_t nga = turn <= 3 ? nco[am.raw[(0 + turn) % 4]] : nso[am.raw[(0 + turn) % 4]];
        uint32_t ngb = turn <= 2 ? nco[am.raw[(1 + turn) % 4]] : nso[am.raw[(1 + turn) % 4]];
        uint32_t ngc = turn <= 1 ? nco[am.raw[(2 + turn) % 4]] : nso[am.raw[(2 + turn) % 4]];
        uint32_t ngd = turn <= 0 ? nco[am.raw[(3 + turn) % 4]] : nso[am.raw[(3 + turn) % 4]];
        
        uint32_t out_nga = nso[am.raw[(0 + turn) % 4]];

        for(uint32_t d = 0;d < ngd;d++)
        {
            for(uint32_t c = 0;c < ngc;c++)
            {
                for(uint32_t b = 0;b < ngb;b++)
                {
                    uint32_t dcb_idx = (d * ngc + c) * ngb + b;
                    Eigen::Map<Eigen::VectorXd> a_vec(&in_quartet[dcb_idx * nga], nga);
                    auto a_out = this->c2s[am.raw[turn]] * a_vec;
                    for(uint32_t a = 0;a < out_nga;a++)
                    {
                        out_quartet[a * ngb * ngc * ngd + dcb_idx] = a_out[a];
                    }
                }
            }
        }

    }

    std::vector<double> quartet_transform::naive_cart2sph(const am_container& am, const std::vector<double>& cart_quartet)
    {
        std::vector<double> tmp1 = cart_quartet, tmp2(am.cart_size());
        std::vector<double> sph_quartet(am.sph_size());
        this->naive_cart2sph_step(0, am, tmp1, tmp2);
        this->naive_cart2sph_step(1, am, tmp2, tmp1);
        this->naive_cart2sph_step(2, am, tmp1, tmp2);
        this->naive_cart2sph_step(3, am, tmp2, sph_quartet);
        return sph_quartet;
    }
    
    template<uint32_t TRANSFORM_KIND>
    void __transform(quartet_transform& qt, double* quartet_input, double* quartet_output)
    {
        static_assert(TRANSFORM_KIND < 8);
        if constexpr (TRANSFORM_KIND == quartet_transform::NONE) // Only (ss|ss)
        {
            return;
        }
        else 
        {
            Eigen::Map<Eigen::MatrixXd> quartet_input_mat(quartet_input, qt.input_bra_dim, qt.input_ket_dim);
            Eigen::Map<Eigen::MatrixXd, Eigen::AlignmentType::Aligned64> quartet_tmp_mat(qt.transform_buffer1, qt.tmp_bra_dim, qt.tmp_ket_dim);
            Eigen::Map<Eigen::MatrixXd> quartet_output_mat(quartet_output, qt.output_bra_dim, qt.output_ket_dim);

            if constexpr (TRANSFORM_KIND == quartet_transform::BRA) 
            {
                //std::cout << "Input = \n" << quartet_input_mat << '\n';
                quartet_tmp_mat.noalias() = qt.bra_transform * quartet_input_mat;
                //std::cout << "Bra_trans = \n" << quartet_tmp_mat << '\n';
                quartet_output_mat = quartet_tmp_mat;
            }
            else if constexpr (TRANSFORM_KIND == quartet_transform::KET) 
            {
                //std::cout << "Input = \n" << quartet_input_mat << '\n';
                quartet_tmp_mat.noalias() = quartet_input_mat * qt.ket_transform;
                //std::cout << "Ket_trans = \n" << quartet_tmp_mat << '\n';
                quartet_output_mat = quartet_tmp_mat;
            }
            else if constexpr (TRANSFORM_KIND == quartet_transform::BRAKET)
            {
                //std::cout << "Input = \n" << quartet_input_mat << '\n';
                quartet_tmp_mat.noalias() = qt.bra_transform * quartet_input_mat;
                //std::cout << "Bra_trans = \n" << quartet_tmp_mat << '\n';
                quartet_output_mat.noalias() = quartet_tmp_mat * qt.ket_transform;
                //std::cout << "Ket_trans = \n" << quartet_output_mat << '\n';
            }
            else if constexpr (TRANSFORM_KIND == quartet_transform::TRANSPOSE) // Should never occur
            {
                quartet_tmp_mat = quartet_input_mat.transpose();
                quartet_output_mat = quartet_tmp_mat; 
            }
            else if constexpr (TRANSFORM_KIND == quartet_transform::BRA_TRANSPOSE)
            {
                //std::cout << "Input = \n" << quartet_input_mat << '\n';
                quartet_tmp_mat.noalias() = qt.bra_transform * quartet_input_mat;
                //std::cout << "Bra_trans = \n" << quartet_tmp_mat << '\n';
                quartet_output_mat = quartet_tmp_mat.transpose();
                //std::cout << "Tranpose = \n" << quartet_output_mat << '\n';
            } 
            else if constexpr (TRANSFORM_KIND == quartet_transform::KET_TRANSPOSE)
            {
                quartet_tmp_mat.noalias() = quartet_input_mat * qt.ket_transform;
                quartet_output_mat = quartet_tmp_mat.transpose();
            } 
            else if constexpr (TRANSFORM_KIND == quartet_transform::BRAKET_TRANSPOSE)
            {
                Eigen::Map<Eigen::MatrixXd, Eigen::AlignmentType::Aligned64> quartet_tmp_mat2(qt.transform_buffer2, qt.output_ket_dim, qt.output_bra_dim);
                //std::cout << "Input = \n" << quartet_input_mat << '\n';
                quartet_tmp_mat.noalias() = qt.bra_transform * quartet_input_mat; 
                //std::cout << "Bra_trans = \n" << quartet_tmp_mat << '\n';
                quartet_tmp_mat2.noalias() = quartet_tmp_mat * qt.ket_transform;
                //std::cout << "Ket_trans = \n" << quartet_tmp_mat2 << '\n';
                quartet_output_mat = quartet_tmp_mat2.transpose(); 
                //std::cout << "Tranpose = \n" << quartet_output_mat << '\n';
            }
        } 
    }

    quartet_transform::quartet_transform(bool only_permutation)
        : apply_transformation(__transform<0>),
          transform_buffer1((double*)std::aligned_alloc(64, ipow(nco[MAX_AM], 4) * sizeof(double))),
          transform_buffer2((double*)std::aligned_alloc(64, ipow(nco[MAX_AM], 4) * sizeof(double))),
          only_permutation(only_permutation),
          transform_funcs({
                              __transform<0>,
                              __transform<1>,
                              __transform<2>,
                              __transform<3>,
                              __transform<4>,
                              __transform<5>,
                              __transform<6>,
                              __transform<7>
                          })
    {
        // Initialize spherical transformation matrices

        if(only_permutation)
        {
            // Initialize c2s and cc2ss as identity matrices
            for(int32_t l1 = 0;l1 <= MAX_AM;l1++)
            {
                c2s[l1] = Eigen::MatrixXd::Identity(nco[l1], nco[l1]).sparseView();
                for(int32_t l2 = 0;l2 <= MAX_AM;l2++)
                {
                    cc2ss[l1][l2] = Eigen::MatrixXd::Identity(nco[l1] * nco[l2], nco[l1] * nco[l2]).sparseView();
                    if constexpr(DEBUG_PRINT)
                    {
                        std::ostream* logger = get_logger();
                        *logger << " l1 = " << l1 << ", l2 = " << l2 << '\n';
                        *logger << cc2ss[l1][l2] << '\n';
                    }
                }
            }
            return;
        }
        // else

        for(int32_t l = 0;l <= MAX_AM;l++)
        {
            std::vector<Eigen::Triplet<double>> c2s_trps;
            for(int32_t l1 = 0, cart_idx = 0;l1 <= l;l1++)
            {
                for(int32_t l2 = 0;l2 <= l1;l2++,cart_idx++)
                {
                    int32_t lx = l - l1;
                    int32_t ly = l1 - l2;
                    int32_t lz = l2;
                    
                    // Each m > 0 generates two coefficients!
                    for(int32_t m = 0;m <= l;m++)
                    {
                        int32_t j = (lx + ly - m);
                        if(modulo(j, 2) == 1)continue;
                        j /= 2;
                        double maxl_norm = std::sqrt(double(factorial(l - m)) / double(factorial(l + m)));
                        double common_factor = 1.0 / (powern(2.0, l) * double(factorial(l)));

                        maxl_norm *= common_factor;

                        double outer_sum1 = 0;
                        double outer_sum2 = 0;
                        for(int32_t i = 0;i <= (l - m)/2;i++)
                        {
                            double outer_factor = (i % 2 == 0 ? 1 : -1) * binom(l, i) * binom(i, j) 
                                                * (double(factorial(2 * l - 2 * i)) / double(factorial(l - m - 2 * i)));
                            double inner_sum1 = 0;
                            double inner_sum2 = 0;
                            for(int32_t k = 0;k <= j;k++)
                            {
                                double inner_sum_term = binom(j, k) * binom(m, lx - 2 * k);
                                // Y_0^l is real anyways
                                if(m == 0 && lx == 2 * k)
                                {
                                    inner_sum1 += inner_sum_term;
                                    inner_sum2 += 0;
                                }
                                else 
                                {
                                    // Use (Y_{m}^l + Y_{-m}^l) / sqrt(2) for real coefficient
                                    if(modulo(m - lx, 2) == 0)
                                    {
                                        if(modulo(m - lx + 2 * k, 4) == 0) inner_sum1 +=  std::sqrt(2) * inner_sum_term;
                                        else                               inner_sum1 += -std::sqrt(2) * inner_sum_term;
                                    }
                                    // Use (Y_{m}^l - Y_{-m}^l) / (i * sqrt(2)) for real coefficient
                                    else //if(modulo(m - lx, 2) == 1)
                                    {
                                        if(modulo(m - lx + 2 * k, 4) == 1) inner_sum2 +=  std::sqrt(2) * inner_sum_term;
                                        else                               inner_sum2 += -std::sqrt(2) * inner_sum_term;
                                    }
                                }
                            }
                            outer_sum1 += outer_factor * inner_sum1;
                            outer_sum2 += outer_factor * inner_sum2;
                        }
                        if(outer_sum1 != 0)
                        {
                            //std::cout << "non-zero from Y_{m} + Y_{-m} (" << outer_sum1 * maxl_norm << ") for l = (" << lx << ", " << ly << ", " << lz << "), " << " m = " << m << std::endl;
                            int32_t sph_idx = l + m;
                            c2s_trps.push_back({sph_idx, cart_idx, outer_sum1 * maxl_norm});
                        }
                        if(outer_sum2 != 0)
                        {
                            int32_t sph_idx = l - m;
                            //std::cout << "non-zero from Y_{m} - Y_{-m} (" << outer_sum1 * maxl_norm << ") for l = (" << lx << ", " << ly << ", " << lz << "), " << " m = " << m << std::endl;
                            c2s_trps.push_back({sph_idx, cart_idx, outer_sum2 * maxl_norm});
                        }

                    }
                }
            }
            c2s[l] = Eigen::SparseMatrix<double>(nso[l], nco[l]);
            c2s[l].setFromTriplets(c2s_trps.begin(), c2s_trps.end());
            //std::cout << "l = " << l << '\n';
            //std::cout << "max-L normalized cartesian: \n";
            //std::cout << c2s[l] << '\n';
        }
        for(int32_t l1 = 0;l1 <= MAX_AM;l1++)
        {
            for(int32_t l2 = 0;l2 <= MAX_AM;l2++)
            {
                Eigen::MatrixXd dense_bra_transform = Eigen::MatrixXd::Zero(c2s[l1].rows() * c2s[l2].rows(), c2s[l1].cols() * c2s[l2].cols());
                for(uint32_t t1 = 0;t1 < c2s[l1].outerSize();t1++)
                {
                    uint32_t c = t1 * c2s[l2].cols();
                    for(Eigen::SparseMatrix<double>::InnerIterator it1(c2s[l1], t1);it1;++it1)
                    {
                        uint32_t r = it1.row() * c2s[l2].rows();
                        //std::cout << "r = " << r << ", c = " << c << ", val = " << it1.value() << '\n';
                        dense_bra_transform.block(r, c, c2s[l2].rows(), c2s[l2].cols()) = it1.value() * c2s[l2];
                    }
                }
                cc2ss[l1][l2] = dense_bra_transform.sparseView();
                if constexpr(DEBUG_PRINT)
                {
                    std::ostream* logger = get_logger();
                    *logger << " l1 = " << l1 << ", l2 = " << l2 << '\n';
                    std::ios_base::fmtflags f( logger->flags() );
                    *logger << std::scientific << std::setprecision(16);
                    for(int i = 0;i < cc2ss[l1][l2].rows();i++)
                    {
                        for(int j = 0;j < cc2ss[l1][l2].cols();j++)
                            *logger << std::setw(24) << cc2ss[l1][l2].coeffRef(i, j);
                        *logger << '\n';
                    }
                    *logger << '\n';
                    //*logger << cc2ss[l1][l2] << '\n';
                    logger->flags(f);
                }
            }
        }
        
    }

    quartet_transform::~quartet_transform()
    {
        if(transform_buffer1)std::free(transform_buffer1);
        if(transform_buffer2)std::free(transform_buffer2);
    }
    


    void quartet_transform::set_quartet_class(const am_container& am)
    {
        // std::cout << "Input am = " << am << '\n';

        // permutation is the inverse if applied in reverse (i.e. right-to-left)
        uint32_t inv_perm = am.get_perm();
        am_container tgt_am = am;
        tgt_am.undo_perm();
        // std::cout << "Target am = " << tgt_am << '\n';
       
        
        uint32_t nbra = this->only_permutation ? am.ncbra() : am.nsbra();
        uint32_t nket = this->only_permutation ? am.ncket() : am.nsket();
        uint32_t tgt_nbra = this->only_permutation ? tgt_am.ncbra() : tgt_am.nsbra();
        uint32_t tgt_nket = this->only_permutation ? tgt_am.ncket() : tgt_am.nsket();

        uint32_t na = this->only_permutation ? am.nca() : am.nsa();
        uint32_t nb = this->only_permutation ? am.ncb() : am.nsb();
        uint32_t nc = this->only_permutation ? am.ncc() : am.nsc();
        uint32_t nd = this->only_permutation ? am.ncd() : am.nsd();

        Eigen::VectorXi perm_ab_vec = Eigen::VectorXi(nbra);
        Eigen::VectorXi perm_cd_vec = Eigen::VectorXi(nket);
        uint32_t vec_idx = 0;
        for(uint32_t b = 0;b < nb;b++)
            for(uint32_t a = 0;a < na;a++,vec_idx++)
                perm_ab_vec[vec_idx] = inv_perm & 0x2 ? a * nb + b : b * na + a;
        
        vec_idx = 0;
        for(uint32_t d = 0;d < nd;d++)
            for(uint32_t c = 0;c < nc;c++,vec_idx++)
                perm_cd_vec[vec_idx] = inv_perm & 0x1 ? c * nd + d : d * nc + c;
        
        Eigen::PermutationMatrix<Eigen::Dynamic> perm_ab{perm_ab_vec}, perm_cd{perm_cd_vec};
        
        this->bra_transform = perm_ab * this->cc2ss[am.b()][am.a()];
        Eigen::SparseMatrix<double> tmp = perm_cd * this->cc2ss[am.d()][am.c()];
        this->ket_transform = tmp.transpose();

        //std::cout << "cc2ss[c][d] = \n" << this->cc2ss[am.c()][am.d()] << '\n';
        //std::cout << "perm_cd * cc2ss[c][d] = \n" << tmp << '\n';
        //std::cout << "(perm_cd * cc2ss[c][d])^T = \n" << tmp.transpose() << '\n';

        this->cart_size = am.cart_size();
        this->sph_size = am.cart_size();

        uint32_t transform_kind = 0;
        if(am.a() + am.b() > 0)transform_kind += this->BRA;
        if(am.c() + am.d() > 0)transform_kind += this->KET;
        if(inv_perm & 0x4)transform_kind += this->TRANSPOSE;

        this->input_bra_dim = am.ncbra();
        this->input_ket_dim = am.ncket();

        this->output_bra_dim = tgt_nbra;
        this->output_ket_dim = tgt_nket; 

        if(transform_kind & this->BRA)
        {
            this->tmp_bra_dim = nbra;
            this->tmp_ket_dim = am.ncket();
        }
        else if(transform_kind & this->KET)
        {
            this->tmp_bra_dim = am.ncbra();
            this->tmp_ket_dim = nket;
        }
        else
        {
            // (ss|ss) case
            transform_kind = 0;
            this->tmp_bra_dim = 1;
            this->tmp_ket_dim = 1;
        }
        // std::cout << "transform_kind = " << transform_kind << '\n';
        //std::cout << "unpermuted_bra_transform = " << this->cc2ss[am.b()][am.a()];
        //std::cout << "bra_transform = " << bra_transform << '\n';
        //std::cout << "perm_ab = " << Eigen::MatrixXi(perm_ab) << '\n';
        //std::cout << "ket_transform = " << ket_transform << '\n';
        //std::cout << "perm_cd = " << Eigen::MatrixXi(perm_cd) << '\n';

        this->apply_transformation = this->transform_funcs[transform_kind]; 
    }

    uint64_t quartet_transform::mem_usage()
    {
        uint64_t sz = 0;
        for(int32_t l1 = 0;l1 <= MAX_AM;l1++)
        {
            sz += c2s[l1].nonZeros() * (sizeof(double) + sizeof(int)) + c2s[l1].outerSize() * sizeof(int);
            for(int32_t l2 = 0;l2 <= MAX_AM;l2++)
            {
                sz += cc2ss[l1][l2].nonZeros() * (sizeof(double) + sizeof(int)) + cc2ss[l1][l2].outerSize() * sizeof(int);
            }
        }
        sz += 2 * (cc2ss[MAX_AM][MAX_AM].nonZeros() * (sizeof(double) + sizeof(int)) + cc2ss[MAX_AM][MAX_AM].outerSize() * sizeof(int));
        sz += 2 * ipow(nco[MAX_AM], 4) * sizeof(double);
        sz += sizeof(*this);
        return sz;
    }

}
