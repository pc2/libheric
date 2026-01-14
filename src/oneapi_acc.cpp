#include "oneapi_acc.h"

#include "constants.h"
#include "math_util.h"
#include "transform.h"
#include "util.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <new>
#include <thread>
#include <utility>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <dlfcn.h>

extern "C" {
#include <rys_roots.h>
}

#include <libint2.h>
#include <libint2.hpp>
#include <libint2/boys.h>

namespace libheric
{
    template<typename FLOAT, size_t VEC_SIZE>
    void print_vector(const char* v_name, const std::array<FLOAT, VEC_SIZE>& v)
    {
        printf("%s = [", v_name);
        for(uint32_t i = 0;i < v.size()-1;i++)
        {
            printf("%f, ", v[i]);
        }
        printf("%f]\n", v[v.size()-1]);
    }
    template<typename FLOAT>
    std::vector<FLOAT> rys_quadrature(const am_container& am, quartet_input<double> input_quartet, rtwt_container<double> rtwt)
    {
        static_assert(std::is_floating_point<FLOAT>::value, "[ERROR] rys_quadrature can only be instantiated with floating point types");
        uint32_t len_a = am.a() + 1;
        uint32_t len_b = am.b() + 1;
        uint32_t len_c = am.c() + 1;
        uint32_t len_d = am.d() + 1;

        uint32_t am_n = am.a() + am.b();
        uint32_t am_m = am.c() + am.d();
        uint32_t len_m = am_m + 1;
        uint32_t len_n = am_n + 1;

        uint32_t num_rys = (am_n + am_m) / 2 + 1;

        constexpr uint32_t EXP = 3;
        constexpr uint32_t NUM_COORDINATES = 3;


        constexpr uint32_t RYS_ORDER_MAX = 8;
        if(num_rys > RYS_ORDER_MAX)
        {
            std::cout << "[ERROR] Maximum number of Rys roots support is " << RYS_ORDER_MAX << ", but " << num_rys << " are required for quarted (" << am.a() << am.b() << "|" << am.c() << am.d() << ")\n";
            exit(1);
        }


        double normalization_factor = 1.0f;
        for(uint32_t i = 0;i < 4;i++)
        {
            double term1 = std::sqrt(std::sqrt(std::pow(2.0 * input_quartet.raw[i * 4 + EXP]/std::numbers::pi, 3.0)) * std::pow(4.0 * input_quartet.raw[i * 4 + EXP], am.raw[i]));
            double term2 = std::sqrt(double(factorial(am.raw[i]))/double(factorial(2*am.raw[i])) * std::pow(2.0, am.raw[i]));
            //std::cout << "term1 = " << term1 << '\n';
            //std::cout << "term2 = " << term2 << '\n';
            normalization_factor *= term1 * term2;
        }

        double zeta = input_quartet.raw[0 * 4 + EXP] + input_quartet.raw[1 * 4 + EXP];
        double eta  = input_quartet.raw[2 * 4 + EXP] + input_quartet.raw[3 * 4 + EXP];
        double rho  = zeta * eta / (zeta + eta);

        double P[3], Q[3], AB[3], CD[3], PA[3], QC[3], PQ[3];
        FLOAT kappa = 0.0f;
        for(uint32_t i = 0;i < 3;i++)
        {
            P[i] = (input_quartet.raw[0 * 4 + EXP] * input_quartet.raw[0 * 4 + i] + input_quartet.raw[1 * 4 + EXP] * input_quartet.raw[1 * 4 + i]) / zeta;
            Q[i] = (input_quartet.raw[2 * 4 + EXP] * input_quartet.raw[2 * 4 + i] + input_quartet.raw[3 * 4 + EXP] * input_quartet.raw[3 * 4 + i]) / eta;
            AB[i] = (input_quartet.raw[0 * 4 + i] - input_quartet.raw[1 * 4 + i]);
            CD[i] = (input_quartet.raw[2 * 4 + i] - input_quartet.raw[3 * 4 + i]);
            PA[i] = (P[i] - input_quartet.raw[0 * 4 + i]);
            QC[i] = (Q[i] - input_quartet.raw[2 * 4 + i]);
            PQ[i] = (P[i] - Q[i]);
            kappa += (input_quartet.raw[0 * 4 + EXP] * input_quartet.raw[1 * 4 + EXP])/zeta * AB[i] * AB[i]
                  +  (input_quartet.raw[2 * 4 + EXP] * input_quartet.raw[3 * 4 + EXP])/eta  * CD[i] * CD[i];
        }
        // printf("AB = [%+.8e, %+.8e, %+.8e, %+.8e]\n", AB[0], AB[1], AB[2], 0.0f);
        // printf("CD = [%+.8e, %+.8e, %+.8e, %+.8e]\n", CD[0], CD[1], CD[2], 0.0f);
        // printf("PA = [%+.8e, %+.8e, %+.8e, %+.8e]\n", PA[0], PA[1], PA[2], 0.0f);
        // printf("QC = [%+.8e, %+.8e, %+.8e, %+.8e]\n", QC[0], QC[1], QC[2], 0.0f);
        // printf("PQ = [%+.8e, %+.8e, %+.8e, %+.8e]\n", PQ[0], PQ[1], PQ[2], 0.0f);
        
        // printf("zeta = %f\n", zeta);
        // printf("eta  = %f\n", eta);
        // printf("chi  = %f\n", 1/(zeta + eta));

        FLOAT prefactor = 2 * normalization_factor * std::pow(std::numbers::pi, 5.0/2.0) * std::sqrt(rho) * std::exp(-kappa) / std::pow(zeta * eta, 3.0/2.0);
        //std::cout << "norm = " << normalization_factor << "\n";
        //std::cout << "kappa = " << std::exp(-kappa) << '\n';
        //std::cout << "rho = " << std::sqrt(rho) / std::pow(zeta * eta, 3.0/2.0) << '\n';
        //std::cout << "kc = " << prefactor << "\n";
        // std::cout << "prefactor = " << prefactor << '\n';
        std::array<FLOAT, RYS_ORDER_MAX> prep_roots{};
        std::array<FLOAT, RYS_ORDER_MAX> prep_weights{};

        std::array<FLOAT, RYS_ORDER_MAX> B0{}, B1{}, B2{};
        std::array<FLOAT, RYS_ORDER_MAX> C1[NUM_COORDINATES] = {{},{},{}}, C2[NUM_COORDINATES] = {{},{},{}};
        FLOAT I_abcd[len_d][len_m][len_b][len_n][num_rys][NUM_COORDINATES];

        for(uint32_t u = 0;u < num_rys;u++)
        {
            prep_roots[u] = FLOAT(rtwt.roots()[u] / (1.0 + rtwt.roots()[u])); // Transform u^2 -> t^2 (Somehow the rys_root/rys_weight routines in libcint/pyquante calculate u^2 s.t. u^2 = t^2/(1-t^2) and not u^2 = \rho t^2/(1-t^2), might explain the difference between the roots in libcchem and libcint?)
            prep_weights[u] = FLOAT(std::cbrt(rtwt.weights()[u] * prefactor));


            B0[u] = (FLOAT(1.0)/FLOAT(2.0)) * FLOAT(1.0)/(FLOAT(zeta) + FLOAT(eta)) * prep_roots[u];
            B1[u] = (FLOAT(1.0)/(FLOAT(2.0) * FLOAT(zeta))) * (FLOAT(1.0) - FLOAT( eta)/(FLOAT(zeta) + FLOAT(eta)) * prep_roots[u]);
            B2[u] = (FLOAT(1.0)/(FLOAT(2.0) * FLOAT( eta))) * (FLOAT(1.0) - FLOAT(zeta)/(FLOAT(zeta) + FLOAT(eta)) * prep_roots[u]);

            for(uint32_t i = 0;i < NUM_COORDINATES;i++)
            {
                C1[i][u] = FLOAT(PA[i]) - (FLOAT( eta) * FLOAT(PQ[i]))/(FLOAT(eta) + FLOAT(zeta)) * prep_roots[u];
                C2[i][u] = FLOAT(QC[i]) + (FLOAT(zeta) * FLOAT(PQ[i]))/(FLOAT(eta) + FLOAT(zeta)) * prep_roots[u];
                I_abcd[0][0][0][0][u][i] = prep_weights[u];
                // I_abcd[0][0][0][1][u][i] = C1[i][u] * I_abcd[0][0][0][0][u][i];
                // I_abcd[0][1][0][0][u][i] = C2[i][u] * I_abcd[0][0][0][0][u][i];
                // I_abcd[0][1][0][1][u][i] = B0[u]    * I_abcd[0][0][0][0][u][i] 
                //                          + C1[i][u] * I_abcd[0][0][0][1][u][i];
            }
        }
        std::cout << input_quartet << '\n'; 
        std::cout << rtwt << '\n';
            
        // print_vector("B0", B0);
        // print_vector("B1", B1);
        // print_vector("B2", B2);

        // print_vector("C1_x", C1[0]);
        // print_vector("C1_y", C1[1]);
        // print_vector("C1_z", C1[2]);
        // print_vector("C2_x", C2[0]);
        // print_vector("C2_y", C2[1]);
        // print_vector("C2_z", C2[2]);



        for(uint32_t n = 0;n < len_n-1;n++)
        {
            for(uint32_t u = 0;u < num_rys;u++)
            {
                for(uint32_t i = 0;i < 3;i++)
                {
                    uint32_t n0 = n == 0 ? 0 : n - 1;
                    I_abcd[0][0][0][n+1][u][i] = n * B1[u] * I_abcd[0][0][0][n0][u][i] 
                                               +  C1[i][u] * I_abcd[0][0][0][n ][u][i];
                }
            }
        }

        for(uint32_t m = 0;m < len_m-1;m++)
        {
            for(uint32_t u = 0;u < num_rys;u++)
            {
                for(uint32_t i = 0;i < 3;i++)
                {
                    uint32_t m0 = m == 0 ? 0 : m - 1;
                    I_abcd[0][m+1][0][0][u][i] = m * B2[u] * I_abcd[0][m0][0][0][u][i] 
                                               +  C2[i][u] * I_abcd[0][m ][0][0][u][i];
                }
            }
        }

        for(uint32_t m = 1;m < len_m;m++)
        {
            for(uint32_t n = 0;n < len_n-1;n++)
            {
                for(uint32_t u = 0;u < num_rys;u++)
                {
                    for(uint32_t i = 0;i < 3;i++)
                    {
                        uint32_t n0 = n == 0 ? 0 : n - 1;

                        I_abcd[0][m][0][n+1][u][i] = m * B0[u] * I_abcd[0][m-1][0][n ][u][i] 
                                                   + n * B1[u] * I_abcd[0][m  ][0][n0][u][i]
                                                   +  C1[i][u] * I_abcd[0][m  ][0][n ][u][i];
                    }
                }
            }
        }


        //for(uint32_t i = 0;i < NUM_COORDINATES;i++)
        //{
        //    // printf("coord = %u\n", i);
        //    for(uint32_t m = 0;m < len_m;m++)
        //    {
        //        for(uint32_t n = 0;n < len_n;n++)
        //        {
        //            for(uint32_t u = 0;u < num_rys;u++)
        //            {
        //                uint32_t idx = idx = ((m * len_n + n) * num_rys + u) * NUM_COORDINATES + i;
        //                vrr_ref[idx] = I_abcd[0][m][0][n][u][i];
        //                // printf("vrr[m = %u][n = %u][u = %u] = %f\n", m, n, u, I_abcd[0][m  ][0][n  ][u][i]);
        //            }
        //            // printf("\n");
        //        }
        //        // printf("----------------------------------------------------------------\n");
        //    }
        //}

        for(uint32_t m = 0;m < len_m;m++)
        {
            for(uint32_t b = 0;b < len_b-1;b++)
            {
                for(uint32_t a = 0;a < len_n-b-1;a++)
                {
                    for(uint32_t u = 0;u < num_rys;u++)
                    {
                        for(uint32_t i = 0;i < 3;i++)
                        {
                            I_abcd[0][m][b+1][a][u][i] = I_abcd[0][m][b][a+1][u][i]
                                                       + AB[i] * I_abcd[0][m][b][a][u][i];
                        }
                    }
                }
            }
        }

        for(uint32_t d = 0;d < len_d-1;d++)
        {
            for(uint32_t c = 0;c < len_m-d-1;c++)
            {
                for(uint32_t b = 0;b < len_b;b++)
                {
                    for(uint32_t a = 0;a < len_a;a++)
                    {
                        for(uint32_t u = 0;u < num_rys;u++)
                        {
                            for(uint32_t i = 0;i < 3;i++)
                            {
                                I_abcd[d+1][c][b][a][u][i] = I_abcd[d][c+1][b][a][u][i]
                                                           + CD[i] * I_abcd[d][c][b][a][u][i];
                            }
                        }
                    }
                }
            }
        }

        //for(uint32_t d = 0;d < len_d;d++)
        //{
        //    for(uint32_t c = 0;c < len_c;c++)
        //    {
        //        for(uint32_t b = 0;b < len_b;b++)
        //        {
        //            for(uint32_t a = 0;a < len_a;a++)
        //            {
        //                for(uint32_t u = 0;u < num_rys;u++)
        //                {
        //                    for(uint32_t i = 0;i < NUM_COORDINATES;i++)
        //                    {
        //                        uint32_t idx = ((((d * len_c + c) * len_b + b) * len_a + a) * num_rys + u) * NUM_COORDINATES + i;
        //                        hrr_ref[idx] = I_abcd[d][c][b][a][u][i];
        //                        //printf("hrr[%u][%u][%u][%u][%u][%u] = %f\n", d,c,b,a,u,i,I_abcd[d][c][b][a][u][i]);
        //                    }
        //                }
        //            }
        //        }
        //    }
        //}


        std::vector<FLOAT> eris;

        for(uint32_t d1 = 0;d1 < len_d;d1++)
        {
        for(uint32_t d2 = 0;d2 <= d1;d2++)
        {
            uint32_t dx = am.d() - d1;
            uint32_t dy = d1 - d2;
            uint32_t dz = d2;

            for(uint32_t c1 = 0;c1 < len_c;c1++)
            {
            for(uint32_t c2 = 0;c2 <= c1;c2++)
            {
                uint32_t cx = am.c() - c1;
                uint32_t cy = c1 - c2;
                uint32_t cz = c2;

                for(uint32_t b1 = 0;b1 < len_b;b1++)
                {
                for(uint32_t b2 = 0;b2 <= b1;b2++)
                {
                    uint32_t bx = am.b() - b1;
                    uint32_t by = b1 - b2;
                    uint32_t bz = b2;

                    for(uint32_t a1 = 0;a1 < len_a;a1++)
                    {
                    for(uint32_t a2 = 0;a2 <= a1;a2++)
                    {
                        uint32_t ax = am.a() - a1;
                        uint32_t ay = a1 - a2;
                        uint32_t az = a2;

                        FLOAT eri = 0.0f;
                        // std::cout << "ax = " << ax << ", bx = " << bx << ", cx = " << cx << ", dx = " << dx << '\n';
                        // std::cout << "ay = " << ay << ", by = " << by << ", cy = " << cy << ", dy = " << dy << '\n';
                        // std::cout << "az = " << az << ", bz = " << bz << ", cz = " << cz << ", dz = " << dz << '\n';
                        for(uint32_t u = 0;u < num_rys;u++)
                        {
                            // std::cout << "\t u = " << u 
                            //           << ": I_abcd_x = " 
                            //           << I_abcd[dx][cx][bx][ax][u][0] 
                            //           << ", I_abcd_y = " 
                            //           << I_abcd[dy][cy][by][ay][u][1] 
                            //           << ", I_abcd_z = " 
                            //           << I_abcd[dz][cz][bz][az][u][1] 
                            //           << '\n';

                            eri += I_abcd[dx][cx][bx][ax][u][0]
                                 * I_abcd[dy][cy][by][ay][u][1]
                                 * I_abcd[dz][cz][bz][az][u][2];
                        }
                        eris.push_back(eri);

                    }
                    }

                }
                }

            }
            }
        }
        }
        return eris;
    }

    std::array<std::vector<double>, 3> libint_oshgp(uint32_t l_max, const am_container& am, quartet_input<double> q, int libint_perm)
    {
        Libint_eri_t lint_engine;
        libint2_init_eri(&lint_engine, l_max, 0);

        libint2::Engine eri_engine(libint2::Operator::coulomb, 1, l_max);
        

        libint2::Shell s_a = libint2::Shell({{q.alpha()}}, {{am.a(), true, {{1.0}}}}, {{q.A()(0), q.A()(1), q.A()(2)}});
        libint2::Shell s_b = libint2::Shell({{q.beta() }}, {{am.b(), true, {{1.0}}}}, {{q.B()(0), q.B()(1), q.B()(2)}});
        libint2::Shell s_c = libint2::Shell({{q.gamma()}}, {{am.c(), true, {{1.0}}}}, {{q.C()(0), q.C()(1), q.C()(2)}});
        libint2::Shell s_d = libint2::Shell({{q.delta()}}, {{am.d(), true, {{1.0}}}}, {{q.D()(0), q.D()(1), q.D()(2)}});
        
        std::cout << s_a << s_b << s_c << s_d << '\n';
        eri_engine.compute(s_a, s_b, s_c, s_d);
        
        std::vector<double> l2_res = permute_sph_quartet_scalar_double[7](eri_engine.results()[0], am);

        am_container perm_am = am;
        perm_am.apply_permutation(libint_perm);

        double tgt_alpha = q.alpha();
        double tgt_beta  = q.beta();
        double tgt_gamma = q.gamma();
        double tgt_delta = q.delta();

        q.apply_permutation(libint_perm);
        
        std::cout << "Libint input:\n" << q << '\n';
        std::cout << perm_am << '\n';

        auto A = q.A();
        auto B = q.B();
        auto C = q.C();
        auto D = q.D();

        double alpha = q.alpha();
        double beta  = q.beta(); 
        double gamma = q.gamma();
        double delta = q.delta();

        double zeta = alpha + beta;
        double eta  = gamma + delta;
        double chi  = (zeta * eta) / (zeta + eta);

        double P[3], Q[3], W[3], AB[3], CD[3], PA[3], QC[3], PQ[3], WP[3], WQ[3];
        double AB2 = 0, CD2 = 0, PQ2 = 0;

        for(int xyz = 0;xyz < 3;xyz++)
        {
            P[xyz] = (alpha * A[xyz] + beta  * B[xyz]) / zeta; 
            Q[xyz] = (gamma * C[xyz] + delta * D[xyz]) / eta; 
            W[xyz] = (zeta  * P[xyz] + eta   * Q[xyz]) / (zeta + eta); 
            
            AB[xyz] = A[xyz] - B[xyz];
            CD[xyz] = C[xyz] - D[xyz];
            PA[xyz] = P[xyz] - A[xyz];
            QC[xyz] = Q[xyz] - C[xyz];
            PQ[xyz] = P[xyz] - Q[xyz];
            WP[xyz] = W[xyz] - P[xyz];
            WQ[xyz] = W[xyz] - Q[xyz];

            AB2 += AB[xyz] * AB[xyz];
            CD2 += CD[xyz] * CD[xyz];
            PQ2 += PQ[xyz] * PQ[xyz];
        }
        
        lint_engine.AB_x[0] = AB[0]; 
        lint_engine.AB_y[0] = AB[1]; 
        lint_engine.AB_z[0] = AB[2]; 
        
        lint_engine.CD_x[0] = CD[0]; 
        lint_engine.CD_y[0] = CD[1]; 
        lint_engine.CD_z[0] = CD[2]; 

        lint_engine.PA_x[0] = PA[0]; 
        lint_engine.PA_y[0] = PA[1]; 
        lint_engine.PA_z[0] = PA[2]; 
                                          
        lint_engine.QC_x[0] = QC[0]; 
        lint_engine.QC_y[0] = QC[1]; 
        lint_engine.QC_z[0] = QC[2]; 

        lint_engine.WP_x[0] = WP[0]; 
        lint_engine.WP_y[0] = WP[1]; 
        lint_engine.WP_z[0] = WP[2]; 

        lint_engine.WQ_x[0] = WQ[0]; 
        lint_engine.WQ_y[0] = WQ[1]; 
        lint_engine.WQ_z[0] = WQ[2]; 
        
        lint_engine.oo2z [0] = 1.0 / (2.0 * zeta); 
        lint_engine.oo2e [0] = 1.0 / (2.0 *  eta); 
        lint_engine.oo2ze[0] = 1.0 / (2.0 * (zeta + eta));
        lint_engine.roz  [0] = chi / zeta; 
        lint_engine.roe  [0] = chi /  eta;



        lint_engine.contrdepth = 1;
        
        double kappa = std::exp(-((alpha * beta * AB2)/zeta + (gamma * delta * CD2)/eta));
        double prefactor = 2.0 * std::pow(std::numbers::pi, 2.5) * kappa * std::sqrt(chi / (zeta * eta)) / (zeta * eta);
        
        uint32_t total_am = am.a() + am.b() + am.c() + am.d();
        libint2::FmEval_Chebyshev7 fm_eval(total_am);
        double X = PQ2 * chi;
        double F[total_am + 1];
        fm_eval.eval(F, X, total_am);
        for(uint32_t i = 0;i <= total_am;i++)F[i] *= prefactor;
        memcpy(&lint_engine.LIBINT_T_SS_EREP_SS(0)[0], F, (total_am + 1) * sizeof(double));
        //std::cout << "Evaluated boys functions" << std::endl;
        //std::cout << "Permuted am_a = " << am.a() << ", am_b = " << am.b() << ", am_c = " << am.c() << ", am_d = " << am.d() << std::endl;
        //std::cout << "eri_fct_ptr = " << libint2_build_eri[am.a()][am.b()][am.c()][am.d()] << std::endl;
        libint2_build_eri[perm_am.a()][perm_am.b()][perm_am.c()][perm_am.d()](&lint_engine);
        double* results = lint_engine.targets[0];
        //std::cout << "Compute ERIs with libint" << std::endl;

        std::vector<double> maxl_eris;
        std::vector<double> unit_eris;

        uint32_t eri_ctr = 0;

        double maxl_norm = 1.0;
        double exps[4] = {alpha, beta, gamma, delta}; 
        for(uint32_t i = 0;i < 4;i++)
        {
            double term1 = std::sqrt(std::sqrt(std::pow(2.0 * exps[i]/std::numbers::pi, 3.0)) * std::pow(4.0 * exps[i], perm_am.raw[i]));
            double term2 = std::sqrt(double(factorial(perm_am.raw[i]))/double(factorial(2*perm_am.raw[i])) * std::pow(2.0, perm_am.raw[i]));
            maxl_norm *= term1 * term2;
        }
        std::cout << '\n';

        auto unit_norm_lambda = [](int32_t l, int32_t lx, int32_t ly, int32_t lz, double exp) {
            double term1 = std::sqrt(std::sqrt(std::pow(2.0 * exp/std::numbers::pi, 3.0)) * std::pow(4.0 * exp, l));
            double term2 = std::sqrt(dfac[std::max(2 * lx - 1, 0)] * dfac[std::max(2 * ly - 1, 0)] * dfac[std::max(2 * lz - 1, 0)]);
            return term1 / term2;
        };
        
        //libint integrals are stored "Row major", i.e. (a0b0|c0d0), (a0b0|c0d1), (a0b0|c0d2), ...
        //while CP2K expects integral stored as "Column major", i.e. (a0b0|c0d0), (a1b0|c0d0), ...
        int output_perm = libint_perm ^ 0x7;
        std::vector<double> permuted_results = permute_cart_quartet_scalar_double[output_perm](results, am);

        for(int32_t d1 = 0;d1 <= am.d();d1++)
        {
        for(int32_t d2 = 0;d2 <= d1;d2++)
        {
            int32_t dx = am.d() - d1;
            int32_t dy = d1 - d2;
            int32_t dz = d2;
            double unit_norm_d = unit_norm_lambda(am.d(), dx, dy, dz, tgt_delta);
            for(int32_t c1 = 0;c1 <= am.c();c1++)
            {
            for(int32_t c2 = 0;c2 <= c1;c2++)
            {
                int32_t cx = am.c() - c1;
                int32_t cy = c1 - c2;
                int32_t cz = c2;
                double unit_norm_cd = unit_norm_d * unit_norm_lambda(am.c(), cx, cy, cz, tgt_gamma);

                for(int32_t b1 = 0;b1 <= am.b();b1++)
                {
                for(int32_t b2 = 0;b2 <= b1;b2++)
                {
                    int32_t bx = am.b() - b1;
                    int32_t by = b1 - b2;
                    int32_t bz = b2;
                    double unit_norm_bcd = unit_norm_cd * unit_norm_lambda(am.b(), bx, by, bz, tgt_beta);
                    for(int32_t a1 = 0;a1 <= am.a();a1++)
                    {
                    for(int32_t a2 = 0;a2 <= a1;a2++)
                    {
                        int32_t ax = am.a() - a1;
                        int32_t ay = a1 - a2;
                        int32_t az = a2;
                        double unit_norm_abcd = unit_norm_bcd * unit_norm_lambda(am.a(), ax, ay, az, tgt_alpha);

                        maxl_eris.push_back(permuted_results[eri_ctr] * maxl_norm);
                        unit_eris.push_back(permuted_results[eri_ctr] * unit_norm_abcd);

                        eri_ctr++;
                    }
                    }
                }
                }
            }
            }
        }
        }
        libint2_cleanup_eri(&lint_engine);
        return {{maxl_eris, unit_eris, l2_res}}; 
         
    }

    std::vector<acc_device> discover_oneapi_accelerators(const acc_kind& acc_query) 
    {
        std::vector<acc_device> found_devices;

        sycl::platform platform;
        try {
            platform = sycl::platform(sycl::ext::intel::fpga_selector_v);
        } catch (const sycl::exception& e) {
            std::cout << "[ERROR] Failed to query SYCL platform: " << e.what() << std::endl;
            return {};
        }
        std::string platform_name = platform.get_info<sycl::info::platform::name>();
        if(platform_name.find("Intel(R) FPGA SDK") == std::string::npos) return {};

        std::vector<sycl::device> devices;
        try {
            devices = platform.get_devices();
        } catch (const sycl::exception& e) {
            std::cout << "[ERROR] Failed to query SYCL devices from platform: " << e.what() << std::endl;
            return {};
        }

        for(size_t i = 0;i < devices.size(); i++)
        {
            std::string dev_name = devices[i].get_info<sycl::info::device::name>();
            std::string dev_name_upper = dev_name;
            std::for_each(dev_name_upper.begin(), dev_name_upper.end(), [](char& c){c = std::toupper(c);});
            
            if(dev_name_upper.find(acc_query.name) != std::string::npos)
            {
                std::string fake_dev_bdf = "0000:00:0" + std::to_string(i) + ".0";
                found_devices.push_back({dev_name, fake_dev_bdf, "", 1, acc_query});
            }
        }
        return found_devices;
    }

    oneapi_acc_context::oneapi_acc_context(const acc_device& acc_dev,
                                           const simulation_box& sim_box)
       : acc_context(acc_dev, sim_box), 
         lib_handle(nullptr), kernel(nullptr),
		 batch_size(0), cbits(0), n_rtwt(0), n512_per_quartet(0), 
		  host_eri_d{nullptr, nullptr}, host_eri_c{nullptr, nullptr}, host_eps{nullptr, nullptr}, 
		  dev_abcd{nullptr, nullptr}, dev_rtwt{nullptr, nullptr}, dev_eri_c{nullptr, nullptr}, dev_eps{nullptr, nullptr},
          overwrite_quartet_batch{1}, start_consuming{0}
    {
        // PC2 specific solution might/should be changed in the future!
        if(acc_dev.bdf != "0000:00:00.0" && acc_dev.bdf != "0000:00:01.0")
        {
            std::cout << "[ERROR] Unexpected device BDF for Intel FPGA: " << acc_dev.bdf << std::endl;
            return;
        }

        std::optional<sycl::platform> platform;
        try {
            platform = sycl::platform(sycl::ext::intel::fpga_selector_v);
        } catch (const sycl::exception& e) {
            std::cout << "[ERROR] Failed to query SYCL platform: " << e.what() << std::endl;
            return;
        }

        std::string platform_name = platform->get_info<sycl::info::platform::name>();
        if(platform_name.find("Intel(R) FPGA SDK") == std::string::npos)
        {
            std::cout << "[ERROR] expected to find Intel(R) FPGA SDK platform but instead got " << platform_name << std::endl;
            return;
        }
        
        std::vector<sycl::device> devices;
        try {
            devices = platform->get_devices();
        } catch (const sycl::exception& e) {
            std::cout << "[ERROR] Failed to query SYCL devices from platform: " << e.what() << std::endl;
            return;
        }
        
        this->dev = devices.back();
        std::string dev_name = this->dev->get_info<sycl::info::device::name>();
        if(dev_name != acc_dev.name)
        {
            std::cout << "[ERROR] allocated device " << dev_name << " is not the requested device " << acc_dev.name << std::endl;
            return;
        }

		if (!this->dev->get_info<sycl::info::device::usm_device_allocations>())
        {
            std::cout << "[ERROR] allocated device " << dev_name << " does not support explicit USM device allocations" << std::endl;
			return;
		}

        this->ctx = sycl::context(this->dev.value());
        this->q = sycl::queue(this->ctx.value(), this->dev.value(), sycl::property::queue::enable_profiling{});
        //libint2_static_init();
        libint2::initialize();
    }

    oneapi_acc_context::~oneapi_acc_context()
    {
		if(this->lib_handle != nullptr)
		{
            if(dlclose(this->lib_handle) != 0)
                std::cout << "[WARNING] Failed to close open shared object handle!" << std::endl;
		}
        for(int db = 0;db < 2;db++)
        {
            if(host_eri_d[db] != nullptr)delete[] host_eri_d[db];
            if(host_eri_c[db] != nullptr)delete[] host_eri_c[db];
            if(host_eps[db]   != nullptr)delete[] host_eps[db];
            
            if(this->q)
            {
                if(dev_abcd[db]  != nullptr)sycl::free(dev_abcd[db] , this->q.value());
                if(dev_rtwt[db]  != nullptr)sycl::free(dev_rtwt[db] , this->q.value());
                if(dev_eri_c[db] != nullptr)sycl::free(dev_eri_c[db], this->q.value());
                if(dev_eps[db]   != nullptr)sycl::free(dev_eps[db]  , this->q.value());
            }
        }
        //libint2_static_cleanup();
        libint2::finalize();
    }

    void oneapi_acc_context::set_quartet(int la, int lb, int lc, int ld)
    {
        this->new_quartet_class_for_consume = true;
        this->acc_context::set_quartet(la, lb, lc, ld);
        this->norm_prefactor = 1.0 / std::sqrt(dfac[std::max(2 * am.a() - 1, 0)] // Avoid access to dfac[-1] for l=0 
                                             * dfac[std::max(2 * am.b() - 1, 0)]
                                             * dfac[std::max(2 * am.c() - 1, 0)]
                                             * dfac[std::max(2 * am.d() - 1, 0)]);
        if(this->is_functional)
        {
            this->is_functional = false;
            std::string base_dir = "/pc2/groups/pc2-mitarbeiter/xinwu/results-hpc-prf-erifpga/from-pc2data/Xin/For_Robert/1st_round/";


            std::stringstream ss_abcd_int;
            int abcd_id;
            if(am.is_fpga_canonical())
            {
                abcd_id = this->am.d() + (MAX_AM + 1) * ( this->am.c() + (MAX_AM + 1) * ( this->am.b() + (MAX_AM + 1) * this->am.a())); 
                ss_abcd_int << this->am.a() << this->am.b() << this->am.c() << this->am.d();
            }
            else
            {
                abcd_id = this->am.b() + (MAX_AM + 1) * ( this->am.a() + (MAX_AM + 1) * ( this->am.d() + (MAX_AM + 1) * this->am.d())); 
                ss_abcd_int << this->am.c() << this->am.d() << this->am.a() << this->am.b();
            }
            
            std::stringstream ss_kernel_dir;

            ss_kernel_dir << "t_" << std::setw(3) << std::setfill('0') << abcd_id << "_" << ss_abcd_int.str();
            std::filesystem::path kernel_dir_path(base_dir);
            kernel_dir_path /= ss_kernel_dir.str();
            if(!std::filesystem::exists(kernel_dir_path) || !std::filesystem::is_directory(kernel_dir_path))
            {
                std::cout << "[ERROR] kernel_dir_path = " << kernel_dir_path << " is not a directory!" << std::endl;
                return;
            }

            // Get dir name with random seed
            auto dir_iter = std::filesystem::directory_iterator(kernel_dir_path);
            auto num_entries = std::distance(std::filesystem::begin(dir_iter), std::filesystem::end(dir_iter));
            if(num_entries != 1)
            {
                std::cout << "[ERROR] kernel_dir_path = " << kernel_dir_path << " contains != 1 entries!" << std::endl;
                return;
            }
            dir_iter = std::filesystem::directory_iterator(kernel_dir_path);
            kernel_dir_path = dir_iter->path();
            
            // Get .so library path
            dir_iter = std::filesystem::directory_iterator(kernel_dir_path);
            num_entries = std::distance(std::filesystem::begin(dir_iter), std::filesystem::end(dir_iter));
            if(num_entries != 1)
            {
                std::cout << "[ERROR] kernel_dir_path = " << kernel_dir_path << " contains != 1 entries!" << std::endl;
                return;
            }
            dir_iter = std::filesystem::directory_iterator(kernel_dir_path);
            auto kernel_path = dir_iter->path();

            std::cout << "Discovered kernel: " << kernel_path << std::endl;
            
            if(!std::filesystem::is_regular_file(kernel_path))
            {
                std::cout << "[ERROR] kernel_path = " << kernel_path << " is not a file!" << std::endl;
                return;
            }

            if(kernel_path.extension() != ".so")
            {
                std::cout << "[ERROR] kernel_path = " << kernel_path << " is not a .so file!" << std::endl;
                return;
            }

            auto kernel_name = kernel_path.stem();

            std::vector<std::string> kernel_name_parts;
            std::stringstream ss_kernel_name(kernel_name);
            for(std::string part; std::getline(ss_kernel_name, part, '_');)
                kernel_name_parts.push_back(part);

            std::vector<std::string> expected_parts = {"Rys4Compute", "float", ss_abcd_int.str()};
            if(kernel_name_parts.size() != 6)
            {
                std::cout << "[ERROR] " << kernel_name 
                          << ".so at kernel_path = " << kernel_path 
                          << " has not the expected format: "
                          << expected_parts[0] << "_"
                          << expected_parts[1] << "_"
                          << expected_parts[2] << "_XXX_XXX_XXX.so" << std::endl;
                return;
            }

            std::array<uint32_t, 3> kernel_params;
            for(size_t i = 0;i < kernel_name_parts.size();i++)
            {
                if(i < 3)
                {
                    if(kernel_name_parts[i] != expected_parts[i])
                    {
                      
                        std::cout << "[ERROR] " << kernel_name 
                                  << ".so at kernel_path = " << kernel_path 
                                  << " has not the expected format: "
                                  << expected_parts[0] << "_"
                                  << expected_parts[1] << "_"
                                  << expected_parts[2] << "_XXX_XXX_XXX.so" << std::endl;
                        return;
                    }
                }
                else
                {
                    try
                    {
                        kernel_params[i - 3] = std::stoi(kernel_name_parts[i], nullptr, 10);
                    }
                    catch(std::invalid_argument& e)
                    {
                        std::cout << "[ERROR] Invalid argument while parsing integer param #" << i - 3 
                                  << " for " << kernel_name << ".so -> " << e.what() << std::endl;
                        return;
                    }
                    catch(std::out_of_range& e)
                    {
                        std::cout << "[ERROR] Out of range while parsing integer param #" << i - 3 
                                  << " for " << kernel_name << ".so -> " << e.what() << std::endl;
                        return;
                    }
                }
            }

            this->batch_size = kernel_params[0];
            this->cbits = kernel_params[2];
			uint32_t num_compressed_per_512 = 512 / this->cbits;
			this->n512_per_quartet = (this->am.sph_size() + num_compressed_per_512 - 1) / num_compressed_per_512;			
           
            for(int db = 0;db < 2;db++)
            { 
                this->input_quartets[db].clear();
                this->input_quartets[db].reserve(this->batch_size);
                this->input_rtwt[db].clear();
                this->input_rtwt[db].reserve(this->batch_size);
                this->quartet_coeffs[db].clear();
                this->quartet_coeffs[db].reserve(this->batch_size);
                this->ceri_offsets[db].clear();
                this->ceri_offsets[db].reserve(this->batch_size + 1);
            }
            this->ceri_offsets[0].push_back(0);

            std::cout << "Kernel_params: batch_size = " << this->batch_size << ", #bits for compression = " << this->cbits << std::endl;

            void* lib_handle = dlopen(kernel_path.c_str(), RTLD_NOW);
            if(lib_handle == nullptr)
            {
                std::cout << "[ERROR] Failed to open shared object: " << kernel_path << "!" << std::endl;
                return;
            }

            dlerror(); // Clear error state
            this->kernel = (Rys4Compute_kernel)dlsym(lib_handle, kernel_name.c_str());
            auto dlsym_err = dlerror();
            if(dlsym_err != nullptr)
            {
                std::cout << "[ERROR] while establishing symbolic link to " << kernel_name.c_str() << ": " << dlsym_err << std::endl;
                dlclose(lib_handle);
                return;
            }
				
			n_rtwt = am.total() / 2 + 1;
            
            for(int db = 0;db < 2;db++)
            {
                if(host_eri_d[db] != nullptr) delete[] host_eri_d[db];
                if(host_eri_c[db] != nullptr) delete[] host_eri_c[db];
                if(host_eps[db]   != nullptr) delete[] host_eps[db];

                host_eri_d[db] = new(std::align_val_t(32)) float  [this->batch_size * this->am.sph_size()];
                host_eri_c[db] = new(std::align_val_t(32)) uint8_t[this->batch_size * this->n512_per_quartet * this->n8_per_n512];
                host_eps[db]   = new(std::align_val_t(32)) float  [this->batch_size];
                
                if(this->q)
                {
                    if(dev_abcd[db]  != nullptr) sycl::free(dev_abcd[db] , this->q.value());
                    if(dev_rtwt[db]  != nullptr) sycl::free(dev_rtwt[db] , this->q.value());
                    if(dev_eri_c[db] != nullptr) sycl::free(dev_eri_c[db], this->q.value());
                    if(dev_eps[db]   != nullptr) sycl::free(dev_eps[db]  , this->q.value());

                    dev_abcd[db]  = sycl::malloc_device<float>  (this->batch_size * num_abcd_elems, this->q.value());
                    dev_rtwt[db]  = sycl::malloc_device<float>  (this->batch_size * num_rtwt_elems, this->q.value());
                    dev_eri_c[db] = sycl::malloc_device<uint8_t>(this->batch_size * this->n512_per_quartet * this->n8_per_n512, this->q.value());
                    dev_eps[db]   = sycl::malloc_device<float>  (this->batch_size, this->q.value());
                }
                else
                {
                    std::cout << "[ERROR] queue not initialized in oneapi_acc_context::set_quartet!" << std::endl;
                    return; 
                }
            }
			
            gen_db = 0;
            consume_db = 1;
            ceri_ctr = 0;
            this->new_batch = false;
            this->new_quartet_class_for_gen = true;
            comp_events.clear();
            // Push back two fake completed computation events to trigger new computation events
            comp_events.push_back(sycl::event());
            comp_events.push_back(sycl::event());
            this->is_functional = true;
        }
    }
    
    void oneapi_acc_context::set_quartet_permutation(int perm)
    {
        acc_context::set_quartet_permutation(perm);
        if(perm < 0 || perm >= NUM_PERMS)
        {
            std::cout << "[ERROR] quartet permutation number not in 0 <= " << perm << " < " << NUM_PERMS << "\n";
            this->is_functional = false;
            return;
        }
        this->perm = perm;

        // most quartets that are canonical are also "FPGA canonical", however there are some outliers:
        // e.g. (gs|dp) is canonical but not "FPGA canonical"
        if(am.is_fpga_canonical())
            this->fpga_perm = this->perm;
        else
            this->fpga_perm = perm ^ 0x4;
    }

    void oneapi_acc_context::set_atom_id(int quartet_pos, int atom_id)
    {
        acc_context::set_atom_id(quartet_pos, atom_id);
        Eigen::Vector3d tmp_pos = this->sim_box.positions(Eigen::all, atom_id);      
        this->quartet_tmpl.pos(quartet_pos) = tmp_pos; 
    }
    
    void oneapi_acc_context::set_exponent_id(int quartet_pos, int exp_id)
    {
        acc_context::set_exponent_id(quartet_pos, exp_id);
        double exp_val =  sim_box.exponents(this->exp_off(quartet_pos));
        double t = two_over_pi * exp_val;

        this->quartet_tmpl.exp(quartet_pos) = exp_val;
        this->norm_coeffs[quartet_pos] = (quartet_pos == 0 ? 1.0 : this->norm_coeffs[quartet_pos - 1])
                                       * std::sqrt(std::sqrt(t * t * t) * powern(4.0 * exp_val, am.raw[quartet_pos]));  
    }
    
    void oneapi_acc_context::set_offset(int off_pos, int off_x, int off_y, int off_z)
    {
        acc_context::set_offset(off_pos, off_x, off_y, off_z);
        this->quartet_off.off(off_pos) = this->sim_box.cell_mat * this->offsets_d(Eigen::all, off_pos);  
    }

    void oneapi_acc_context::submit_peri()
    {
        quartet_input offset_quartet = this->quartet_tmpl;
        offset_quartet += this->quartet_off;

        std::cout << "Primitive #" << quartet_coeffs[gen_db].size() << '\n';
        std::cout << this->quartet_tmpl << '\n';
        std::cout << "Offset quartet:\n";
        std::cout << offset_quartet << '\n';

        auto cpu_rys_quartet = offset_quartet;
        auto fpga_quartet = offset_quartet;

        auto cpu_rys_am = am;

        cpu_rys_quartet.apply_permutation(this->perm);
        cpu_rys_am.apply_permutation(this->perm);
        fpga_quartet.apply_permutation(this->fpga_perm);
        
        auto A = offset_quartet.A();
        auto B = offset_quartet.B();
        auto C = offset_quartet.C();
        auto D = offset_quartet.D();

        double alpha = offset_quartet.alpha();
        double beta  = offset_quartet.beta();
        double gamma = offset_quartet.gamma();
        double delta = offset_quartet.delta();

        auto AB = A - B;
        double AB2 = AB.squaredNorm();
        auto CD = C - D;
        double CD2 = CD.squaredNorm();

        double zeta = alpha + beta;
        double eta  = gamma + delta;

        auto P = (alpha * A + beta * B) / zeta;
        auto Q = (gamma * C + delta * D) / eta;
        auto PQ = P - Q;
        double PQ2 = PQ.squaredNorm();
        

        //std::cout << "AB2 = " << AB2 << '\n';
        //std::cout << "CD2 = " << CD2 << '\n';
        double kc = two_pi_52 * norm_prefactor * norm_coeffs[3]
                  * std::exp(-(alpha * beta * AB2) / zeta - (gamma * delta * CD2) / eta)
                  / (zeta * eta * std::sqrt(zeta + eta));
        //std::cout << "norm_prefactor = " << norm_prefactor << '\n';
        //std::cout << "norm_coeffs = [" << norm_coeffs[0] << ", " << norm_coeffs[1] << ", " << norm_coeffs[2] << ", " << norm_coeffs[3] << "]\n";
        //std::cout << "my_norm = " << norm_prefactor * norm_coeffs[3] << '\n';
        //std::cout << "my_kappa = " << std::exp(-(alpha * beta * AB2) / zeta - (gamma * delta * CD2) / eta) << '\n';
        //std::cout << "my_rho = " << 1.0 / (zeta * eta * std::sqrt(zeta + eta)) << '\n';
        //std::cout << "my_kc = " << kc << '\n'; 
        double rhopq2 = zeta * eta / (zeta + eta) * PQ2; 
        CINTrys_roots(this->n_rtwt, rhopq2, this->rtwt_tmpl.roots(), this->rtwt_tmpl.weights());
        auto cpu_rys_rtwt = rtwt_tmpl;

        for(uint32_t u = 0;u < this->n_rtwt;u++)
        {
            double rt = rtwt_tmpl.roots()[u];
            double wt = rtwt_tmpl.weights()[u];
            rtwt_tmpl.roots()[u] = rt / (1.0 + rt);
            rtwt_tmpl.weights()[u] = std::cbrt(kc * wt);
        }
        
        this->input_quartets[gen_db].push_back(quartet_input<float>(fpga_quartet));
        this->input_rtwt[gen_db].push_back(rtwt_container<float>(rtwt_tmpl));
        this->quartet_coeffs[gen_db].push_back(this->coeff_prod(3));
        if(quartet_coeffs[gen_db].size() == batch_size)
        {
            gen_db = (gen_db + 1) % 2;
            new_batch = true;
        }
        
       
        int32_t lmax = am.max();
        auto libint_res = libint_oshgp(lmax, am, offset_quartet, this->libint_perm);
        libint_maxl_ref.push(cart2sph(cpu_rys_am, mnc2uns, libint_res[0]));
        libint_unit_ref.push(cart2sph(cpu_rys_am, unc2uns, libint_res[1]));
        libint2_ref.push(libint_res[2]);

        std::vector<double> rys_res = rys_quadrature<double>(cpu_rys_am, cpu_rys_quartet, cpu_rys_rtwt);
        rys_ref.push(cart2sph(cpu_rys_am, mnc2uns, rys_res));

    }
    
    void oneapi_acc_context::submit_ceri()
    {
        if(this->new_batch)
        {
            uint32_t prev_db = (gen_db + 1) % 2;
            this->ceri_offsets[prev_db].push_back(batch_size + this->input_quartets[gen_db].size());
            this->ceri_offsets[gen_db].clear();
            this->ceri_offsets[gen_db].push_back(0);

            size_t last_comp_event = this->comp_events.size() - 1;

            this->evt_abcd[prev_db] = this->q.value().memcpy(this->dev_abcd[prev_db], 
                                                             reinterpret_cast<float*>(this->input_quartets[prev_db].data()), 
                                                             this->batch_size * sizeof(quartet_input<float>), 
                                                             this->comp_events[last_comp_event - 1]);
            this->evt_rtwt[prev_db] = this->q.value().memcpy(this->dev_rtwt[prev_db], 
                                                             reinterpret_cast<float*>(this->input_rtwt[prev_db].data()), 
                                                             this->batch_size * sizeof(rtwt_container<float>), 
                                                             this->comp_events[last_comp_event - 1]);
            
            overwrite_quartet_batch.acquire();
            this->comp_events.push_back(sycl::event()); 
            kernel(this->q.value(),
                   this->dev_abcd[prev_db], this->dev_rtwt[prev_db],
                   this->dev_eri_c[prev_db], this->dev_eps[prev_db],
                   {this->evt_abcd[prev_db], this->evt_rtwt[prev_db], this->comp_events[last_comp_event - 1]},
                   this->comp_events.back());

            
            this->evt_eri_c[prev_db] = this->q.value().memcpy(this->host_eri_c[prev_db],
                                                              this->dev_eri_c[prev_db],
                                                              this->batch_size * this->n512_per_quartet * this->n8_per_n512,
                                                              this->comp_events.back());
            this->evt_eps[prev_db]   = this->q.value().memcpy(this->host_eps[prev_db],
                                                              this->dev_eps[prev_db],
                                                              this->batch_size * sizeof(float),
                                                              this->comp_events.back());
            if(this->new_quartet_class_for_gen)
            {
                this->start_consuming.release(); 
                this->new_quartet_class_for_gen = false;
            }
            this->new_batch = false;
        }

        uint32_t ceri_range_start = this->ceri_offsets[gen_db].back();
        this->ceri_offsets[gen_db].push_back(this->input_quartets[gen_db].size());
        uint32_t ceri_range_end = this->ceri_offsets[gen_db].back();
        //std::cout << "New contraction job for primitive range [" << ceri_range_start << ", " << ceri_range_end << std::endl;
        

        //std::cout << "Submit uncontracted quartets for contraction:\n";
        //for(uint32_t i = ceri_range_start;i < ceri_range_end;i++)
        //{
        //    std::cout << this->input_quartets[i] << this->input_rtwt[i];
        //    std::cout << "contraction coefficient = " << this->quartet_coeffs[i] << '\n';
        //    std::cout << "\n" << std::string(50, '-') << "\n\n";
        //}
    }

    void oneapi_acc_context::finish_quartet_class()
    {
        if(this->input_quartets[gen_db].size() > 0)
        {
            size_t last_comp_event = this->comp_events.size() - 1;

            //float* abcd_data = reinterpret_cast<float*>(this->input_quartets[gen_db].data());
            //float* rtwt_data = reinterpret_cast<float*>(this->input_rtwt[gen_db].data());
            //uint32_t num_elems = sizeof(quartet_input<float>)/sizeof(float);
            //for(uint32_t blk = 0;blk < this->input_quartets[gen_db].size();blk++)
            //{
            //    std::cout << "Primitive input #" << blk << ":\n";
            //    std::cout << "\tabcd = ";
            //    for(uint32_t elem = 0;elem < num_elems;elem++)std::cout << abcd_data[blk * num_elems + elem] << (elem < num_elems - 1 ? ", " : "\n");
            //    std::cout << "\trtwt = ";
            //    for(uint32_t elem = 0;elem < num_elems;elem++)std::cout << rtwt_data[blk * num_elems + elem] << (elem < num_elems - 1 ? ", " : "\n");
            //}


            this->evt_abcd[gen_db] = this->q.value().memcpy(this->dev_abcd[gen_db], 
                                                            reinterpret_cast<float*>(this->input_quartets[gen_db].data()), 
                                                            this->batch_size * sizeof(quartet_input<float>), 
                                                            this->comp_events[last_comp_event - 1]);
            this->evt_rtwt[gen_db] = this->q.value().memcpy(this->dev_rtwt[gen_db], 
                                                            reinterpret_cast<float*>(this->input_rtwt[gen_db].data()), 
                                                            this->batch_size * sizeof(rtwt_container<float>), 
                                                            this->comp_events[last_comp_event - 1]);
            
            overwrite_quartet_batch.acquire();
            this->comp_events.push_back(sycl::event()); 
            kernel(this->q.value(),
                   this->dev_abcd[gen_db], this->dev_rtwt[gen_db],
                   this->dev_eri_c[gen_db], this->dev_eps[gen_db],
                   {this->evt_abcd[gen_db], this->evt_rtwt[gen_db], this->comp_events[last_comp_event - 1]},
                   this->comp_events.back());

            
            this->evt_eri_c[gen_db] = this->q.value().memcpy(this->host_eri_c[gen_db],
                                                             this->dev_eri_c[gen_db],
                                                             this->batch_size * this->n512_per_quartet * this->n8_per_n512,
                                                             this->comp_events.back());
            this->evt_eps[gen_db]   = this->q.value().memcpy(this->host_eps[gen_db],
                                                             this->dev_eps[gen_db],
                                                             this->batch_size * sizeof(float),
                                                             this->comp_events.back());
            this->evt_eri_c[gen_db].wait();
            this->evt_eps[gen_db].wait();

            auto abcd_start = this->evt_abcd[gen_db].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto abcd_end   = this->evt_abcd[gen_db].get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "abcd event : " << abcd_start - abcd_start << " - " << abcd_end - abcd_start << std::endl;
            
            auto rtwt_start = this->evt_rtwt[gen_db].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto rtwt_end   = this->evt_rtwt[gen_db].get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "rtwt event : " << rtwt_start - abcd_start << " - " << rtwt_end - abcd_start << std::endl;
            
            auto comp_start = this->comp_events.back().get_profiling_info<sycl::info::event_profiling::command_start>();
            auto comp_end   = this->comp_events.back().get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "comp event : " << comp_start - abcd_start << " - " << comp_end - abcd_start << std::endl;
            
            auto eri_c_start = this->evt_eri_c[gen_db].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto eri_c_end   = this->evt_eri_c[gen_db].get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "eric event : " << eri_c_start - abcd_start << " - " << eri_c_end - abcd_start << std::endl;
            
            auto eps_start = this->evt_eps[gen_db].get_profiling_info<sycl::info::event_profiling::command_start>();
            auto eps_end   = this->evt_eps[gen_db].get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "eps  event : " << eps_start - abcd_start << " - " << eps_end - abcd_start << std::endl;


            std::cout << "gen_db = " << gen_db << std::endl;
            if(this->comp_events.size() == 3)start_consuming.release();
        }
    }

    void oneapi_acc_context::get_next_ceri(double* integral_buffer)
    {
        if(this->new_quartet_class_for_consume || ceri_ctr == this->ceri_offsets[consume_db].size() - 1)
        {
            if(this->new_quartet_class_for_consume)
            {
                std::cout << "Before consuming" << std::endl;
                start_consuming.acquire();
                std::cout << "After consuming" << std::endl;
                this->new_quartet_class_for_consume = false;
            }

            overwrite_quartet_batch.release();
            consume_db = (consume_db + 1) % 2;
            this->evt_eri_c[consume_db].wait();
            this->evt_eps[consume_db].wait();
            ceri_ctr = 0;
        }
        std::cout << "ceri_ctr = " << ceri_ctr << ", consume_db = " << consume_db << std::endl;
        std::cout << "ceri_offsets[0].size = " << ceri_offsets[0].size() << ", ceri_offsets[1].size = " << ceri_offsets[1].size() << std::endl;

        uint32_t ceri_range_start = this->ceri_offsets[consume_db][ceri_ctr];
        uint32_t ceri_range_end   = this->ceri_offsets[consume_db][ceri_ctr + 1];
        uint32_t eris_per_n512 = 512 / this->cbits;
        float* _tmp_integral_buffer = new(std::align_val_t(32)) float[n512_per_quartet * eris_per_n512];
        std::cout << "Contracting " << ceri_range_end - ceri_range_start << " quartets" << std::endl;
        for(uint32_t c = ceri_range_start;c < ceri_range_end;c++)
        {
            uint32_t c_off = c * this->n512_per_quartet * this->n8_per_n512;
            //decomp.decompress(&(this->host_eri_c[consume_db][c_off]), 
            //                  this->host_eps[consume_db][c],
            //                  this->n512_per_quartet,
            //                  _tmp_integral_buffer
            //                  );
            sdecomp.decompress(&(this->host_eri_c[consume_db][c_off]), 
                               this->host_eps[consume_db][c],
                               this->n512_per_quartet,
                               _tmp_integral_buffer
                               );
            std::vector<float> _fpga_sph_integrals = permute_sph_quartet_scalar_single[this->fpga_perm](_tmp_integral_buffer, am);
            std::vector<double> fpga_sph_integrals;
            std::transform(_fpga_sph_integrals.begin(), _fpga_sph_integrals.end(), std::back_inserter(fpga_sph_integrals), [](float elem){return double(elem);});
            std::vector<double> lmref = libint_maxl_ref.front();
            std::vector<double> luref = libint_unit_ref.front();
            std::vector<double> l2ref = libint2_ref.front();
            std::vector<double> rref = rys_ref.front();
            libint_maxl_ref.pop();
            libint_unit_ref.pop();
            libint2_ref.pop();
            rys_ref.pop();

            std::cout << "Primitive #" << c << ": ( fpga_eps = " << this->host_eps[consume_db][c] << '\n';
            std::cout << "libint-maxl-libint-unit-error: ";
            display_quartet_error<false>(this->am, &luref[0], &lmref[0]); 
            std::cout << "libint-maxl-libint2-error: ";
            display_quartet_error<false>(this->am, &l2ref[0], &lmref[0]); 
            std::cout << "libint-maxl-rys-cpu-error: ";
            display_quartet_error<false>(this->am, &rref[0], &lmref[0]); 
            std::cout << "libint-maxl-rys-fpga-error: ";
            display_quartet_error<false>(this->am, &fpga_sph_integrals[0], &lmref[0], 9.999e-5); 
            std::cout << '\n';
        }
        delete[] _tmp_integral_buffer;

        for(uint32_t q = 0;q < this->am.sph_size();q++)integral_buffer[q] = 0.0;
        
        ceri_ctr++;
    }
}
