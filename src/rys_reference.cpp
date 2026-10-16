#include <rys_reference.h>

#include <math_util.h>

#include <array>
#include <cstdint>

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
    std::vector<FLOAT> rys_quadrature(const am_container& am, 
                                      const quartet_input<double>& input_quartet, 
                                      const rtwt_container<double>& rtwt)
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
            constexpr int MAX_RTWT = rtwt._MAX_RTWT; 
            prep_roots[u] = FLOAT(rtwt.raw[u] / (1.0 + rtwt.raw[u])); // Transform u^2 -> t^2 (Somehow the rys_root/rys_weight routines in libcint/pyquante calculate u^2 s.t. u^2 = t^2/(1-t^2) and not u^2 = \rho t^2/(1-t^2), might explain the difference between the roots in libcchem and libcint?)
            prep_weights[u] = FLOAT(std::cbrt(rtwt.raw[MAX_RTWT + u] * prefactor));


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
        // std::cout << input_quartet << '\n'; 
        // std::cout << rtwt << '\n';
            
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
    
    template std::vector<double> rys_quadrature<double>(const am_container&, 
                                                        const quartet_input<double>&, 
                                                        const rtwt_container<double>&);
    template std::vector<float> rys_quadrature<float>(const am_container&, 
                                                      const quartet_input<double>&, 
                                                      const rtwt_container<double>&);
}
