#include <libheric.h>
#include <math_util.h>
#include <rys_reference.h>
#include <transform.h>
#include <util.h>

#include <libint2.h>
#include <libint2/boys.h>

#include <vectormath_exp.h>

#include <array>
#include <iostream>
#include <random>
#include <vector>


std::ostream& operator<<(std::ostream& os, const LIBINT2_REALTYPE& v)
{

    os << "[";
    constexpr int n = sizeof(LIBINT2_REALTYPE) / sizeof(double);
    for(int i = 0; i < n;i++) os << v[i] << (i == n - 1 ? "]" : ", ");
    return os;
}

template<uint32_t NVEC>
std::array<std::vector<double>, NVEC> libint_oshgp(const libheric::am_container& am, std::array<std::vector<libheric::quartet_input<double>>, NVEC> q)
{
    static_assert(NVEC == sizeof(LIBINT2_REALTYPE) / sizeof(double));
    
    uint32_t max_contr = 0;
    for(uint32_t v = 0;v < NVEC;v++)max_contr = std::max(uint64_t(max_contr), q[v].size());

    Libint_eri_t* lint_engines = new Libint_eri_t[max_contr];
    libheric_libint2_init_eri(lint_engines, am.max(), 0);

    //libint2::Shell s_a = libint2::Shell({{q.alpha()}}, {{am.a(), true, {{1.0}}}}, {{q.A()(0), q.A()(1), q.A()(2)}});
    //libint2::Shell s_b = libint2::Shell({{q.beta() }}, {{am.b(), true, {{1.0}}}}, {{q.B()(0), q.B()(1), q.B()(2)}});
    //libint2::Shell s_c = libint2::Shell({{q.gamma()}}, {{am.c(), true, {{1.0}}}}, {{q.C()(0), q.C()(1), q.C()(2)}});
    //libint2::Shell s_d = libint2::Shell({{q.delta()}}, {{am.d(), true, {{1.0}}}}, {{q.D()(0), q.D()(1), q.D()(2)}});
    //
    //std::cout << s_a << s_b << s_c << s_d << '\n';
    //eri_engine.compute(s_a, s_b, s_c, s_d);
    
    //std::vector<double> l2_res = permute_sph_quartet_scalar_double[7](eri_engine.results()[0], am);

    libheric::am_container perm_am = am;
    perm_am.make_libint_canonical();

    //double tgt_alpha = q.alpha();
    //double tgt_beta  = q.beta();
    //double tgt_gamma = q.gamma();
    //double tgt_delta = q.delta();
    
    LIBINT2_REALTYPE A[3];
    LIBINT2_REALTYPE B[3];
    LIBINT2_REALTYPE C[3];
    LIBINT2_REALTYPE D[3];

    LIBINT2_REALTYPE alpha = 0;
    LIBINT2_REALTYPE beta = 0; 
    LIBINT2_REALTYPE gamma = 0;
    LIBINT2_REALTYPE delta = 0;
   
    for(uint32_t c = 0;c < max_contr;c++)
    {
        std::array<bool, NVEC> elem_valid; 
        for(uint32_t v = 0;v < NVEC;v++)
        {
            elem_valid[v] = c < q[v].size();
            uint32_t c_idx = elem_valid[v] ? c : q[v].size() - 1;
            if(elem_valid[v])
            {
                q[v][c_idx].apply_permutation(perm_am.get_perm());
                //std::cout << "Libint input v = " << v << ", c = " << c_idx << ":\n" << q[v][c_idx] << '\n';
                //std::cout << perm_am << '\n';
            }    
            
            for(int xyz = 0;xyz < 3;xyz++)
            {
                A[xyz].insert(v,q[v][c_idx].A()[xyz]);
                B[xyz].insert(v,q[v][c_idx].B()[xyz]);
                C[xyz].insert(v,q[v][c_idx].C()[xyz]);
                D[xyz].insert(v,q[v][c_idx].D()[xyz]);
            }
            alpha.insert(v, q[v][c_idx].alpha());
            beta .insert(v, q[v][c_idx].beta());
            gamma.insert(v, q[v][c_idx].gamma());
            delta.insert(v, q[v][c_idx].delta());
        }

        LIBINT2_REALTYPE c_a = 1.0 / alpha;
        LIBINT2_REALTYPE c_b = 1.0 / beta;
        LIBINT2_REALTYPE c_c = 1.0 / gamma;
        LIBINT2_REALTYPE c_d = 1.0 / delta;

        LIBINT2_REALTYPE zeta = alpha + beta;
        LIBINT2_REALTYPE eta  = gamma + delta;
        LIBINT2_REALTYPE chi  = (zeta * eta) / (zeta + eta);

        LIBINT2_REALTYPE P[3], Q[3], W[3], AB[3], CD[3], PA[3], QC[3], PQ[3], WP[3], WQ[3];
        LIBINT2_REALTYPE AB2 = 0, CD2 = 0, PQ2 = 0;

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
        
        lint_engines[c].AB_x[0] = AB[0]; 
        lint_engines[c].AB_y[0] = AB[1]; 
        lint_engines[c].AB_z[0] = AB[2]; 
        
        lint_engines[c].CD_x[0] = CD[0]; 
        lint_engines[c].CD_y[0] = CD[1]; 
        lint_engines[c].CD_z[0] = CD[2]; 

        lint_engines[c].PA_x[0] = PA[0]; 
        lint_engines[c].PA_y[0] = PA[1]; 
        lint_engines[c].PA_z[0] = PA[2]; 
                                          
        lint_engines[c].QC_x[0] = QC[0]; 
        lint_engines[c].QC_y[0] = QC[1]; 
        lint_engines[c].QC_z[0] = QC[2]; 

        lint_engines[c].WP_x[0] = WP[0]; 
        lint_engines[c].WP_y[0] = WP[1]; 
        lint_engines[c].WP_z[0] = WP[2]; 

        lint_engines[c].WQ_x[0] = WQ[0]; 
        lint_engines[c].WQ_y[0] = WQ[1]; 
        lint_engines[c].WQ_z[0] = WQ[2]; 
        
        LIBINT2_REALTYPE _one = 1.0;

        lint_engines[c].oo2z [0] = _one / (2.0 * zeta); 
        lint_engines[c].oo2e [0] = _one / (2.0 *  eta); 
        lint_engines[c].oo2ze[0] = _one / (2.0 * (zeta + eta));
        lint_engines[c].roz  [0] = chi / zeta; 
        lint_engines[c].roe  [0] = chi /  eta;

        //std::cout << "AB_x = " << lint_engines[c].AB_x[0] << '\n';
        //std::cout << "AB_y = " << lint_engines[c].AB_y[0] << '\n';
        //std::cout << "AB_z = " << lint_engines[c].AB_z[0] << '\n';
        //
        //std::cout << "CD_x = " << lint_engines[c].CD_x[0] << '\n';
        //std::cout << "CD_y = " << lint_engines[c].CD_y[0] << '\n';
        //std::cout << "CD_z = " << lint_engines[c].CD_z[0] << '\n';
        //
        //std::cout << "PA_x = " << lint_engines[c].PA_x[0] << '\n';
        //std::cout << "PA_y = " << lint_engines[c].PA_y[0] << '\n';
        //std::cout << "PA_z = " << lint_engines[c].PA_z[0] << '\n';
        //
        //std::cout << "QC_x = " << lint_engines[c].QC_x[0] << '\n';
        //std::cout << "QC_y = " << lint_engines[c].QC_y[0] << '\n';
        //std::cout << "QC_z = " << lint_engines[c].QC_z[0] << '\n';
        //
        //std::cout << "WP_x = " << lint_engines[c].WP_x[0] << '\n';
        //std::cout << "WP_y = " << lint_engines[c].WP_y[0] << '\n';
        //std::cout << "WP_z = " << lint_engines[c].WP_z[0] << '\n';
        //
        //std::cout << "WQ_x = " << lint_engines[c].WQ_x[0] << '\n';
        //std::cout << "WQ_y = " << lint_engines[c].WQ_y[0] << '\n';
        //std::cout << "WQ_z = " << lint_engines[c].WQ_z[0] << '\n';
        //
        //std::cout << "oo2z  = " << lint_engines[c].oo2z[0] << '\n';
        //std::cout << "oo2e  = " << lint_engines[c].oo2e[0] << '\n';
        //std::cout << "oo2ze = " << lint_engines[c].oo2ze[0] << '\n';
        //std::cout << "roz = " << lint_engines[c].roz[0] << '\n';
        //std::cout << "roe = " << lint_engines[c].roe[0] << '\n';

        LIBINT2_REALTYPE maxl_norm = 1.0;
        LIBINT2_REALTYPE exps[4] = {alpha, beta, gamma, delta}; 
        for(uint32_t i = 0;i < 4;i++)
        {
            LIBINT2_REALTYPE term1 = sqrt(sqrt(pow(2.0 * exps[i]/std::numbers::pi, 3)) * pow(4.0 * exps[i], perm_am.raw[i]));
            LIBINT2_REALTYPE term2 = sqrt(LIBINT2_REALTYPE(libheric::factorial(perm_am.raw[i]))/LIBINT2_REALTYPE(libheric::factorial(2*perm_am.raw[i])) * pow(LIBINT2_REALTYPE(2.0), perm_am.raw[i]));
            maxl_norm *= term1 * term2;
        }
        
        LIBINT2_REALTYPE kappa = exp(-((alpha * beta * AB2)/zeta + (gamma * delta * CD2)/eta));
        LIBINT2_REALTYPE c_fac = c_a * c_b * c_c * c_d;
        LIBINT2_REALTYPE prefactor = LIBINT2_REALTYPE(libheric::two_pi_52) * maxl_norm * c_fac * kappa * sqrt(chi / (zeta * eta)) / (zeta * eta);
      
        //std::cout << "prefactor = " << prefactor << '\n';

        //std::cout << "Evaluating boys functions" << std::endl; 
        uint32_t total_am = am.a() + am.b() + am.c() + am.d();
        libint2::FmEval_Chebyshev7 fm_eval(total_am);
        double F[NVEC * (total_am + 1)];
        for(uint32_t v = 0;v < NVEC;v++)
        {
            if(elem_valid[v])
            {
                double X = PQ2[v] * chi[v];
                double _F[total_am + 1];
                fm_eval.eval(_F, X, total_am);
                for(uint32_t i = 0;i <= total_am;i++)F[i * NVEC + v] = _F[i] * prefactor[v];
            }
            else
            {
                for(uint32_t i = 0;i <= total_am;i++)F[i * NVEC + v] = 0.0;
            }
        }
        memcpy(&lint_engines[c].LIBINT_T_SS_EREP_SS(0)[0], F, (total_am + 1) * sizeof(LIBINT2_REALTYPE));
        //std::cout << "LIBINT_T_SS_EREP_SS(0)[0] = " << lint_engines[c].LIBINT_T_SS_EREP_SS(0)[0] << '\n';
        //std::cout << "Evaluated boys functions" << std::endl;
        //std::cout << "Permuted am_a = " << perm_am.a() << ", am_b = " << perm_am.b() << ", am_c = " << perm_am.c() << ", am_d = " << perm_am.d() << std::endl;
        //std::cout << "eri_fct_ptr = " << libheric_libint2_build_eri[perm_am.a()][perm_am.b()][perm_am.c()][perm_am.d()] << std::endl;
    }
    lint_engines->contrdepth = max_contr;
    libheric_libint2_build_eri[perm_am.a()][perm_am.b()][perm_am.c()][perm_am.d()](lint_engines);
    LIBINT2_REALTYPE* results = lint_engines->targets[0];
    //std::cout << "Computed ERIs with libint" << std::endl;

    std::array<std::vector<double>, NVEC> eris;
    for(uint32_t v = 0;v < NVEC;v++)
    {
        for(uint32_t q = 0;q < am.cart_size();q++)
        {
            eris[v].push_back(results[q][v]);
        }
    } 
    libheric_libint2_cleanup_eri(lint_engines);
    delete[] lint_engines;
    return eris;
    //auto unit_norm_lambda = [](int32_t l, int32_t lx, int32_t ly, int32_t lz, double exp) {
    //    double term1 = std::sqrt(std::sqrt(std::pow(2.0 * exp/std::numbers::pi, 3.0)) * std::pow(4.0 * exp, l));
    //    double term2 = std::sqrt(dfac[std::max(2 * lx - 1, 0)] * dfac[std::max(2 * ly - 1, 0)] * dfac[std::max(2 * lz - 1, 0)]);
    //    return term1 / term2;
    //};
    
    //libint integrals are stored "Row major", i.e. (a0b0|c0d0), (a0b0|c0d1), (a0b0|c0d2), ...
    //while CP2K expects integral stored as "Column major", i.e. (a0b0|c0d0), (a1b0|c0d0), ...

    //for(int32_t d1 = 0;d1 <= am.d();d1++)
    //{
    //for(int32_t d2 = 0;d2 <= d1;d2++)
    //{
    //    int32_t dx = am.d() - d1;
    //    int32_t dy = d1 - d2;
    //    int32_t dz = d2;
    //    double unit_norm_d = unit_norm_lambda(am.d(), dx, dy, dz, tgt_delta);
    //    for(int32_t c1 = 0;c1 <= am.c();c1++)
    //    {
    //    for(int32_t c2 = 0;c2 <= c1;c2++)
    //    {
    //        int32_t cx = am.c() - c1;
    //        int32_t cy = c1 - c2;
    //        int32_t cz = c2;
    //        double unit_norm_cd = unit_norm_d * unit_norm_lambda(am.c(), cx, cy, cz, tgt_gamma);

    //        for(int32_t b1 = 0;b1 <= am.b();b1++)
    //        {
    //        for(int32_t b2 = 0;b2 <= b1;b2++)
    //        {
    //            int32_t bx = am.b() - b1;
    //            int32_t by = b1 - b2;
    //            int32_t bz = b2;
    //            double unit_norm_bcd = unit_norm_cd * unit_norm_lambda(am.b(), bx, by, bz, tgt_beta);
    //            for(int32_t a1 = 0;a1 <= am.a();a1++)
    //            {
    //            for(int32_t a2 = 0;a2 <= a1;a2++)
    //            {
    //                int32_t ax = am.a() - a1;
    //                int32_t ay = a1 - a2;
    //                int32_t az = a2;
    //                double unit_norm_abcd = unit_norm_bcd * unit_norm_lambda(am.a(), ax, ay, az, tgt_alpha);

    //                maxl_eris.push_back(permuted_results[eri_ctr] * maxl_norm);
    //                unit_eris.push_back(permuted_results[eri_ctr] * unit_norm_abcd);

    //                eri_ctr++;
    //            }
    //            }
    //        }
    //        }
    //    }
    //    }
    //}
    //}
}

int main()
{
    libheric_libint2_static_init();

    constexpr uint32_t n_vec = sizeof(LIBINT2_REALTYPE) / sizeof(double);

    std::cout << "sizeof(LIBINT2_REALTYPE) = " << sizeof(LIBINT2_REALTYPE) << '\n';
    std::cout << "#vec elements = " << n_vec << '\n';

    uint32_t max_contr = 7;

    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<double> pos_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> exp_dist(0.1, 2.0);
    std::uniform_int_distribution<uint32_t> contr_dist(1, max_contr);

    constexpr int LOCAL_MAX_AM = libheric::MAX_AM;
    libheric::quartet_transform qt(false);
    for(uint32_t la = 0; la <= LOCAL_MAX_AM;la++) {
    for(uint32_t lb = 0; lb <= LOCAL_MAX_AM;lb++) {
    for(uint32_t lc = 0; lc <= LOCAL_MAX_AM;lc++) {
    for(uint32_t ld = 0; ld <= LOCAL_MAX_AM;ld++) {
        std::array<std::vector<double>, n_vec> rys_ref;
        std::array<std::vector<libheric::quartet_input<double>>, n_vec> qi;
        libheric::am_container am(la, lb, lc, ld);
        std::cout << "unpermuted      : " << am << '\n';
        libheric::am_container perm_am = am;
        
        perm_am.make_canonical();
        std::cout << "canonical       : " << perm_am << '\n';
        if(!perm_am.is_canonical())
        {
            std::cout << "[ERROR] am not canonical after permutation!\n";
            return 0;
        }
        perm_am.undo_perm();
        if(!(perm_am == am))
        {
            std::cout << "[ERROR] am -> canonical -> am failed!\n";
            return 0;
        }

        perm_am.make_fpga_canonical();
        std::cout << "fpga canonical  : " << perm_am << '\n';
        if(!perm_am.is_fpga_canonical())
        {
            std::cout << "[ERROR] am not FPGA canonical after permutation!\n";
            return 0;
        }
        perm_am.undo_perm();
        if(!(perm_am == am))
        {
            std::cout << "[ERROR] am -> FPGA canonical -> am failed!\n";
            return 0;
        }
        
        perm_am.make_libint_canonical();
        std::cout << "libint canonical: " << perm_am << '\n';
        if(!perm_am.is_libint_canonical())
        {
            std::cout << "[ERROR] am not LIBINT canonical after permutation!\n";
            return 0;
        }
        perm_am.undo_perm();
        if(!(perm_am == am))
        {
            std::cout << "[ERROR] am -> LIBINT canonical -> am failed!\n";
            return 0;
        }


        std::cout << "contractions: ";
        uint32_t n_rtwt = am.total() / 2 + 1;
        int32_t off_dim_low = -1, off_dim_high = 1;
        for (uint32_t v = 0;v < n_vec;v++)
        {
            libheric::quartet_input<double> _qi;
            _qi.A() = {pos_dist(gen), pos_dist(gen), pos_dist(gen)};
            _qi.B() = {pos_dist(gen), pos_dist(gen), pos_dist(gen)};
            _qi.C() = {pos_dist(gen), pos_dist(gen), pos_dist(gen)};
            _qi.D() = {pos_dist(gen), pos_dist(gen), pos_dist(gen)};
            auto A = _qi.A();
            auto B = _qi.B();
            auto C = _qi.C();
            auto D = _qi.D();
            rys_ref[v] = std::vector(am.cart_size(), 0.0);
            uint32_t contr = contr_dist(gen);
            for(uint32_t c = 0;c < contr;c++)
            {

                _qi.alpha() = exp_dist(gen);
                _qi.beta() = exp_dist(gen);
                _qi.gamma() = exp_dist(gen);
                _qi.delta() = exp_dist(gen);
                
                //std::cout << "v = " << v << ", c = " << c << ":\n" << _qi << '\n'; 

                double alpha = _qi.alpha();
                double beta  = _qi.beta();
                double gamma = _qi.gamma();
                double delta = _qi.delta();

                double c_a = 1.0 / alpha; // Fake contraction factor
                double c_b = 1.0 / beta;
                double c_c = 1.0 / gamma;
                double c_d = 1.0 / delta;
                double c_fac = c_a * c_b * c_c * c_d;

                double zeta = alpha + beta;
                double eta  = gamma + delta;

                auto P = (alpha * A + beta * B) / zeta;
                auto Q = (gamma * C + delta * D) / eta;
                for(int32_t Q_off_x = off_dim_low;Q_off_x <= 1;Q_off_x++)
                {
                for(int32_t Q_off_y = off_dim_low;Q_off_y <= 1;Q_off_y++)
                {
                for(int32_t Q_off_z = off_dim_low;Q_off_z <= 1;Q_off_z++)
                {
                    if(pos_dist(gen) >= 0.8) break; // Use only about every 10th offset
                    Eigen::Vector3d Q_off = {double(Q_off_x), double(Q_off_y), double(Q_off_z)};
                    auto __qi = _qi;
                    __qi.C() += Q_off;
                    __qi.D() += Q_off;
                    auto QQ = Q + Q_off;
                    auto PQQ = P - QQ;
                    double PQQ2 = PQQ.squaredNorm();
                    
                    double rhopqq2 = zeta * eta / (zeta + eta) * PQQ2; 
                    libheric::rtwt_container<double> rtwt;
                    rtwt.generate(n_rtwt, rhopqq2);

                    auto rys_res = rys_quadrature<double>(am, __qi, rtwt);
                    for(size_t q = 0;q < rys_res.size();q++)rys_ref[v][q] += rys_res[q] * c_fac;
                    qi[v].push_back(__qi);
                }
                }
                }
            }
            std::cout << qi[v].size() << (v == n_vec - 1 ? "\n" : ", ");
        }
        auto l2_res = libint_oshgp<n_vec>(am, qi);
        
        libheric::am_container l2_in_am = am;
        l2_in_am.make_libint_canonical();
        auto l2_out_am = l2_in_am;
        l2_out_am.reverse();
        int output_perm = l2_out_am.get_perm();
        std::cout << "output_perm = " << output_perm << '\n';
        qt.set_quartet_class(l2_out_am); 
       
        for(uint32_t v = 0;v < n_vec;v++)
        {
            auto perm_l2_res = libheric::permute_cart_quartet_scalar_double[output_perm](l2_res[v].data(), am);
            std::cout << "Cart error\n";
            display_quartet_error<true>(am, perm_l2_res.data(), rys_ref[v].data(), 1e-7); 
            
            auto sph_rys = qt.naive_cart2sph(am, rys_ref[v]);
            auto sph_l2 = qt.naive_cart2sph(am, perm_l2_res);
            std::cout << "Naive error\n";
            display_quartet_error<false>(am, sph_l2.data(), sph_rys.data(), 1e-7);

            qt.apply_transformation(qt, l2_res[v].data(), l2_res[v].data());
            std::cout << "Opt error\n";
            display_quartet_error<false>(am, l2_res[v].data(), sph_rys.data(), 1e-7);

            std::cout << '\n'; 

        } 
        std::cout << std::string(50, '-') << '\n';
    }
    }
    }
    }


    libheric_libint2_static_cleanup();
}
