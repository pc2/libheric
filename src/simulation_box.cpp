#include <simulation_box.h>

#include <logging.h>
#include <math_util.h>

#include <iostream>
#include <vector>

namespace libheric
{

    simulation_box::simulation_box(const Eigen::Matrix3d&  cell_mat, 
                                   const Eigen::Matrix3Xd& positions,
                                   const std::vector<int>&    kinds,
                                   const std::vector<double>& exponents,
                                   const std::vector<double>& coefficients,
                                   const std::vector<int>&    kind_nset,
                                   const std::vector<int>&    set_nexp,
                                   const std::vector<int>&    set_nshl,
                                   int max_set, int nkind, int l_max, int natom)
        : 
        cell_mat          (cell_mat          ),
        positions         (positions         ),
        kinds             (kinds             ),
        exponents         (exponents         ),
        coefficients      (coefficients      ),
        kind_nset         (kind_nset         ),
        set_nexp          (set_nexp          ),
        set_nshl          (set_nshl          ),          // Use default value that results in OOB
        coeff_offsets     (max_set * nkind * (l_max + 1), std::numeric_limits<int>::lowest()),
        exp_offsets       (max_set * nkind * (l_max + 1), std::numeric_limits<int>::lowest()),
        norm_offsets      (max_set * nkind * (l_max + 1), std::numeric_limits<int>::lowest()),
        normalizations    (),
        max_set(max_set), nkind(nkind), l_max(l_max), natom(natom)
    {

        constexpr double pi_32 = std::sqrt(std::numbers::pi * std::numbers::pi * std::numbers::pi);
        int32_t coeff_ctr = 0;
        int32_t exp_ctr = 0;
        int32_t norm_ctr = 0;
        for(int32_t l = 0;l <= l_max;l++)
        {
            uint32_t max_shellset_count = 0;
            uint32_t max_shellprim_count = 0;
            for(int32_t kind = 0;kind < nkind;kind++)
            {
                int32_t nset = kind_nset[l * nkind + kind];
                uint32_t shellset_count = 0;
                for(int32_t set = 0;set < nset;set++)
                {
                    int32_t lkindset = (l * nkind + kind) * max_set + set;
                    int32_t nshl = set_nshl[lkindset];
                    int32_t nexp = set_nexp[lkindset];
                    coeff_offsets[lkindset] = coeff_ctr;
                    exp_offsets[lkindset] = exp_ctr;
                    norm_offsets[lkindset] = norm_ctr;

                    for(int32_t shl = 0;shl < nshl;shl++)
                    {
                        double norm = 0.0;
                        for(int32_t exp1 = 0;exp1 < nexp;exp1++)
                        {
                            for(int32_t exp2 = 0;exp2 < nexp;exp2++)
                            {
                                double a1 = exponents[exp_ctr + exp1];
                                double a2 = exponents[exp_ctr + exp2];
                                double c1 = coefficients[coeff_ctr + shl * nexp + exp1];
                                double c2 = coefficients[coeff_ctr + shl * nexp + exp2];
                                norm += c1 * c2 / std::sqrt(powern(a1 + a2, 2 * l + 3));
                            }
                        }
                        norm *= pi_32 * dfac[std::max(0, 2 * l - 1)] / powern(2.0, l); 
                        normalizations.push_back(norm == 0 ? 0.0 : 1.0 / std::sqrt(norm));
                    }
                    shellset_count += nshl;
                    max_shellprim_count = std::max(max_shellprim_count, uint32_t(nshl * nexp));
                    
                    coeff_ctr += nshl * nexp;
                    exp_ctr   += nexp;
                    norm_ctr  += nshl;
                }
                max_shellset_count = std::max(max_shellset_count, shellset_count);
            }
            max_shellsets_for_l[l] = max_shellset_count;
            max_shellprims_for_l[l] = max_shellprim_count;
        }
    }

    uint32_t simulation_box::max_shellset_quartets(const am_container& am)
    {
        if constexpr(DEBUG_PRINT)
        {
            std::ostream* logger = get_logger();
            *logger << MESSAGE_TYPE::TIMESTAMP << '\n'; 
            *logger << "max_shellset_for_l(la = " << am.a() << ") = " << this->max_shellsets_for_l[am.a()] << '\n';
            *logger << "max_shellset_for_l(lb = " << am.b() << ") = " << this->max_shellsets_for_l[am.b()] << '\n';
            *logger << "max_shellset_for_l(lc = " << am.c() << ") = " << this->max_shellsets_for_l[am.c()] << '\n';
            *logger << "max_shellset_for_l(ld = " << am.d() << ") = " << this->max_shellsets_for_l[am.d()] << '\n';
        }

        return this->max_shellsets_for_l[am.a()]
             * this->max_shellsets_for_l[am.b()]
             * this->max_shellsets_for_l[am.c()]
             * this->max_shellsets_for_l[am.d()];
    }
    
    uint32_t simulation_box::max_shellprim_quartets(const am_container& am)
    {
        std::ostream* logger = get_logger();
        if constexpr(DEBUG_PRINT)
        {
            std::ostream* logger = get_logger();
            *logger << MESSAGE_TYPE::TIMESTAMP << '\n'; 
            *logger << "max_shellprim_for_l(la = " << am.a() << ") = " << this->max_shellprims_for_l[am.a()] << '\n';
            *logger << "max_shellprim_for_l(lb = " << am.b() << ") = " << this->max_shellprims_for_l[am.b()] << '\n';
            *logger << "max_shellprim_for_l(lc = " << am.c() << ") = " << this->max_shellprims_for_l[am.c()] << '\n';
            *logger << "max_shellprim_for_l(ld = " << am.d() << ") = " << this->max_shellprims_for_l[am.d()] << '\n';
        }

        return this->max_shellprims_for_l[am.a()]
             * this->max_shellprims_for_l[am.b()]
             * this->max_shellprims_for_l[am.c()]
             * this->max_shellprims_for_l[am.d()];
    }

    uint64_t simulation_box::mem_usage() const
    {
        uint64_t sz = 0;
        sz += (cell_mat.size() + positions.size() + exponents.size() + coefficients.size() + normalizations.size()) * sizeof(double);
        sz += (kinds.size() + kind_nset.size() + set_nexp.size() + set_nshl.size()) * sizeof(int);
        sz += (coeff_offsets.size() + exp_offsets.size() + norm_offsets.size()) * sizeof(int);
        sz += (max_shellsets_for_l.size() + max_shellprims_for_l.size()) * sizeof(int);
        return sz;
    }

    void simulation_box::update_positions(const Eigen::Matrix3Xd& new_positions)
    {
        std::ostream* logger = get_logger();
        if(new_positions.cols() != this->positions.cols())
        {
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << "Updating the positions changes the amount of particles in the system from "
                   << this->positions.cols() << " to " << new_positions.cols() << '\n';
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
        }
        this->positions = new_positions;
    }
    
    void simulation_box::update_cell(const Eigen::Matrix3d& new_cell_mat)
    {
        this->cell_mat = new_cell_mat;
    }
}
