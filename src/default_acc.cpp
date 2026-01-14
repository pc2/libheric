#include "default_acc.h"
#include "constants.h"
#include "math_util.h"

#include <algorithm>
#include <fstream>
#include <optional>

#include <omp.h>

namespace libheric
{
    bool operator<(const acc_kind& lhs, const acc_kind& rhs)
    {
        return (lhs.name < rhs.name) || 
               (lhs.name == rhs.name && lhs.type < rhs.type) ||
               (lhs.name == rhs.name && lhs.type == rhs.type && lhs.library < rhs.library);
    }

    std::ostream& operator<<(std::ostream& os, const acc_kind& a)
    {
        os << "(" << a.name << ", " << a.type << ", " << a.library << ")";
        return os;
    }

    std::vector<acc_device> discover_cpu_accelerator(const acc_kind& acc_query)
    {
        std::vector<acc_device> found_cpu;
        std::ifstream f_cpuinfo("/proc/cpuinfo");
        if(f_cpuinfo.is_open())
        {
            std::optional<std::string> cpu_name;
            for(std::string line; std::getline(f_cpuinfo,line);)
            {
                if(line.find("model name") != std::string::npos)
                {
                    auto delim_pos = line.find(":");
                    if(delim_pos != std::string::npos)
                    {
                        cpu_name = line.substr(delim_pos + 1);
                        // Trim leading whitespace
                        cpu_name = cpu_name->substr(
                            std::find_if(cpu_name->begin(), cpu_name->end(), [](char c){return !std::isspace(c);}) - cpu_name->begin()
                        );
                        break;
                    }
                }
            }
            if(cpu_name.has_value())
            {
                int num_cores = omp_get_max_threads();
                found_cpu.push_back({cpu_name.value(), "FFFF:FF:FF.F", cpu_name.value(), num_cores, acc_query});
            }
            else
            {
                std::cout << "[WARNING] Found /proc/cpuinfo but unable to query model name!\n";
            }
        }
        else
        {
            std::cout << "[WARNING] Unable to query cpuinfo!\n";
        }
        return found_cpu;
    }

    acc_context::acc_context(const acc_device& acc_dev,
                             const simulation_box& sim_box) 
        : dev(acc_dev),
          sim_box(sim_box),
          set_offs(sim_box.set_offsets.rows(), 4),    
          set_sizes(sim_box.set_offsets.rows(), 4)    
    {
        std::cout << "Cell_vectors:\n";
        std::cout << "e_x = [" << sim_box.cell_mat.col(0).transpose() << "]^T\n";
        std::cout << "e_y = [" << sim_box.cell_mat.col(1).transpose() << "]^T\n";
        std::cout << "e_z = [" << sim_box.cell_mat.col(2).transpose() << "]^T\n\n";
       
        Eigen::MatrixX4d tmp_pos_kind(sim_box.positions.cols(), 4);
        tmp_pos_kind(Eigen::all, Eigen::seq(0, 2)) = sim_box.positions.transpose();
        tmp_pos_kind(Eigen::all, Eigen::seq(3, 3)) = sim_box.kinds.cast<double>();

        std::cout << "Positions:\n x   y   z   kind\n";
        std::cout << tmp_pos_kind << "\n\n";
        
        std::stringstream ss_set_str;
        std::cout << spdf << '\n';
        ss_set_str << "exp ";
        for(int i = 0;i <= MAX_AM;i++)
        {
            for(int j = 0;j < sim_box.shell_sizes(i);j++)
            {
                ss_set_str << spdf[i] << j << "  ";
            }
        }
        std::cout << ss_set_str.str() << '\n';


        for(int i = 0;i < sim_box.kind_offsets.size();i++)
        {
            std::cout << "Kind = " << i 
                      << " (offset = " << sim_box.kind_offsets(i) 
                      << ", size = " << sim_box.kind_sizes(i) << ")\n";
            int curr_offset = sim_box.kind_offsets(i);
            for(int j = 0;j < sim_box.kind_sizes(i);j++)
            {
                std::cout << "   Set = " << j
                          << " (offset = " << sim_box.set_offsets(j, i) 
                          << ", size = " << sim_box.set_sizes(j, i) << ")\n";
                Eigen::MatrixXd exp_coeff(sim_box.set_sizes(j, i), sim_box.coefficients.cols() + 1);
                int set_offset = curr_offset + sim_box.set_offsets(j, i);
                int set_size = sim_box.set_sizes(j, i);
                exp_coeff(Eigen::all, Eigen::seq(0, 0)) = sim_box.exponents(Eigen::seqN(set_offset, set_size));
                exp_coeff(Eigen::all, Eigen::seq(1, Eigen::last)) 
                    = sim_box.coefficients(Eigen::seqN(set_offset, set_size), Eigen::all);
                std::cout << "     " << ss_set_str.str() << '\n';
                for(int k = 0;k < exp_coeff.rows();k++)
                    std::cout << "     " << exp_coeff(Eigen::seq(k, k), Eigen::all) << '\n';
                std::cout << '\n';
            }
            std::cout << std::string(50, '-') << "\n\n";
        }

        for(int32_t l = 0;l <= MAX_AM;l++)
        {
            std::vector<Eigen::Triplet<double>> unc2uns_trps;
            std::vector<Eigen::Triplet<double>> mnc2uns_trps;
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
                        double unit_norm = std::sqrt(
                                                      ( 
                                                        double(factorial(2 * lx))
                                                      * double(factorial(2 * ly))
                                                      * double(factorial(2 * lz))
                                                      * double(factorial(l))
                                                      * double(factorial(l - m))
                                                      )
                                                      /
                                                      (
                                                        double(factorial(2 * l))
                                                      * double(factorial(lx))
                                                      * double(factorial(ly))
                                                      * double(factorial(lz))
                                                      * double(factorial(l + m))
                                                      )
                                                    );
                        double maxl_norm = std::sqrt(double(factorial(l - m)) / double(factorial(l + m)));
                        double common_factor = 1.0 / (powern(2.0, l) * double(factorial(l)));

                        unit_norm *= common_factor;
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
                            std::cout << "non-zero from Y_{m} + Y_{-m} (" << outer_sum1 * unit_norm << ") for l = (" << lx << ", " << ly << ", " << lz << "), " << " m = " << m << std::endl;
                            int32_t sph_idx = l + m;
                            unc2uns_trps.push_back({sph_idx, cart_idx, outer_sum1 * unit_norm});
                            mnc2uns_trps.push_back({sph_idx, cart_idx, outer_sum1 * maxl_norm});
                        }
                        if(outer_sum2 != 0)
                        {
                            int32_t sph_idx = l - m;
                            std::cout << "non-zero from Y_{m} - Y_{-m} (" << outer_sum1 * unit_norm << ") for l = (" << lx << ", " << ly << ", " << lz << "), " << " m = " << m << std::endl;
                            unc2uns_trps.push_back({sph_idx, cart_idx, outer_sum2 * unit_norm});
                            mnc2uns_trps.push_back({sph_idx, cart_idx, outer_sum2 * maxl_norm});
                        }

                    }
                }
            }
            unc2uns.push_back(Eigen::SparseMatrix<double>(nso[l], nco[l]));
            mnc2uns.push_back(Eigen::SparseMatrix<double>(nso[l], nco[l]));
            unc2uns.back().setFromTriplets(unc2uns_trps.begin(), unc2uns_trps.end());
            mnc2uns.back().setFromTriplets(mnc2uns_trps.begin(), mnc2uns_trps.end());
            std::cout << "l = " << l << '\n';
            std::cout << "unit normalized cartesian: \n";
            std::cout << unc2uns.back() << '\n';
            std::cout << "max-L normalized cartesian: \n";
            std::cout << mnc2uns.back() << '\n';
        }

        is_functional = false;
    }

    acc_context::~acc_context()
    {
    }

	bool acc_context::functional(){return this->is_functional;}

    void acc_context::set_quartet(int la, int lb, int lc, int ld)
    {
        if(la < 0 || la > MAX_AM || 
           lb < 0 || lb > MAX_AM || 
           lc < 0 || lc > MAX_AM ||
           ld < 0 || ld > MAX_AM)
        {
            std::cout << "[ERROR] Quartet = (" 
                      << la << lb << "|" 
                      << lc << ld << ") out of bounds -- "
                      << " library only configured for am <= " << MAX_AM << '\n';
            this->is_functional = false;
            return;
        }

        std::string abcd = spdf.substr(la, 1) + spdf.substr(lb, 1) + spdf.substr(lc, 1) + spdf.substr(ld, 1);

        if(!am_container(la, lb, lc, ld).is_canonical())
        {
            std::cout << "[ERROR] Quartet = " << abcd << " is not canonical\n";
            this->is_functional = false;
            return;
        }
        this->la = la;
        this->lb = lb;
        this->lc = lc;
        this->ld = ld;
        this->am = {la, lb, lc, ld};
        this->abcd = abcd;

        this->shell_off(0) = sim_box.shell_offsets(la);
        this->shell_off(1) = sim_box.shell_offsets(lb);
        this->shell_off(2) = sim_box.shell_offsets(lc);
        this->shell_off(3) = sim_box.shell_offsets(ld);

        this->is_functional = true;
        std::cout << "[WARNING] quartet = " << abcd << " not implemented for CPU\n";
    }

    void acc_context::set_quartet_permutation(int perm)
    {
        if(perm < 0 || perm >= NUM_PERMS)
        {
            std::cout << "[ERROR] quartet permutation number not in 0 <= " << perm << " < " << NUM_PERMS << "\n";
            this->is_functional = false;
            return;
        }
        this->perm = perm;

        // After applying the last permutation symmetry (ab|cd) -> (cd|ab), canonical quartets are always libint canonical
        this->libint_perm = perm ^ 0x4;

    }
    
    void acc_context::set_atom_id(int quartet_pos, int atom_id)
    {
        int kind = sim_box.kinds(atom_id);
        this->kind_off(quartet_pos) = sim_box.kind_offsets(kind);
        this->set_offs(Eigen::all, quartet_pos) = sim_box.set_offsets(Eigen::all, kind);
        this->set_sizes(Eigen::all, quartet_pos) = sim_box.set_sizes(Eigen::all, kind);
    }

    void acc_context::set_set_id(int quartet_pos, int set_id)
    {
        this->set_off(quartet_pos) = this->kind_off(quartet_pos) + this->set_offs(set_id, quartet_pos);
        this->set_size(quartet_pos) = this->set_sizes(set_id, quartet_pos);
    }

    void acc_context::set_shell_id(int quartet_pos, int shell_id)
    {
        int qp_set_size  = this->set_size(quartet_pos);
        int qp_set_off   = this->set_off(quartet_pos);
        int qp_shell_off = this->shell_off(quartet_pos);

        //std::cout << "qp_set_size  = " << qp_set_size << '\t';
        //std::cout << "qp_set_off   = " << qp_set_off << '\t';
        //std::cout << "qp_shell_off = " << qp_shell_off << '\t';
        //std::cout << "shell_id = " << shell_id << '\n';

        this->coeffs(Eigen::seqN(0, qp_set_size), quartet_pos) = sim_box.coefficients(Eigen::seqN(qp_set_off, qp_set_size), qp_shell_off + shell_id);
    }

    void acc_context::set_exponent_id(int quartet_pos, int exp_id)
    {
        this->coeff(quartet_pos) = this->coeffs(exp_id, quartet_pos);
        this->exp_off(quartet_pos) = this->set_off(quartet_pos) + exp_id; 
        this->coeff_prod(quartet_pos) = (quartet_pos == 0 ? 1.0 : this->coeff_prod(quartet_pos - 1)) * this->coeff(quartet_pos);

        //std::cout << "Set coeff = " << this->coeff(quartet_pos) << '\t';
        //std::cout << "exp_off = " << this->exp_off(quartet_pos) << '\n';
    }
    
    void acc_context::set_offset(int off_pos, int off_x, int off_y, int off_z)
    {
        this->offsets(Eigen::all, off_pos) = Eigen::Vector3i(off_x, off_y, off_z);    
        this->offsets_d(Eigen::all, off_pos) = Eigen::Vector3d(double(off_x), double(off_y), double(off_z));    
    }

    void acc_context::submit_peri()
    {
    }
    
    void acc_context::submit_ceri()
    {
    }

    void acc_context::finish_quartet_class()
    {
    }

    void acc_context::get_next_ceri(double* integral_buffer)
    {
    }
}
