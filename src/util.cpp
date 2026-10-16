#include "util.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace libheric 
{
    template<bool IS_CART>
    void display_quartet_error(const am_container& am, double* quartet, double* quartet_ref, double eps, std::ostream* os)
    {

        uint32_t nga = IS_CART ? am.nca() : am.nsa(); 
        uint32_t ngb = IS_CART ? am.ncb() : am.nsb(); 
        uint32_t ngc = IS_CART ? am.ncc() : am.nsc(); 
        uint32_t ngd = IS_CART ? am.ncd() : am.nsd(); 
        uint32_t q_size = IS_CART ? am.cart_size() : am.sph_size();

        double max_error = 0.0;
        for(uint32_t q = 0;q < q_size;q++)
        {
            double res_round = std::round(quartet[q] / eps) * eps;
            double ref_round = std::round(quartet_ref[q] / eps) * eps;
            double error = std::abs(res_round - ref_round);
            max_error = std::max(max_error, error);
        } 

        *os << " max_error = " << max_error << ", error_tolerance = " << eps << '\n';
            
        std::ios::fmtflags restore_flags = (*os).flags();
        uint64_t n_digits = (uint64_t)std::ceil(-std::log10(eps));
        *os << std::scientific << std::setprecision(n_digits + 1);

        if(max_error > eps)
        {
            for(uint32_t dd = 0;dd < ngd;dd++)
            {
                for(uint32_t cc = 0;cc < ngc;cc++)
                {
                    *os << " d = " << dd << ", c = " << cc << '\n';
                    for(uint32_t bb = 0;bb < ngb;bb++)
                    {
                        *os << "\tb" << bb << ": ";
                        for(uint32_t aa = 0;aa < nga;aa++)
                        {
                            uint32_t q = ((dd * ngc + cc) * ngb + bb) * nga + aa;
                            double res_round = std::round(quartet[q] / eps) * eps;
                            double ref_round = std::round(quartet_ref[q] / eps) * eps;
                            double error = std::abs(res_round - ref_round);
                            *os << std::setw(n_digits + 6) << quartet_ref[q] << "(" 
                                << std::setw(n_digits + 6) << quartet[q] << ", " 
                                << std::setw(n_digits + 6) << error << ")" 
                                << (aa < nga - 1 ? ", " : "\n");
                        }
                    }
                }
            }
        }
        (*os).flags(restore_flags);
    }

    template void display_quartet_error<true>(const am_container&, double*, double*, double, std::ostream*);
    template void display_quartet_error<false>(const am_container&, double*, double*, double, std::ostream*);
    
    template<bool IS_CART>
    void compare_atom_quartets(const am_container& am, 
                               const abstract_acc_context::atom_quartet_descriptor& atom_desc,
                               const abstract_acc_context::atom_quartet_descriptor& ref_desc,
                               double eps,
                               uint64_t data_inc)
    {
        double* shellset_data     = (double*)atom_desc.first_shellset_data_ptr;
        double* ref_shellset_data = (double*)ref_desc.first_shellset_data_ptr;
        auto* shellset_ids        = atom_desc.first_shellset_ptr;
        auto* ref_shellset_ids    = ref_desc.first_shellset_ptr;
        uint64_t ref_shlset = 0;

        std::ostream* logger;
        if constexpr(LOG_PRINT)logger = get_logger();
        else logger = &std::cout;

        if(data_inc == 0)data_inc = am.sph_size();

        std::array<uint8_t, 8> _tmp = ref_shellset_ids[ref_shlset];
        for(uint64_t i = 0;i < 8;i++)_tmp[i]--;
        uint64_t ref_shellset_id_u64 = bytes_to_u64(_tmp);
        for(uint64_t shlset = 0;shlset < atom_desc.n_shellset_quartets;shlset++, shellset_data += data_inc)
        {
            uint64_t shellset_id_u64 = bytes_to_u64(shellset_ids[shlset]);
            while(ref_shlset < ref_desc.n_shellset_quartets && ref_shellset_id_u64 < shellset_id_u64)
            {
                double ref_max = *std::max_element(ref_shellset_data, ref_shellset_data + am.sph_size());
                double ref_min = *std::min_element(ref_shellset_data, ref_shellset_data + am.sph_size());
                ref_max = std::max(std::abs(ref_max), std::abs(ref_min));
                if(ref_max > eps)
                {
                    *logger << "Non-zero (max = " << ref_max << ") shellset = " << _tmp 
                            << ", id = " << ref_shellset_id_u64
                            << " in ref without match in libheric computation\n";
                }
                ref_shlset++;
                if(ref_shlset == ref_desc.n_shellset_quartets)break; 
                ref_shellset_data += am.sph_size();
                _tmp = ref_shellset_ids[ref_shlset];
                for(uint64_t i = 0;i < 8;i++)_tmp[i]--;
                ref_shellset_id_u64 = bytes_to_u64(_tmp);
            }
            if(ref_shellset_id_u64 == shellset_id_u64)
            {
                bool mismatch = false;
                for(uint32_t eri = 0;eri < am.sph_size();eri++)
                {
                    int64_t res_round = std::round(shellset_data[eri] / eps);
                    int64_t ref_round = std::round(ref_shellset_data[eri] / eps);
                    uint64_t error = std::abs(res_round - ref_round);
                    if(error > 1)
                    {
                        mismatch = true;
                        break;
                    }
                }
                if(mismatch)
                {
                    *logger << "Mismatch in shellset = " << shellset_ids[shlset] 
                            << ", id = " << shellset_id_u64 << '\n';
                    display_quartet_error<IS_CART>(am, shellset_data, ref_shellset_data, eps, logger);
                }
                ref_shlset++;
                if(ref_shlset >= ref_desc.n_shellset_quartets)continue;
                ref_shellset_data += am.sph_size();
                _tmp = ref_shellset_ids[ref_shlset];
                for(uint64_t i = 0;i < 8;i++)_tmp[i]--;
                ref_shellset_id_u64 = bytes_to_u64(_tmp);
            }
            else
            {
                double comp_max = *std::max_element(shellset_data, shellset_data + am.sph_size());
                double comp_min = *std::min_element(shellset_data, shellset_data + am.sph_size());
                comp_max = std::max(std::abs(comp_max), std::abs(comp_min));
                if(comp_max > eps)
                {
                    *logger << "Non-zero (max = " << comp_max << ") shellset = " << shellset_ids[shlset] 
                            << ", id = " << shellset_id_u64
                            << " in libheric computation without match in ref\n";
                }
            }
        } 

    }
    
    template void compare_atom_quartets<false>(const am_container&, 
                                               const abstract_acc_context::atom_quartet_descriptor&,
                                               const abstract_acc_context::atom_quartet_descriptor&,
                                               double, uint64_t);
    template void compare_atom_quartets<true>(const am_container&, 
                                              const abstract_acc_context::atom_quartet_descriptor&,
                                              const abstract_acc_context::atom_quartet_descriptor&,
                                              double, uint64_t);
    
    std::ostream& operator<<(std::ostream& os, const std::array<uint8_t, 8> shellset)
    {
        os << "(" << uint32_t(shellset[0]) << "{" << uint32_t(shellset[4]) << "}," 
                  << uint32_t(shellset[1]) << "{" << uint32_t(shellset[5]) << "}|" 
                  << uint32_t(shellset[2]) << "{" << uint32_t(shellset[6]) << "}," 
                  << uint32_t(shellset[3]) << "{" << uint32_t(shellset[7]) << "})";
        return os; 
    } 
}
