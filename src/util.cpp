#include "util.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace libheric 
{
    template<bool IS_CART>
    void display_quartet_error(const am_container& am, double* quartet, double* quartet_ref, double eps)
    {

        int32_t nga = IS_CART ? am.nca() : am.nsa(); 
        int32_t ngb = IS_CART ? am.ncb() : am.nsb(); 
        int32_t ngc = IS_CART ? am.ncc() : am.nsc(); 
        int32_t ngd = IS_CART ? am.ncd() : am.nsd(); 
        int32_t q_size = IS_CART ? am.cart_size() : am.sph_size();

        double ref_max = *std::max_element(quartet_ref, quartet_ref + q_size);
        int32_t ref_max_idx = std::max_element(quartet_ref, quartet_ref + q_size) - quartet_ref;
        std::cout << "ref_max @ a = " << ref_max_idx % nga 
                  << ", b = " << (ref_max_idx/nga) % ngb 
                  << ", c = " << (ref_max_idx/(nga * ngb)) % ngc
                  << ", d = " << (ref_max_idx/(nga * ngb * ngc)) % ngd << '\n';


        double error_norm = ref_max == 0.0 ? 1.0 : (1.0 / std::abs(ref_max));
        double max_error = 0.0;

        for(int32_t q = 0;q < am.sph_size();q++)
        {
            double new_error = std::abs(quartet_ref[q] - quartet[q]) * error_norm;
            max_error = std::max(max_error, new_error);
            new_error = std::abs(quartet_ref[q] - quartet[q]) * error_norm;
        } 

        std::cout << " max_error = " << max_error << ", error_tolerance = " << eps << '\n';
        
        if(max_error > eps)
        {
            for(int32_t dd = 0;dd < ngd;dd++)
            {
                for(int32_t cc = 0;cc < ngc;cc++)
                {
                    std::cout << " d = " << dd << ", c = " << cc << '\n';
                    for(int32_t bb = 0;bb < ngb;bb++)
                    {
                        std::cout << "\tb" << bb << ": ";
                        for(int32_t aa = 0;aa < nga;aa++)
                        {
                            int32_t q = ((dd * ngc + cc) * ngb + bb) * nga + aa;
                            std::cout << std::setw(11) << quartet_ref[q] << "(" 
                                      << std::setw(11) << quartet[q] << ", " 
                                      << std::setw(11) << (std::abs(quartet_ref[q] - quartet[q]) * error_norm) << ")" 
                                      << (aa < nga - 1 ? ", " : "\n");
                        }
                    }
                }
            }
        }
    }

    template void display_quartet_error<true>(const am_container&, double*, double*, double);
    template void display_quartet_error<false>(const am_container&, double*, double*, double);
}
