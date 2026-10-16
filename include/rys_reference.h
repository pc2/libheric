#ifndef __LIBHERIC_RYS_REF_H__ 
#define __LIBHERIC_RYS_REF_H__ 

#include <quartet_input.h>

#include <vector>

namespace libheric
{

    template<typename FLOAT>
    std::vector<FLOAT> rys_quadrature(const am_container& am, 
                                      const quartet_input<double>& input_quartet, 
                                      const rtwt_container<double>& rtwt);
}
#endif
