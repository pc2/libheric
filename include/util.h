#ifndef __LIBHERIC_UTIL_H__
#define __LIBHERIC_UTIL_H__

#include "quartet_input.h"
namespace libheric 
{
    template<bool IS_CART>
    void display_quartet_error(const am_container& am, double* quartet, double* quartet_ref, double eps = 1e-6);
}
#endif
