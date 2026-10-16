#ifndef __LIBHERIC_OPTIMIZE_H__ 
#define __LIBHERIC_OPTIMIZE_H__ 

#include <cstdint>
extern "C"
{
    void optimize_screening_function(double* x_data, double* y_data, int32_t* n_data, double y_cutoff, double* model, int32_t n_model);
}

#endif
