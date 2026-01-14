#ifndef __LIBHERIC_SIMD_H__
#define __LIBHERIC_SIMD_H__
#include <immintrin.h>

namespace libheric {
    template<typename FLOAT> 
    struct vec4
    {
        // Not implemented
    };

    template<>
    struct vec4<float>
    {
        vec4(const float* mem_addr){v = _mm_load_ps(mem_addr);}
        vec4(const __m128& vv): v(vv) {};
        void load(float* mem_addr){v = _mm_load_ps(mem_addr);}
        void store(float* mem_addr){_mm_store_ps(mem_addr, v);}
        vec4<float> add(const vec4<float>& other){return {_mm_add_ps(v, other.v)};}

        __m128 v;
    };

    template<>
    struct vec4<double>
    {
        vec4(const double* mem_addr){v = _mm256_load_pd(mem_addr);}
        vec4(const __m256d& vv): v(vv) {};
        void load(double* mem_addr){v = _mm256_load_pd(mem_addr);}
        void store(double* mem_addr){_mm256_store_pd(mem_addr, v);}
        vec4<double> add(const vec4<double>& other){return  {_mm256_add_pd(v, other.v)};}

        __m256d v;
    };

    template<typename FLOAT, typename OTHER_FLOAT> 
    vec4<OTHER_FLOAT> cvt(const vec4<FLOAT>& v)
    {
        static_assert(std::is_same_v<FLOAT, float> || std::is_same_v<FLOAT, double>, "cvt FLOAT instantiated for non-float type");
        static_assert(std::is_same_v<OTHER_FLOAT, float> || std::is_same_v<OTHER_FLOAT, double>, "cvt OTHER_FLOAT instantiated for non-float type");
        if constexpr (std::is_same_v<FLOAT, OTHER_FLOAT>)return v;
        else if constexpr (std::is_same_v<FLOAT, float>) return {_mm256_cvtps_pd(v.v)};
        else                                             return {_mm256_cvtpd_ps(v.v)};
    }
}
#endif
