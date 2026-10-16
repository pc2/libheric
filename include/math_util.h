#ifndef __LIBHERIC_MATH_UTIL_H__
#define __LIBHERIC_MATH_UTIL_H__

namespace libheric {
    // Compute x^n for real x and positive integer n
    inline double powern(const double x, const uint32_t n)
    {
        double res = 1.0;
        for (uint32_t i = 0; i < n; i++) res *= x;
        return res;
    }

    inline int64_t factorial(int64_t n)
    {
        if(n < 0) return 0;
        int64_t res = 1;
        for(int64_t i = 1;i <= n;i++) res *= i;
        return res;
    }

    constexpr uint64_t ipow2(uint64_t n)
    {
        uint64_t res = 1;
        for(uint64_t i = 0;i < n;i++)res *= 2;
        return res;
    }
    
    constexpr uint64_t ipow(uint64_t x, uint64_t n)
    {
        uint64_t res = 1;
        for(uint64_t i = 0;i < n;i++)res *= x;
        return res;
    }

    inline int64_t binom(int64_t n, int64_t k)
    {
        if(n < 0 || k < 0 || n < k)return 0;
        if(n == 0 || k == 0 || n == k) return 1;
        return (n * binom(n - 1, k - 1)) / k;
    }

    inline int64_t modulo(int64_t a, int64_t b)
    {
        return ((a % b) + b) % b;
    }

}

#endif
