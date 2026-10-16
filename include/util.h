#ifndef __LIBHERIC_UTIL_H__
#define __LIBHERIC_UTIL_H__

#include "quartet_input.h"
#include "abstract_acc.h"

#include <type_traits>

namespace libheric 
{
    template<bool IS_CART>
    void display_quartet_error(const am_container& am, double* quartet, double* quartet_ref, double eps = 1e-6, std::ostream* os = &std::cout);

    template<bool IS_CART>
    void compare_atom_quartets(const am_container& am, 
                               const abstract_acc_context::atom_quartet_descriptor& atom_desc,
                               const abstract_acc_context::atom_quartet_descriptor& ref_desc,
                               double eps, uint64_t data_inc = 0);

    std::ostream& operator<<(std::ostream& os, const std::array<uint8_t, 8> shellset); 

    template<typename PTR>
    inline PTR byte_add(PTR ptr, int64_t num_bytes)
    {
        static_assert(std::is_pointer_v<PTR>);
        uint8_t* byte_ptr = (uint8_t*)ptr;
        return (PTR)(byte_ptr + num_bytes);
    }

	template<typename T, std::enable_if_t<std::is_integral_v<T>>...>
	constexpr T ceil_div(T a,T b) 
	{
		return (a-1)/b + 1;
	}

	template<typename T, std::enable_if_t<std::is_integral_v<T>>...>
	constexpr T pad_to_next_multiple(T val, T multiple)
	{
		return ceil_div(val, multiple) * multiple;
	}

    inline uint64_t bytes_to_u64(const std::array<uint8_t, 8>& bytes)
    {
        return (uint64_t(bytes[0]) << 56)
             | (uint64_t(bytes[1]) << 48)
             | (uint64_t(bytes[2]) << 40)
             | (uint64_t(bytes[3]) << 32)
             | (uint64_t(bytes[4]) << 24)
             | (uint64_t(bytes[5]) << 16)
             | (uint64_t(bytes[6]) <<  8)
             | (uint64_t(bytes[7]) <<  0);
    }

}
#endif
