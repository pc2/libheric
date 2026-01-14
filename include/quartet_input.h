#ifndef __LIBHERIC_QUARTET_INPUT_H__
#define __LIBHERIC_QUARTET_INPUT_H__

#include <iostream>
#include <type_traits>

#include <Eigen/Dense>

namespace libheric
{
    struct am_container
    {
        am_container() = default;
        am_container(int la, int lb, int lc, int ld);
        void apply_permutation(int perm);

        int32_t total() const;
        int32_t max()   const;  

        int32_t a() const;
        int32_t b() const;
        int32_t c() const;
        int32_t d() const;

        int32_t nca() const;
        int32_t ncb() const;
        int32_t ncc() const;
        int32_t ncd() const;
        
        int32_t nsa() const;
        int32_t nsb() const;
        int32_t nsc() const;
        int32_t nsd() const;

        int32_t cart_size() const;
        int32_t sph_size()  const;

        bool is_canonical() const;
        bool is_fpga_canonical() const;
        bool is_libint_canonical() const;
        std::array<int32_t, 4> raw;
    };

    std::ostream& operator<<(std::ostream& os, const am_container& am);

    template<typename FLOAT, int MAX_RTWT = 8>
    struct alignas(32) rtwt_container{
        static_assert(std::is_same_v<FLOAT, float> || std::is_same_v<FLOAT, double>, "quartet_offset instantiated for non-float type");
        static_assert(MAX_RTWT % 4 == 0, "MAX_RTWT is not divisible by 4");
        using OTHER_FLOAT = std::conditional_t<std::is_same_v<FLOAT, float>, double, float>;
        
        FLOAT* roots();  
        FLOAT* weights();
        operator rtwt_container<OTHER_FLOAT>();
    public:
        FLOAT raw[2 * MAX_RTWT];
    };
    
    template<typename FLOAT, int MAX_RTWT>
    std::ostream& operator<<(std::ostream& os, const rtwt_container<FLOAT, MAX_RTWT>& rtwt);

    template<typename FLOAT>
    struct alignas(32) quartet_offset{
        static_assert(std::is_same_v<FLOAT, float> || std::is_same_v<FLOAT, double>, "quartet_offset instantiated for non-float type");
        using EigenVector3_t = Eigen::Vector3<FLOAT>;
        EigenVector3_t& B_off();         
        EigenVector3_t& C_off();         
        EigenVector3_t& D_off();         
        EigenVector3_t& off(int off_pos);
    public:
        FLOAT raw[16] = {};
    };
    
    template<typename FLOAT>
    struct alignas(32) quartet_input
    {
        static_assert(std::is_same_v<FLOAT, float> || std::is_same_v<FLOAT, double>, "quartet_input instantiated for non-float type");
        using OTHER_FLOAT = std::conditional_t<std::is_same_v<FLOAT, float>, double, float>;
        using EigenVector3_t = Eigen::Vector3<FLOAT>;
    public:
        EigenVector3_t& A();
        EigenVector3_t& B();
        EigenVector3_t& C();
        EigenVector3_t& D();
        EigenVector3_t& pos(int quartet_pos);

        FLOAT& alpha();
        FLOAT& beta(); 
        FLOAT& gamma();
        FLOAT& delta();
        FLOAT& exp(int quartet_pos);

        quartet_input<FLOAT>& operator+=(const quartet_offset<FLOAT>& q_off);
        operator quartet_input<OTHER_FLOAT>();

        void apply_permutation(int perm);
    public:
        FLOAT raw[16];
    };

    template<typename FLOAT>
    std::ostream& operator<<(std::ostream& os, const quartet_input<FLOAT>& pkt);
}



#endif
