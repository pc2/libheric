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
        am_container(uint32_t la, uint32_t lb, uint32_t lc, uint32_t ld);
        am_container(uint32_t enc32);
        void apply_permutation(uint32_t perm);
        void reverse();

        uint32_t total() const;
        uint32_t max()   const;  

        uint32_t a() const;
        uint32_t b() const;
        uint32_t c() const;
        uint32_t d() const;

        uint32_t nca() const;
        uint32_t ncb() const;
        uint32_t ncc() const;
        uint32_t ncd() const;
        uint32_t ncbra() const;
        uint32_t ncket() const;
        
        uint32_t nsa() const;
        uint32_t nsb() const;
        uint32_t nsc() const;
        uint32_t nsd() const;
        uint32_t nsbra() const;
        uint32_t nsket() const;
        
        uint32_t get_enc32() const;
        uint32_t get_idx(uint32_t l_max) const;

        uint32_t cart_size() const;
        uint32_t sph_size()  const;

        bool is_canonical() const;
        bool is_fpga_canonical() const;
        bool is_libint_canonical() const;

        void make_canonical();
        void make_fpga_canonical();
        void make_libint_canonical();


        void undo_perm(); 
        uint32_t get_perm() const;

        bool operator== (const am_container& other) const;
        
        std::array<uint32_t, 4> raw;
        uint32_t perm;
    private:
        uint32_t get_inverse_perm() const;
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
        void generate(int32_t n_rtwt, FLOAT X);
    public:
        static constexpr int _MAX_RTWT = MAX_RTWT;
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
    
    struct quartet_idx
    {
        uint32_t atom[4];
        union {
            uint64_t shellset_id;
            struct
            {
                uint8_t set[4];
                uint8_t shl[4];
            };
        };
        uint8_t exp[4];
        int8_t off[3 * 3];
        uint8_t diff;
        uint8_t _pad[2];
    };

    std::ostream& operator<<(std::ostream& os, const quartet_idx& qidx);

}



#endif
