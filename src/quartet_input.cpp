#include "quartet_input.h"

#include "constants.h"
#include "simd.h"

namespace libheric 
{
    am_container::am_container(int la, int lb, int lc, int ld) : raw({la, lb, lc, ld}) {}
    void am_container::apply_permutation(int perm)
    {
        if((perm & 0x1) != 0)
        {
            int32_t tmp = raw[2];
            raw[2] = raw[3];
            raw[3] = tmp;
        }
        if((perm & 0x2) != 0)
        {
            int32_t tmp = raw[0];
            raw[0] = raw[1];
            raw[1] = tmp;
        }
        if((perm & 0x4) != 0)
        {
            int32_t tmp = raw[0];
            raw[0] = raw[2];
            raw[2] = tmp;

            tmp = raw[1];
            raw[1] = raw[3];
            raw[3] = tmp;
        }
    }
    int32_t am_container::total() const {return raw[0] + raw[1] + raw[2] + raw[3];}
    int32_t am_container::max()   const {return std::max(std::max(raw[0], raw[1]), std::max(raw[2], raw[3]));}

    int32_t am_container::a() const {return raw[0];}
    int32_t am_container::b() const {return raw[1];}
    int32_t am_container::c() const {return raw[2];}
    int32_t am_container::d() const {return raw[3];}
    
    int32_t am_container::nca() const {return nco[raw[0]];}
    int32_t am_container::ncb() const {return nco[raw[1]];}
    int32_t am_container::ncc() const {return nco[raw[2]];}
    int32_t am_container::ncd() const {return nco[raw[3]];}
    
    int32_t am_container::nsa() const {return nso[raw[0]];}
    int32_t am_container::nsb() const {return nso[raw[1]];}
    int32_t am_container::nsc() const {return nso[raw[2]];}
    int32_t am_container::nsd() const {return nso[raw[3]];}

    int32_t am_container::cart_size() const {return nco[raw[0]] * nco[raw[1]] * nco[raw[2]] * nco[raw[3]];}
    int32_t am_container::sph_size()  const {return nso[raw[0]] * nso[raw[1]] * nso[raw[2]] * nso[raw[3]];}


    bool am_container::is_canonical() const
    {
        return (a() >= b()) && (c() >= d()) 
            && ((a() + b() >= c() + d()) && ((a()+ b() != c() + d()) || (a() <= c()))); 
    }
    bool am_container::is_fpga_canonical() const
    {
        return (a() >= b()) && (c() >= d()) && (nco[a()] * nco[b()] >= nco[c()] * nco[d()]);
    }
    bool am_container::is_libint_canonical() const
    {
        return (a() >= b()) && (c() >= d()) && (c() + d() >= a() + b());
    }
    
    std::ostream& operator<<(std::ostream& os, const am_container& am)
    {
        os << "a = " << am.a() << ", b = " << am.b() << ", c = " << am.c() << ", d = " << am.d() << '\n';
        return os;
    }
    
    template<typename FLOAT, int MAX_RTWT>
    FLOAT* rtwt_container<FLOAT, MAX_RTWT>::roots() {return &raw[0 * MAX_RTWT];}
    
    template<typename FLOAT, int MAX_RTWT>
    FLOAT* rtwt_container<FLOAT, MAX_RTWT>::weights() {return &raw[1 * MAX_RTWT];}
    
    template<typename FLOAT, int MAX_RTWT>
    rtwt_container<FLOAT, MAX_RTWT>::operator rtwt_container<rtwt_container<FLOAT, MAX_RTWT>::OTHER_FLOAT>()
    {
        constexpr int RTWT_PACKETS = MAX_RTWT / 4;
        rtwt_container<OTHER_FLOAT> rtwt_out;
        for(int i = 0;i < 2 * RTWT_PACKETS;i++)
        {
            vec4<FLOAT> _rtwt(&this->raw[4 * i]);
            vec4<OTHER_FLOAT> _rtwt_other = cvt<FLOAT, OTHER_FLOAT>(_rtwt);
            _rtwt_other.store(&rtwt_out.raw[4 * i]);
        }
        return rtwt_out;
    }
    template class rtwt_container<double, 8>;
    template class rtwt_container<float, 8>;
    
    template<typename FLOAT, int MAX_RTWT>
    std::ostream& operator<<(std::ostream& os, const rtwt_container<FLOAT, MAX_RTWT>& rtwt)
    {
        os << "roots = [ ";
        for(uint32_t u = 0;u < MAX_RTWT;u++)
            std::cout << rtwt.raw[0 * MAX_RTWT + u] << (u == MAX_RTWT - 1 ? " ]\n" : ", "); 
        os << "weights = [ ";
        for(uint32_t u = 0;u < MAX_RTWT;u++)
            std::cout << rtwt.raw[1 * MAX_RTWT + u] << (u == MAX_RTWT - 1 ? " ]\n" : ", "); 
        return os;
    }

    template std::ostream& operator<<<double, 8>(std::ostream&, const rtwt_container<double, 8>&);
    template std::ostream& operator<<<float, 8>(std::ostream&, const rtwt_container<float, 8>&);

    template<typename FLOAT> 
    quartet_offset<FLOAT>::EigenVector3_t& quartet_offset<FLOAT>::B_off() 
    {
        return *reinterpret_cast<quartet_offset<FLOAT>::EigenVector3_t*>(&raw[4]);
    }

    template<typename FLOAT> 
    quartet_offset<FLOAT>::EigenVector3_t& quartet_offset<FLOAT>::C_off() 
    {
        return *reinterpret_cast<quartet_offset<FLOAT>::EigenVector3_t*>(&raw[8]);
    }
    
    template<typename FLOAT> 
    quartet_offset<FLOAT>::EigenVector3_t& quartet_offset<FLOAT>::D_off()
    {
        return *reinterpret_cast<quartet_offset<FLOAT>::EigenVector3_t*>(&raw[12]);
    }

    template<typename FLOAT> 
    quartet_offset<FLOAT>::EigenVector3_t& quartet_offset<FLOAT>::off(int off_pos)
    {
        return *reinterpret_cast<quartet_offset<FLOAT>::EigenVector3_t*>(&raw[(off_pos + 1) * 4]);
    }
    
    template class quartet_offset<double>;
    template class quartet_offset<float>;

    template<typename FLOAT>
    quartet_input<FLOAT>::EigenVector3_t& quartet_input<FLOAT>::A()
    {
        return *reinterpret_cast<quartet_input<FLOAT>::EigenVector3_t*>(&raw[0]);
    }

    template<typename FLOAT>
    quartet_input<FLOAT>::EigenVector3_t& quartet_input<FLOAT>::B()
    {
        return *reinterpret_cast<quartet_input<FLOAT>::EigenVector3_t*>(&raw[4]);
    }

    template<typename FLOAT>
    quartet_input<FLOAT>::EigenVector3_t& quartet_input<FLOAT>::C()
    {
        return *reinterpret_cast<quartet_input<FLOAT>::EigenVector3_t*>(&raw[8]);
    }

    template<typename FLOAT>
    quartet_input<FLOAT>::EigenVector3_t& quartet_input<FLOAT>::D()
    {
        return *reinterpret_cast<quartet_input<FLOAT>::EigenVector3_t*>(&raw[12]);
    }

    template<typename FLOAT>
    quartet_input<FLOAT>::EigenVector3_t& quartet_input<FLOAT>::pos(int quartet_pos)
    {
        return *reinterpret_cast<quartet_input<FLOAT>::EigenVector3_t*>(&raw[4 * quartet_pos]);
    }

    template<typename FLOAT>
    FLOAT& quartet_input<FLOAT>::alpha(){return raw[3];}
    template<typename FLOAT>
    FLOAT& quartet_input<FLOAT>::beta(){return raw[7];}
    template<typename FLOAT>
    FLOAT& quartet_input<FLOAT>::gamma(){return raw[11];}
    template<typename FLOAT>
    FLOAT& quartet_input<FLOAT>::delta(){return raw[15];}
    template<typename FLOAT>
    FLOAT& quartet_input<FLOAT>::exp(int quartet_pos) {return raw[4 * quartet_pos + 3];}

    template<typename FLOAT>
    quartet_input<FLOAT>& quartet_input<FLOAT>::operator+=(const quartet_offset<FLOAT>& q_off)
    {
        for(int i = 0;i < 4;i++)
        {
            vec4<FLOAT> _pos(&this->raw[4 * i]);
            vec4<FLOAT> _off(&q_off.raw[4 * i]);
            _pos = _pos.add(_off);
            _pos.store(&this->raw[4 * i]);
        }
        return *this;
    }

    template<typename FLOAT>
    quartet_input<FLOAT>::operator quartet_input<quartet_input::OTHER_FLOAT>()
    {
        quartet_input<OTHER_FLOAT> qi_out;
        for(int i = 0;i < 4;i++)
        {
            vec4<FLOAT> _pos_exp(&this->raw[4 * i]);
            vec4<OTHER_FLOAT> _pos_exp_other = cvt<FLOAT, OTHER_FLOAT>(_pos_exp);
            _pos_exp_other.store(&qi_out.raw[4 * i]);
        }
        return qi_out;
    } 

    template<typename FLOAT, int PERM> 
    void __apply_permutation(FLOAT* raw_q)
    {
        static_assert(0 <= PERM && PERM < 8, "PERM not in [0, 7]");
        
        vec4<FLOAT> AA(&raw_q[4 * 0]);
        vec4<FLOAT> BB(&raw_q[4 * 1]);
        vec4<FLOAT> CC(&raw_q[4 * 2]);
        vec4<FLOAT> DD(&raw_q[4 * 3]);
        
        if constexpr (PERM == 0)return; // (ab|cd) -> (ab|cd)
        else if constexpr (PERM == 1)   // (ab|cd) -> (ab|dc)
        {
            DD.store(&raw_q[4 * 2]);
            CC.store(&raw_q[4 * 3]);
        }
        else if constexpr (PERM == 2)   // (ab|cd) -> (ba|cd)
        {
            BB.store(&raw_q[4 * 0]);
            AA.store(&raw_q[4 * 1]);
        }
        else if constexpr (PERM == 3)   // (ab|cd) -> (ba|dc)
        {
            BB.store(&raw_q[4 * 0]);
            AA.store(&raw_q[4 * 1]);
            DD.store(&raw_q[4 * 2]);
            CC.store(&raw_q[4 * 3]);
        }
        else if constexpr (PERM == 4)   // (ab|cd) -> (cd|ab)
        {
            CC.store(&raw_q[4 * 0]);
            DD.store(&raw_q[4 * 1]);
            AA.store(&raw_q[4 * 2]);
            BB.store(&raw_q[4 * 3]);
        }
        else if constexpr (PERM == 5)   // (ab|cd) -> (dc|ab)
        {
            DD.store(&raw_q[4 * 0]);
            CC.store(&raw_q[4 * 1]);
            AA.store(&raw_q[4 * 2]);
            BB.store(&raw_q[4 * 3]);
        }
        else if constexpr (PERM == 6)   // (ab|cd) -> (cd|ba)
        {
            CC.store(&raw_q[4 * 0]);
            DD.store(&raw_q[4 * 1]);
            BB.store(&raw_q[4 * 2]);
            AA.store(&raw_q[4 * 3]);
        }
        else if constexpr (PERM == 7)   // (ab|cd) -> (dc|ba)
        {
            DD.store(&raw_q[4 * 0]);
            CC.store(&raw_q[4 * 1]);
            BB.store(&raw_q[4 * 2]);
            AA.store(&raw_q[4 * 3]);
        }
    }
    std::array<std::function<void(double*)>, 8> __apply_permutation_double = {
        __apply_permutation<double, 0>, 
        __apply_permutation<double, 1>, 
        __apply_permutation<double, 2>, 
        __apply_permutation<double, 3>, 
        __apply_permutation<double, 4>, 
        __apply_permutation<double, 5>, 
        __apply_permutation<double, 6>, 
        __apply_permutation<double, 7>, 
    };
    std::array<std::function<void(float*)>, 8> __apply_permutation_float = {
        __apply_permutation<float, 0>, 
        __apply_permutation<float, 1>, 
        __apply_permutation<float, 2>, 
        __apply_permutation<float, 3>, 
        __apply_permutation<float, 4>, 
        __apply_permutation<float, 5>, 
        __apply_permutation<float, 6>, 
        __apply_permutation<float, 7>, 
    };

    template<typename FLOAT>
    void quartet_input<FLOAT>::apply_permutation(int perm)
    {
        if constexpr(std::is_same_v<FLOAT, float>) __apply_permutation_float[perm](&this->raw[0]);
        else __apply_permutation_double[perm](&this->raw[0]);
    }

    template class quartet_input<double>;
    template class quartet_input<float>;

    template<typename FLOAT>
    std::ostream& operator<<(std::ostream& os, const quartet_input<FLOAT>& pkt)
    {
        os << "A = [" << pkt.raw[0] << ", " << pkt.raw[1] << ", " << pkt.raw[2] << "], alpha = " << pkt.raw[3] << '\n';
        os << "B = [" << pkt.raw[4] << ", " << pkt.raw[5] << ", " << pkt.raw[6] << "], beta  = " << pkt.raw[7] << '\n';
        os << "C = [" << pkt.raw[8] << ", " << pkt.raw[9] << ", " << pkt.raw[10] << "], gamma = " << pkt.raw[11] << '\n';
        os << "D = [" << pkt.raw[12] << ", " << pkt.raw[13] << ", " << pkt.raw[14] << "], delta = " << pkt.raw[15] << '\n';
        return os;
    }

    template std::ostream& operator<<<double>(std::ostream&, const quartet_input<double>&);
    template std::ostream& operator<<<float>(std::ostream&, const quartet_input<float>&);
    

}
