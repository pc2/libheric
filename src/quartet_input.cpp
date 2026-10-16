#include <quartet_input.h>

#include <algorithm>
#include <cmath>

#include <constants.h>
#include <logging.h>
#include <simd.h>

// Disable libcint dependency until necessary
//extern "C" {
//#include <rys_roots.h>
//}

namespace libheric 
{
    am_container::am_container(uint32_t la, uint32_t lb, uint32_t lc, uint32_t ld) : raw({la, lb, lc, ld}), perm(0) {}
    am_container::am_container(uint32_t enc32) 
        : raw({(enc32 >> 24) & 0xFF, (enc32 >> 16) & 0xFF, 
               (enc32 >> 8) & 0xFF, enc32 & 0xFF}), perm(0) {}

    // Input permutations are executed "left to right", i.e the braket-swap is executed before the bra/ket-swaps
    void am_container::apply_permutation(uint32_t perm)
    {
        if((perm & 0x4) != 0)
        {
            uint32_t tmp = raw[0];
            raw[0] = raw[2];
            raw[2] = tmp;

            tmp = raw[1];
            raw[1] = raw[3];
            raw[3] = tmp;
        }
        if((perm & 0x1) != 0)
        {
            uint32_t tmp = raw[2];
            raw[2] = raw[3];
            raw[3] = tmp;
        }
        if((perm & 0x2) != 0)
        {
            uint32_t tmp = raw[0];
            raw[0] = raw[1];
            raw[1] = tmp;
        }
        this->perm = perm;
    }
    void am_container::reverse()
    {
        uint32_t tmp = raw[0];
        raw[0] = raw[3];
        raw[3] = tmp;
        
        tmp = raw[1];
        raw[1] = raw[2];
        raw[2] = tmp;

        // Reversing the am inverts the permutation and swaps the bra/ket swaps
        // (ab|cd) --p000--> (ab|cd) --rev--> (dc|ba) <=> (ab|cd) --p111--> (dc|ba)
        // (ab|cd) --p001--> (ab|dc) --rev--> (cd|ba) <=> (ab|cd) --p101--> (cd|ba)
        // (ab|cd) --p010--> (ba|cd) --rev--> (dc|ab) <=> (ab|cd) --p110--> (dc|ab)
        // (ab|cd) --p011--> (ba|dc) --rev--> (cd|ab) <=> (ab|cd) --p100--> (cd|ab)
        // (ab|cd) --p100--> (cd|ab) --rev--> (ba|dc) <=> (ab|cd) --p011--> (ba|dc)
        // (ab|cd) --p101--> (cd|ba) --rev--> (ab|dc) <=> (ab|cd) --p001--> (ab|dc)
        // (ab|cd) --p110--> (dc|ab) --rev--> (ba|cd) <=> (ab|cd) --p010--> (ba|cd)
        // (ab|cd) --p111--> (dc|ba) --rev--> (ab|cd) <=> (ab|cd) --p000--> (ab|cd)

        // Change of reverse to permutation can be subtle
        // (dd|ss) --p100--> (ss|dd) --rev--> (dd|ss) // Note that the dd has also swapped, i.e. needs to be transposed
        // --> p011 is the correct permutation
        perm ^= 0x7;
        perm = (perm & 0x4) | ((perm & 0x2) >> 1) | ((perm & 0x1) << 1);
    }
    

    uint32_t am_container::total() const {return raw[0] + raw[1] + raw[2] + raw[3];}
    uint32_t am_container::max()   const {return std::max(std::max(raw[0], raw[1]), std::max(raw[2], raw[3]));}

    uint32_t am_container::a() const {return raw[0];}
    uint32_t am_container::b() const {return raw[1];}
    uint32_t am_container::c() const {return raw[2];}
    uint32_t am_container::d() const {return raw[3];}
    
    uint32_t am_container::nca() const {return nco[raw[0]];}
    uint32_t am_container::ncb() const {return nco[raw[1]];}
    uint32_t am_container::ncc() const {return nco[raw[2]];}
    uint32_t am_container::ncd() const {return nco[raw[3]];}
    uint32_t am_container::ncbra() const {return this->nca() * this->ncb();}
    uint32_t am_container::ncket() const {return this->ncc() * this->ncd();}
    
    uint32_t am_container::nsa() const {return nso[raw[0]];}
    uint32_t am_container::nsb() const {return nso[raw[1]];}
    uint32_t am_container::nsc() const {return nso[raw[2]];}
    uint32_t am_container::nsd() const {return nso[raw[3]];}
    uint32_t am_container::nsbra() const {return this->nsa() * this->nsb();}
    uint32_t am_container::nsket() const {return this->nsc() * this->nsd();}

    uint32_t am_container::get_enc32() const {return ((a() & 0xFF) << 24) 
                                                   | ((b() & 0xFF) << 16)
                                                   | ((c() & 0xFF) << 8)
                                                   | (d() & 0xFF);}
    uint32_t am_container::get_idx(uint32_t l_max) const 
    {
        uint32_t lmp1 = l_max + 1;
        return ((a() * lmp1 + b()) * lmp1 + c()) * lmp1 + d();
    }

    uint32_t am_container::cart_size() const {return nco[raw[0]] * nco[raw[1]] * nco[raw[2]] * nco[raw[3]];}
    uint32_t am_container::sph_size()  const {return nso[raw[0]] * nso[raw[1]] * nso[raw[2]] * nso[raw[3]];}


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

    void am_container::make_canonical()
    {
        uint32_t perm = 0;
        uint32_t _a = a() >= b() ? a() : b();
        uint32_t _c = c() >= d() ? c() : d();
        bool needs_braket_swap = (a() + b() < c() + d()) || ((a() + b() == c() + d()) && (_a > _c));
        if(c() < d()) perm += (needs_braket_swap ? 2 : 1);
        if(a() < b()) perm += (needs_braket_swap ? 1 : 2);
        if(needs_braket_swap)perm += 4;
        this->apply_permutation(perm);
    }
    
    void am_container::make_fpga_canonical()
    {
        uint32_t perm = 0;
        bool needs_braket_swap = nco[a()] * nco[b()] < nco[c()] * nco[d()];
        if(c() < d()) perm += (needs_braket_swap ? 2 : 1);
        if(a() < b()) perm += (needs_braket_swap ? 1 : 2);
        if(needs_braket_swap)perm += 4;
        this->apply_permutation(perm);
    }

    void am_container::make_libint_canonical()
    {
        uint32_t perm = 0;
        bool needs_braket_swap = c() + d() < a() + b();
        if(c() < d()) perm += (needs_braket_swap ? 2 : 1);
        if(a() < b()) perm += (needs_braket_swap ? 1 : 2);
        if(needs_braket_swap)perm += 4;
        this->apply_permutation(perm);
    }

    void am_container::undo_perm()
    {
        uint32_t inv_perm = this->get_inverse_perm();
        this->apply_permutation(inv_perm);
        this->perm = 0;
    }

    uint32_t am_container::get_perm() const {return this->perm;}
    uint32_t am_container::get_inverse_perm() const 
    {
        int32_t inv_perm = this->perm;
        if((this->perm & 0x4) != 0)
            inv_perm = ((this->perm & 0x1) << 1) + ((this->perm & 0x2) >> 1) + (this->perm & 0x4);
        return inv_perm;
    }
    
    bool am_container::operator== (const am_container& other) const
    {
        return this->raw == other.raw;
    }
    
    std::ostream& operator<<(std::ostream& os, const am_container& am)
    {
        os << "a = " << am.a() << ", b = " << am.b() 
           << ", c = " << am.c() << ", d = " << am.d()
           << ", perm = " << am.get_perm();
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

    template<typename FLOAT, int MAX_RTWT>
    void rtwt_container<FLOAT, MAX_RTWT>::generate(int32_t n_rtwt, FLOAT X)
    {
        std::ostream* logger = get_logger();
        *logger << MESSAGE_TYPE::WARNING << "Root generation currently disabled -- setting to NaN!" << '\n';
        std::fill(this->raw, this->raw + 2 * MAX_RTWT, double(NAN)); 
        //if constexpr(std::is_same_v<FLOAT, float>)
        //{
        //    double _rts[MAX_RTWT], _wts[MAX_RTWT];
        //    double _X = double(X);
        //    CINTrys_roots(n_rtwt, _X, &_rts[0], &_wts[0]);
        //    for(int32_t u = 0;u < n_rtwt;u++)
        //    {
        //        this->raw[u] = FLOAT(_rts[u]);
        //        this->raw[MAX_RTWT + u] = FLOAT(_wts[u]);
        //    }
        //}
        //else
        //{
        //    CINTrys_roots(n_rtwt, X, &raw[0], &raw[MAX_RTWT]);
        //}
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
            std::cout << rtwt.raw[1 * MAX_RTWT + u] << (u == MAX_RTWT - 1 ? " ]" : ", "); 
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
        // Input permutations are applied right to left, i.e. the braket swap is executed before the bra/ket swaps 
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
        else if constexpr (PERM == 5)   // (ab|cd) -> (cd|ba)
        {
            CC.store(&raw_q[4 * 0]);
            DD.store(&raw_q[4 * 1]);
            BB.store(&raw_q[4 * 2]);
            AA.store(&raw_q[4 * 3]);
        }
        else if constexpr (PERM == 6)   // (ab|cd) -> (dc|ab)
        {
            DD.store(&raw_q[4 * 0]);
            CC.store(&raw_q[4 * 1]);
            AA.store(&raw_q[4 * 2]);
            BB.store(&raw_q[4 * 3]);
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
        os << "D = [" << pkt.raw[12] << ", " << pkt.raw[13] << ", " << pkt.raw[14] << "], delta = " << pkt.raw[15];
        return os;
    }

    template std::ostream& operator<<<double>(std::ostream&, const quartet_input<double>&);
    template std::ostream& operator<<<float>(std::ostream&, const quartet_input<float>&);
    
    std::ostream& operator<<(std::ostream& os, const quartet_idx& q_idx)
    {
        os << "atom = " << int32_t(q_idx.atom[0]) << ", "<< int32_t(q_idx.atom[1]) << ", "<< int32_t(q_idx.atom[2]) << ", "<< int32_t(q_idx.atom[3]) << " -- ";
        os << "set = " << int32_t(q_idx.set[0]) << ", "<< int32_t(q_idx.set[1]) << ", "<< int32_t(q_idx.set[2]) << ", "<< int32_t(q_idx.set[3]) << " -- ";
        os << "shl = " << int32_t(q_idx.shl[0]) << ", "<< int32_t(q_idx.shl[1]) << ", "<< int32_t(q_idx.shl[2]) << ", "<< int32_t(q_idx.shl[3]) << " -- ";
        os << "exp = " << int32_t(q_idx.exp[0]) << ", "<< int32_t(q_idx.exp[1]) << ", "<< int32_t(q_idx.exp[2]) << ", "<< int32_t(q_idx.exp[3]) << " -- ";
        os << "B_off = " << int32_t(q_idx.off[0]) << ", "<< int32_t(q_idx.off[1]) << ", "<< int32_t(q_idx.off[2]) << " -- ";
        os << "D_off = " << int32_t(q_idx.off[3]) << ", "<< int32_t(q_idx.off[4]) << ", "<< int32_t(q_idx.off[5]) << " -- ";
        os << "Q_off = " << int32_t(q_idx.off[6]) << ", "<< int32_t(q_idx.off[7]) << ", "<< int32_t(q_idx.off[8]) << " -- ";
        os << "diff = " << int32_t(q_idx.diff);
        return os;
    }
    

}
