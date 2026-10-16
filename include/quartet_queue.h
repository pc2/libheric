#ifndef __LIBHERIC_QUARTET_QUEUE_H__
#define __LIBHERIC_QUARTET_QUEUE_H__

#include "quartet_input.h"

#include <utility>
#include <optional>

namespace libheric 
{
    class quartet_queue
    {
    public:
        struct atom_quartet_input
        {
            std::array<uint32_t, 4> id;
            uint64_t last_shellset_idx;
        };
        struct shellset_quartet_input
        {
            union
            {
                // Depends on endianess !
                struct {
                    uint8_t set1;
                    uint8_t set2;
                    uint8_t set3;
                    uint8_t set4;
                    uint8_t shl1;
                    uint8_t shl2;
                    uint8_t shl3;
                    uint8_t shl4;
                };
                uint64_t id;
            };
            uint64_t last_prim_idx;
        };
        struct prim_quartet_input
        {
            uint8_t exp[4];
            int8_t B_off[3];
            int8_t D_off[3];
            int8_t Q_off[3];
            uint8_t diff;
            uint8_t _pad[2];
        };
        struct batch_descriptor
        {
            atom_quartet_input* atom_ptr;
            shellset_quartet_input* shellset_ptr;
            prim_quartet_input* prim_ptr;
            uint64_t batch_size;    
        };
    public:
        quartet_queue();
        ~quartet_queue(); 
        
        void reset();
        void reshape(uint64_t log2_batch_size);
        uint64_t size_remaining();
        uint64_t size();
        uint64_t capacity();
        uint64_t mem_usage();
        bool full();
        bool empty();
        uint64_t num_batches_ready();
        void push(const std::array<uint32_t, 4>& atom_id);
        void push(uint64_t shellset_id);
        void push(const prim_quartet_input& prim_id);
        void pop_batch(uint64_t n_atom_quartets, uint64_t n_shellset_quartets);
        std::optional<batch_descriptor> front_batch();
        batch_descriptor incomplete_front_batch();

    public:
        static constexpr uint64_t default_log2_batch_size = 18; // To fit Xin's eri library
        static constexpr uint64_t num_batches = 2;
        uint64_t max_log2_batch_size;
        uint64_t log2_batch_size;
        uint64_t batch_size;
        atom_quartet_input* atom_quartets, *curr_atom_quartet;
        shellset_quartet_input* shellset_quartets, *curr_shellset_quartet;
        prim_quartet_input* prim_quartets;
        uint64_t wrap_mask;
        
        // assumption max_log2_batch_size is atmost 63
        uint64_t atom_read_idx, shellset_read_idx, prim_read_idx;
        uint64_t atom_write_idx, shellset_write_idx, prim_write_idx;
    };
}
#endif
