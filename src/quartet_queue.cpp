#include <quartet_queue.h>

#include <limits>
#include <exception>

namespace libheric
{
    quartet_queue::quartet_queue()
        :
           max_log2_batch_size(quartet_queue::default_log2_batch_size),
           log2_batch_size(quartet_queue::default_log2_batch_size), 
           batch_size(1ull << this->log2_batch_size),
           atom_quartets(new atom_quartet_input[this->capacity()]), curr_atom_quartet(nullptr),
           shellset_quartets(new shellset_quartet_input[this->capacity()]), curr_shellset_quartet(nullptr),
           prim_quartets(new prim_quartet_input[this->capacity()]),      
           wrap_mask(std::numeric_limits<uint64_t>::max() >> (64 - (max_log2_batch_size + 1))),
           atom_read_idx(0), shellset_read_idx(0), prim_read_idx(0),
           atom_write_idx(0), shellset_write_idx(0), prim_write_idx(0)
    {
        if(max_log2_batch_size >= 63)std::cout << "[ERROR] log2_capacity overshoots read/write ptr datatype!\n";
    }

    quartet_queue::~quartet_queue() 
    {
        if(atom_quartets)     delete[] atom_quartets;
        if(shellset_quartets) delete[] shellset_quartets;
        if(prim_quartets)     delete[] prim_quartets;
    }

    void quartet_queue::reset()
    {
        this->atom_read_idx      = 0;
        this->shellset_read_idx  = 0;
        this->prim_read_idx      = 0;
        this->atom_write_idx     = 0;
        this->shellset_write_idx = 0;
        this->prim_write_idx     = 0;
        this->curr_atom_quartet = nullptr;
        this->curr_shellset_quartet = nullptr;
    }

    void quartet_queue::reshape(uint64_t log2_batch_size)
    {
        this->wrap_mask = std::numeric_limits<uint64_t>::max() >> (64 - (log2_batch_size + 1));
        this->batch_size = 1ull << log2_batch_size;
        this->log2_batch_size = log2_batch_size;
        if(log2_batch_size > this->max_log2_batch_size) 
        {
            this->max_log2_batch_size = log2_batch_size;
            if(log2_batch_size >= 63)std::cout << "[ERROR] new log2_capacity overshoots read/write ptr datatype!\n";
            if(atom_quartets)     delete[] atom_quartets;
            if(shellset_quartets) delete[] shellset_quartets;
            if(prim_quartets)     delete[] prim_quartets;
             
            this->atom_quartets     = new atom_quartet_input[this->capacity()];
            this->shellset_quartets = new shellset_quartet_input[this->capacity()];
            this->prim_quartets     = new prim_quartet_input[this->capacity()];
        }
        this->reset();
    }

    uint64_t quartet_queue::size_remaining()
    {
        return this->capacity() - this->size();
    }

    uint64_t quartet_queue::size()
    {
        return this->prim_write_idx >= this->prim_read_idx 
               ? this->prim_write_idx - this->prim_read_idx 
               : this->prim_write_idx + (std::numeric_limits<uint64_t>::max() - this->prim_read_idx) + 1;
    }

    uint64_t quartet_queue::capacity() 
    {
        return this->batch_size * quartet_queue::num_batches;
    }

    uint64_t quartet_queue::mem_usage()
    {
        return (1ull << (this->max_log2_batch_size + 1)) * (sizeof(atom_quartet_input) + sizeof(shellset_quartet_input) + sizeof(prim_quartet_input))  + sizeof(quartet_queue);
    }

    bool quartet_queue::full()
    {
        return size() == this->capacity();
    }

    bool quartet_queue::empty()
    {
        return size() == 0;
    }

    void quartet_queue::push(const std::array<uint32_t, 4>& atom_id)
    {
        this->curr_atom_quartet = this->atom_quartets + (this->atom_write_idx & this->wrap_mask);
        this->curr_atom_quartet->id = atom_id;
        this->atom_write_idx++; // Wrap is automatic and will probably never happen
    }
    
    void quartet_queue::push(uint64_t shellset_id)
    {
        this->curr_shellset_quartet = this->shellset_quartets + (this->shellset_write_idx & this->wrap_mask);
        this->curr_shellset_quartet->id = shellset_id;
        this->curr_atom_quartet->last_shellset_idx = this->shellset_write_idx - this->shellset_read_idx;
        this->shellset_write_idx++; // Wrap is automatic and will probably never happen
    }
    
    void quartet_queue::push(const prim_quartet_input& prim)
    {
        std::memcpy(&(this->prim_quartets[this->prim_write_idx & this->wrap_mask]), &prim, sizeof(quartet_queue::prim_quartet_input));
        this->curr_shellset_quartet->last_prim_idx = this->prim_write_idx - this->prim_read_idx;
        this->prim_write_idx++; // Wrap is automatic and will probably never happen
    }
    
    void quartet_queue::pop_batch(uint64_t n_atom_quartets, uint64_t n_shellset_quartets)
    {
        this->atom_read_idx     += n_atom_quartets;
        this->shellset_read_idx += n_shellset_quartets;
        this->prim_read_idx     += this->batch_size;
    }

    std::optional<quartet_queue::batch_descriptor> quartet_queue::front_batch()
    {
        if(this->size() < this->batch_size)return {};
        return std::optional<quartet_queue::batch_descriptor>({
            .atom_ptr = &this->atom_quartets[this->atom_read_idx & this->wrap_mask], 
            .shellset_ptr = &this->shellset_quartets[this->shellset_read_idx & this->wrap_mask], 
            .prim_ptr = &this->prim_quartets[this->atom_read_idx & this->wrap_mask], 
            .batch_size = this->batch_size
        });
    }

    quartet_queue::batch_descriptor quartet_queue::incomplete_front_batch()
    {
        if(this->size() >= this->batch_size)
        {
            // has_value is guaranteed
            return *this->front_batch();
        }
        else
        {
            return {
                .atom_ptr = &this->atom_quartets[this->atom_read_idx & this->wrap_mask], 
                .shellset_ptr = &this->shellset_quartets[this->shellset_read_idx & this->wrap_mask], 
                .prim_ptr = &this->prim_quartets[this->atom_read_idx & this->wrap_mask], 
                .batch_size = this->size()
            };
        }
    }
}
