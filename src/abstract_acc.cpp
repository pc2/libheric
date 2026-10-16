#include <abstract_acc.h>
#include <constants.h>
#include <globals.h>
#include <math_util.h>
#include <util.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>

#include <libint2.h>
#include <omp.h>

namespace libheric
{
    bool operator<(const acc_kind& lhs, const acc_kind& rhs)
    {
        return (lhs.name < rhs.name) || 
               (lhs.name == rhs.name && lhs.type < rhs.type) ||
               (lhs.name == rhs.name && lhs.type == rhs.type && lhs.library < rhs.library);
    }

    std::ostream& operator<<(std::ostream& os, const acc_kind& a)
    {
        os << "(" << a.name << ", " << a.type << ", " << a.library << ")";
        return os;
    }

    abstract_acc_context::abstract_acc_context(const acc_device& acc_dev,
                                               const simulation_box& _sim_box,
                                               uint32_t ctx_id,
                                               uint64_t mem_available,
                                               bool compute_raw_cartesian,
                                               abstract_acc_context* master_ctx)
        : dev(acc_dev),
          sim_box(_sim_box),
          ctx_id(ctx_id),
          logger(get_logger(ctx_id)),
          op(potential_operator::invalid),
          op_param(0.0),
          compute_raw_cartesian(compute_raw_cartesian),
          transform(compute_raw_cartesian),
          staging(),
          cache(
            quartet_staging::mem_pool_size + _sim_box.mem_usage() + transform.mem_usage() > mem_available 
            ? 1 : mem_available - quartet_staging::mem_pool_size - _sim_box.mem_usage() + transform.mem_usage()
          ),
          master_ctx(master_ctx == nullptr ? this : master_ctx)
    {
        //uint32_t num_tasks = DEFAULT_QUEUE_SIZE + (accelerator_queue != nullptr);
        //for(uint32_t i = 0;i < num_tasks;i++)
        //{
        //    quartet_tasks.emplace_back(DEFAULT_TASK_SIZE, &task_consumer_callback, &task_consumer_callback_mtx);
        //}
        this->is_initialized = false;
        if(quartet_staging::mem_pool_size + sim_box.mem_usage() + transform.mem_usage() > mem_available)
        {
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << " mem_available = " << mem_available 
                   << ", but minimum requirement is " 
                   << quartet_staging::mem_pool_size + sim_box.mem_usage() + transform.mem_usage() << '\n';
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
            return;
        }

        if(this->master_ctx == this)
        {
            if constexpr(LOG_PRINT)
            {
                *logger << MESSAGE_TYPE::TIMESTAMP << '\n';
                *logger << std::setprecision(12) << std::scientific;
                *logger << "ctx_id = " << ctx_id << '\n';

                *logger << "Cell_vectors:\n";
                *logger << "e_x = [" << sim_box.cell_mat.col(0).transpose() << "]^T\n";
                *logger << "e_y = [" << sim_box.cell_mat.col(1).transpose() << "]^T\n";
                *logger << "e_z = [" << sim_box.cell_mat.col(2).transpose() << "]^T\n\n";
               
                Eigen::MatrixX4d tmp_pos_kind(sim_box.positions.cols(), 4);
                tmp_pos_kind(Eigen::all, Eigen::seq(0, 2)) = sim_box.positions.transpose();
                for(int32_t i = 0;i < sim_box.natom;i++)
                    tmp_pos_kind(i, 3) = double(sim_box.kinds[i]);

                *logger << "Positions:\n x   y   z   kind\n";
                *logger << tmp_pos_kind << "\n\n";
               
 
                std::stringstream ss_set_str;


                for(int l = 0;l <= sim_box.l_max;l++)
                {
                    *logger << "l = " << l << "(" << spdf[l] << ")\n";
                    for(int kind = 0;kind < sim_box.nkind;kind++)
                    {
                        int lkind = l * sim_box.nkind + kind; 
                        *logger << "   kind = " << kind << ", #set = " << sim_box.kind_nset[lkind] << '\n';
                        for(int set = 0;set < sim_box.kind_nset[lkind];set++)
                        {
                            int lkindset = lkind * sim_box.max_set + set;
                            *logger << "      set = " << set 
                                      << " (#exp = " << sim_box.set_nexp[lkindset] 
                                      << ", #shl = " << sim_box.set_nshl[lkindset] << ")\n";

                            int exp_off = sim_box.exp_offsets[lkindset];
                            int coeff_off = sim_box.coeff_offsets[lkindset];
                            int norm_off = sim_box.norm_offsets[lkindset];
                            int nexp = sim_box.set_nexp[lkindset];
                            int nshl = sim_box.set_nshl[lkindset];

                            *logger << std::setw(9 + 4 + 20/2) << "";
                            for(int shl = 0;shl < nshl;shl++)*logger << std::setw(20) << "shl" << shl;
                            *logger << "\n         norm" << std::setw(20) << "";
                            for(int shl = 0;shl < nshl;shl++)*logger << std::setw(20) << sim_box.normalizations[norm_off + shl]; 
                            *logger << "\n";
                            for(int exp = 0;exp < nexp;exp++)
                            {
                                *logger << "         exp" << exp 
                                          << std::setw(20) << sim_box.exponents[exp_off + exp];
                                for(int shl = 0;shl < nshl;shl++)
                                    *logger << std::setw(20) << sim_box.coefficients[coeff_off + shl * nexp + exp];
                                *logger << '\n';
                            } 
                        }
                        *logger << "   " << std::string(50, '-') << "\n\n";
                    }
                }
                *logger << "sim_box.mem_usage()   = " << sim_box.mem_usage() << '\n';
                *logger << "staging.mem_pool_size = " << staging.mem_pool_size << '\n';
                *logger << "transform.mem_usage() = " << transform.mem_usage() << '\n';
                *logger << "cache.mem_pool_size   = " << cache.mem_pool_size << std::endl;
                *logger << "Eigen::nbThreads() = " << Eigen::nbThreads() << '\n';
            }

            std::filesystem::path ref_dir_path = abstract_acc_context::ref_dir;
            std::optional<uint32_t> tgt_l_max;
            if(std::filesystem::exists(ref_dir_path))
            {
                for(auto& file_in_ref : std::filesystem::directory_iterator{ref_dir_path})
                {
                    if(file_in_ref.path().extension() == ".idx")
                    {
                        if constexpr (LOG_PRINT)
                            *logger << "Found index file: " << file_in_ref << '\n';
                        auto file_in_ref_path = file_in_ref.path();
                        std::ifstream ref_index_file{file_in_ref_path, std::ios::binary};
                        ref_index_file.seekg(0, std::ios::end);
                        uint64_t index_file_size = ref_index_file.tellg();
                        ref_index_file.seekg(0, std::ios::beg);
                        
                        uint32_t file_l_max;
                        std::array<uint32_t, 4> ref_atom_quartet_idx;
                        uint64_t file_offset;
                        uint64_t idx_size = ref_atom_quartet_idx.size() * sizeof(ref_atom_quartet_idx[0]);
                        uint64_t offset_size = sizeof(file_offset);
                        uint64_t entry_size = idx_size + offset_size; 
                        if((index_file_size - sizeof(file_l_max)) % entry_size != 0)
                        {
                            std::stringstream ss_err;
                            ss_err << MESSAGE_TYPE::ERROR << " Index file of size " << index_file_size << " is not properly formated\n";
                            *logger << ss_err.str() << std::flush;
                            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                            return;
                        }
                        
                        file_in_ref_path.replace_extension(".dat");
                        if(!std::filesystem::exists(file_in_ref_path))
                        {
                            std::stringstream ss_err;
                            ss_err << MESSAGE_TYPE::ERROR << " Corresponding data_file does not exist!\n";
                            *logger << ss_err.str() << std::flush;
                            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                            return;
                        }
                        this->ref_data_filenames.push_back(file_in_ref_path);

                        std::ifstream ref_data_file{file_in_ref_path, std::ios::binary};
                        ref_data_file.seekg(0, std::ios::end);
                        uint64_t data_file_size = ref_data_file.tellg();
                        ref_data_file.close();
                        if constexpr (LOG_PRINT)
                            *logger << "data_file_size = " << data_file_size << '\n';

                        ref_index_file.read(reinterpret_cast<char*>(&file_l_max), sizeof(file_l_max));
                        if(tgt_l_max.has_value() && tgt_l_max.value() != file_l_max)
                        {
                            std::stringstream ss_err;
                            ss_err << MESSAGE_TYPE::ERROR << " Got l_max = " << file_l_max << ", but expected l_max = " << tgt_l_max.value() << '\n';
                            *logger << ss_err.str() << std::flush;
                            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                            return;
                        }
                        else tgt_l_max = file_l_max;
                        uint64_t n_entries = (index_file_size - sizeof(file_l_max)) / entry_size;
                        for(uint64_t idx_entry = 0;idx_entry < n_entries; idx_entry++)
                        {
                            ref_index_file.read(reinterpret_cast<char*>(&ref_atom_quartet_idx[0]), idx_size);
                            ref_index_file.read(reinterpret_cast<char*>(&file_offset), offset_size);
                            if constexpr (LOG_PRINT)
                            {
                                *logger << "Index file contains atom quartet (" 
                                        << ref_atom_quartet_idx[0] << ", " << ref_atom_quartet_idx[1] << "|" 
                                        << ref_atom_quartet_idx[2] << ", " << ref_atom_quartet_idx[3] << ") -> "
                                        << " data_file offset = " << file_offset << "\n"; 
                            }
                            if(file_offset >= data_file_size)
                            {
                                std::stringstream ss_err;
                                ss_err << MESSAGE_TYPE::ERROR << " file_offset >= data_file_size = " << data_file_size << '\n';
                                *logger << ss_err.str() << std::flush;
                                if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                                return;
                            }
                            if(this->ref_index.contains(ref_atom_quartet_idx))
                            {
                                std::stringstream ss_err;
                                ss_err << MESSAGE_TYPE::ERROR << " ref_index already contains atom_quartet at file" 
                                       << this->ref_data_filenames[this->ref_index[ref_atom_quartet_idx].first] << "\n";                            
                                *logger << ss_err.str() << std::flush;
                                if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                                return;
                            }
                            this->ref_index[ref_atom_quartet_idx] = {this->ref_data_filenames.size() - 1, file_offset};
                        }
                    }
                }
            }
            this->ref_l_max = tgt_l_max ? *tgt_l_max : 0;
        }
         
        this->is_functional = false;
        this->is_initialized = true;
        this->quartet_class_finalizing = false;
    }

    abstract_acc_context::~abstract_acc_context()
    {
        logger = nullptr;
    }

    //bool abstract_acc_context::quartet_cache::quartet_index::emplace(const abstract_acc_context::quartet_cache::quartet_index::index_type& key, 
    //                                                                 const atom_quartet_descriptor& value)
    //{
    //    if(this->size == this->capacity)return false; // No more space
    //    uint64_t insert_idx = this->size;
    //    if(this->size > 0 && key <= this->ids[insert_idx - 1]) return false; // Key is already in cache or not in sorted order
    //    this->ids[insert_idx] = key;
    //    this->descs[insert_idx] = value;
    //    this->size++;
    //    return true;
    //}

    abstract_acc_context::quartet_cache::quartet_cache(uint64_t cache_size)
        :
        mem_pool_size(pad_to_next_multiple(cache_size, uint64_t(64))),
        mem_pool(std::aligned_alloc(64, this->mem_pool_size)),
        data_ptr(mem_pool),
        index_ptr(((cache_index*)mem_pool) + (this->mem_pool_size / sizeof(cache_index)))
    {
    }
    
    abstract_acc_context::quartet_cache::~quartet_cache()
    {
        if(mem_pool)std::free(mem_pool);
        data_ptr = nullptr;
        index_ptr = nullptr;
    }

    
    void abstract_acc_context::quartet_cache::reset()
    {
        this->data_ptr = mem_pool;
        this->index_ptr = ((cache_index*)mem_pool) + this->mem_pool_size / sizeof(cache_index);
    }

    abstract_acc_context::cache_index* abstract_acc_context::quartet_cache::reserve(const std::array<uint32_t, 4>& idx, uint64_t n_shellset_quartets,
                                                                                    uint64_t num_data_bytes, uint64_t data_align)
    {
        return nullptr;
    }
    //bool abstract_acc_context::quartet_cache::can_reserve(uint64_t num_data_bytes, uint64_t num_desc_bytes)
    //{
    //    return this->size_remaining() >= num_data_bytes + num_desc_bytes;
    //    //void* old_data_ptr = this->data_ptr;
    //    //void* old_shellset_ptr = this->shellset_ptr;
    //    //this->data_ptr = byte_add(this->data_ptr, num_data_bytes);
    //    //this->shellset_ptr = byte_add(this->shellset_ptr, -int64_t(num_desc_byte));
    //    //// Check if there is enough space left for the new atom quartet
    //    //if(this->staging.size_remaining() < num_data_byte + num_desc_bytes)
    //    //{
    //    //    this->data_ptr = old_data_ptr;
    //    //    this->shellset_ptr = old_shellset_ptr;
    //    //    return false;
    //    //}
    //    //return true;
    //}

    //void abstract_acc_context::quartet_cache::commit(const std::array<uint32_t, 5> idx, const atom_quartet_descriptor& q_desc, uint64_t shellset_quartet_size)
    //{
    //    if(!this->index.contains(idx))
    //    {
    //        this->index.emplace(idx, q_desc);
    //        this->data_ptr = byte_add(this->data_ptr, q_desc.n_shellset_quartets * shellset_quartet_size);        
    //        this->shellset_ptr = byte_add(this->shellset_ptr, -int64_t(q_desc.n_shellset_quartets * sizeof(shellset_quartet_descriptor)));        
    //    }
    //}

    int64_t abstract_acc_context::quartet_cache::size_remaining()
    {
        uint8_t* data_byte_ptr = (uint8_t*)this->data_ptr;
        uint8_t* index_byte_ptr = (uint8_t*)this->index_ptr;
        return index_byte_ptr - data_byte_ptr;
    }

    abstract_acc_context::quartet_staging::quartet_staging()
        : mem_pool(std::aligned_alloc(64, abstract_acc_context::quartet_staging::mem_pool_size))
    {

    }

    abstract_acc_context::quartet_staging::~quartet_staging()
    {
        if(mem_pool)std::free(mem_pool);
    }

    bool abstract_acc_context::quartet_staging::reset(const am_container& am, uint64_t max_shellset_quartets)
    {
        std::memset(this->mem_pool, 0, abstract_acc_context::quartet_staging::mem_pool_size);
        uint32_t sph_size = am.sph_size();
        int32_t cart_size = am.cart_size();
        this->am_enc32 = am.get_enc32();
        if(sph_size >= LIBINT2_REALTYPE::size())
        {
            this->eris_allocated_per_shellset = pad_to_next_multiple(cart_size, LIBINT2_REALTYPE::size());
        }
        else
        {
			this->eris_allocated_per_shellset = cart_size;
        }
		uint64_t data_mem_usage = this->eris_allocated_per_shellset * sizeof(double) * max_shellset_quartets;
		uint64_t desc_mem_usage = sizeof(shellset_quartet_descriptor) * max_shellset_quartets;  
		uint64_t atom_quartet_mem_usage = data_mem_usage + desc_mem_usage + sizeof(staging_index);
		this->n_staging = this->mem_pool_size / atom_quartet_mem_usage;
		uint64_t actual_mem_usage;
        void* first_data_ptr;
        std::array<uint8_t, 8>* first_desc_ptr;
		while(this->n_staging != 0)
		{
			actual_mem_usage = 0;
			first_data_ptr = this->mem_pool;
			actual_mem_usage += pad_to_next_multiple(data_mem_usage * this->n_staging, sizeof(uint64_t));
			first_desc_ptr = (std::array<uint8_t, 8>*)byte_add(this->mem_pool, actual_mem_usage);
			actual_mem_usage += pad_to_next_multiple(desc_mem_usage * this->n_staging, sizeof(staging_index));
			this->index_ptr = (staging_index*)byte_add(this->mem_pool, actual_mem_usage);
			actual_mem_usage += this->n_staging * sizeof(staging_index);
			if(actual_mem_usage <= this->mem_pool_size)break;
			else this->n_staging--;
		}
        // Atleast two staging areas needed: one for storing computation and one for transformations
		if(n_staging <= 2) return false;
        //Initialize staging indices
        for(uint64_t slot = 0;slot < this->n_staging;slot++)
        {
            this->index_ptr[slot].value.first_shellset_data_ptr = byte_add(first_data_ptr, data_mem_usage * slot);
            this->index_ptr[slot].value.first_shellset_ptr = (std::array<uint8_t, 8>*)byte_add(first_desc_ptr, desc_mem_usage * slot);
            this->index_ptr[slot].value.n_shellset_quartets = 0;
        }
		
		this->read_idx = 0;
		this->write_idx = 0;
		return true;
    }

    bool abstract_acc_context::quartet_staging::full()
    {
        return modulo((int64_t(this->read_idx) - (int64_t(this->write_idx) + 1)), this->n_staging) == 0;
    }

    bool abstract_acc_context::quartet_staging::empty()
    {
        return this->read_idx == this->write_idx;
    }

    abstract_acc_context::staging_index* abstract_acc_context::quartet_staging::reserve(uint32_t A, uint32_t B, uint32_t C, uint32_t D)
	{
        // Check if there is enough space for performing transformations
        if(this->full())return nullptr;
        write_idx = (write_idx + 1) % this->n_staging;
        staging_index& curr_index = this->index_ptr[this->write_idx];
        // Only clear out as much as necessary
        if(curr_index.value.n_shellset_quartets > 0)
        {
            std::memset(curr_index.value.first_shellset_data_ptr, 0, 
                        curr_index.value.n_shellset_quartets * this->eris_allocated_per_shellset * sizeof(double));
        }
        curr_index.key = {A, B, C, D};
        curr_index.value.n_shellset_quartets = 0;
        curr_index.prims_requested = 0;
        curr_index.prims_computed = 0;
        return &this->index_ptr[this->write_idx];
	}

    abstract_acc_context::staging_index* abstract_acc_context::quartet_staging::get_current_index()
    {
        if(this->read_idx == this->write_idx)return nullptr; //Initial condition with no data
        return &this->index_ptr[this->write_idx];
    }
    
    abstract_acc_context::atom_quartet_descriptor* abstract_acc_context::quartet_staging::get_transform_descriptor()
    {
        return &this->index_ptr[this->read_idx].value;
    }

    abstract_acc_context::staging_index* abstract_acc_context::quartet_staging::get_next_atom_quartet()
    {
        uint64_t new_read_idx = (this->read_idx + 1) % this->n_staging;
        if(!this->empty())
        {
            if(this->index_ptr[new_read_idx].is_finished())
            {
                this->read_idx = new_read_idx;
                return &this->index_ptr[this->read_idx];
            }
            else
            {
                if constexpr (LOG_PRINT)
                {
                    auto logger = get_logger();
                    *logger << "Requested quartet, but requested = " << this->index_ptr[new_read_idx].prims_requested
                            << " and computed = " << this->index_ptr[new_read_idx].prims_computed << '\n';
                }
                return nullptr;
            }
        }
        return nullptr;
    }
    
    std::ostream& operator<<(std::ostream& os, const abstract_acc_context::staging_index& stg_idx)
    {
        os << "{ quartet = (" << stg_idx.key[0] << ", " << stg_idx.key[1]
           << "|" << stg_idx.key[2] << ", " << stg_idx.key[3]
           << "), data_ptr = " << stg_idx.value.first_shellset_data_ptr 
           << ", shellset_ptr = " << stg_idx.value.first_shellset_ptr 
           << ", n_shellset_quartets = " << stg_idx.value.n_shellset_quartets
           << ", eris_allocated_per_shellset = " << stg_idx.eris_allocated_per_shellset << " }";
        return os;
    }



	bool abstract_acc_context::functional(){return this->is_functional;}

    void abstract_acc_context::set_potential_operator(potential_operator op, double op_param)
    {
        this->op = op;
        this->op_param = op_param;
    }

    void abstract_acc_context::set_quartet_class(int la, int lb, int lc, int ld)
    {
        this->is_functional = false;

        if(this->op == potential_operator::invalid)
        {
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << " Invalid potential operator set, forgot to set it?\n";
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
            return;
        }
        if(la < 0 || la > MAX_AM || 
           lb < 0 || lb > MAX_AM || 
           lc < 0 || lc > MAX_AM ||
           ld < 0 || ld > MAX_AM)
        {
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << " Quartet = (" << la << lb << "|" << lc << ld << ") out of bounds -- "
                   << " library only configured for am <= " << MAX_AM << '\n';
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
            return;
        }

        // If constructor fails
        if(!this->is_initialized)
        {
            return;
        }

        // Check if previous quartet class has been consumed
        if(this->quartet_class_finalizing && !this->staging.empty())
        {
            auto atom_quartet = this->staging.get_current_index();
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << " Requested to change quartet class from (" 
                   << this->am.a() << this->am.b() << "|" << this->am.c() << this->am.d() << ") to ("
                   << la << lb << "|" << lc << ld << "), but quartet staging contains atom quartet ("
                   << atom_quartet->key[0] << "," << atom_quartet->key[1] << "|" 
                   << atom_quartet->key[2] << "," << atom_quartet->key[3] << ")\n";
            ss_err << "descriptor = {" 
                   << atom_quartet->value.first_shellset_data_ptr << ", " 
                   << atom_quartet->value.first_shellset_ptr << ", " 
                   << atom_quartet->value.n_shellset_quartets << ", " 
                   << atom_quartet->prims_requested << ", " 
                   << atom_quartet->prims_computed << "}\n";
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
            return;
        }

        std::string abcd = spdf.substr(la, 1) + spdf.substr(lb, 1) + spdf.substr(lc, 1) + spdf.substr(ld, 1);

        this->am = {uint32_t(la), uint32_t(lb), uint32_t(lc), uint32_t(ld)};
        this->total_am = am.total();
        this->abcd = abcd;
        //this->max_input_shellprim_size = sim_box.max_shellprim_quartets(am);
        //
        //if(this->max_input_shellprim_size > max_batch_size)
        //{
        //    std::cout << "[ERROR] max_input_shellprim_size = " << this->max_input_shellprim_size 
        //              << " is too large for the max. quartet_queue size = " << max_batch_size << '\n';
        //    return;
        //}
            
        
        //try
        //{
        //    uint64_t batch_size = 1 << default_log2_batch_size;
        //    if(batch_size < this->max_input_shellprim_size)
        //    {
        //        uint64_t log2_batch_size = uint64_t(std::ceil(std::log2(this->max_input_shellprim_size)));
        //        
        //        std::cout << "[WARNING] Choosing log2_batch_size = " << log2_batch_size 
        //                  << " for max_input_shellprim_size = " << this->max_input_shellprim_size << '\n';
        //        qq.reshape(log2_batch_size, 2);
        //    }
        //    else
        //    {
        //        qq.reshape(default_log2_batch_size, 2);
        //    }
        //}
        //catch(std::exception& e)
        //{
        //    std::cout << e.what() << '\n';
        //    return;
        //}
 

        this->max_shellset_quartets = sim_box.max_shellset_quartets(am);
        this->cart_size = this->am.cart_size();
        this->sph_size = this->am.sph_size();

        if(!this->staging.reset(am, this->max_shellset_quartets))return;

        this->is_functional = true;
        this->quartet_class_finalizing = false;
    }

    void abstract_acc_context::set_atom_ids(int atom_a, int atom_b, int atom_c, int atom_d)
    {
        this->lkind(0) = this->am.a() * this->sim_box.nkind + this->sim_box.kinds[atom_a];
        this->lkind(1) = this->am.b() * this->sim_box.nkind + this->sim_box.kinds[atom_b];
        this->lkind(2) = this->am.c() * this->sim_box.nkind + this->sim_box.kinds[atom_c];
        this->lkind(3) = this->am.d() * this->sim_box.nkind + this->sim_box.kinds[atom_d];
    }
    
    void abstract_acc_context::set_set_ids(int set_a, int set_b, int set_c, int set_d, double pmax)
    {
        this->lkindset(0) = this->lkind(0) * this->sim_box.max_set + set_a;
        this->lkindset(1) = this->lkind(1) * this->sim_box.max_set + set_b;
        this->lkindset(2) = this->lkind(2) * this->sim_box.max_set + set_c;
        this->lkindset(3) = this->lkind(3) * this->sim_box.max_set + set_d;

        this->exp_off(0) = this->sim_box.exp_offsets[this->lkindset(0)];
        this->exp_off(1) = this->sim_box.exp_offsets[this->lkindset(1)];
        this->exp_off(2) = this->sim_box.exp_offsets[this->lkindset(2)];
        this->exp_off(3) = this->sim_box.exp_offsets[this->lkindset(3)];
        
        this->nexp(0) = this->sim_box.set_nexp[this->lkindset(0)];
        this->nexp(1) = this->sim_box.set_nexp[this->lkindset(1)];
        this->nexp(2) = this->sim_box.set_nexp[this->lkindset(2)];
        this->nexp(3) = this->sim_box.set_nexp[this->lkindset(3)];
        
        this->norm_off(0) = this->sim_box.norm_offsets[this->lkindset(0)];
        this->norm_off(1) = this->sim_box.norm_offsets[this->lkindset(1)];
        this->norm_off(2) = this->sim_box.norm_offsets[this->lkindset(2)];
        this->norm_off(3) = this->sim_box.norm_offsets[this->lkindset(3)];
        
        this->coeff_off_base(0) = this->sim_box.coeff_offsets[this->lkindset(0)];
        this->coeff_off_base(1) = this->sim_box.coeff_offsets[this->lkindset(1)];
        this->coeff_off_base(2) = this->sim_box.coeff_offsets[this->lkindset(2)];
        this->coeff_off_base(3) = this->sim_box.coeff_offsets[this->lkindset(3)];

        if(DEBUG_PRINT)
        {
            *logger << "\nsets = (" << set_a << ", " << set_b << ", " << set_c << ", " << set_d << ")" << std::endl;
            *logger << "exp_off = (" << exp_off(0) << ", " << exp_off(1) << ", " << exp_off(2) << ", " << exp_off(3) << ")" << std::endl;
            *logger << "coeff_off_base = (" << coeff_off_base(0) << ", " << coeff_off_base(1) << ", " << coeff_off_base(2) << ", " << coeff_off_base(3) << ")" << std::endl;
            *logger << "norm_off = (" << norm_off(0) << ", " << norm_off(1) << ", " << norm_off(2) << ", " << norm_off(3) << ")" << std::endl;
        }
        this->shellset_id[0] = set_a & 0xFF;
        this->shellset_id[1] = set_b & 0xFF;
        this->shellset_id[2] = set_c & 0xFF;
        this->shellset_id[3] = set_d & 0xFF;
        this->pmax = pmax;
        //this->shellset_tmpl.set1 = set_a;
        //this->shellset_tmpl.set2 = set_b;
        //this->shellset_tmpl.set3 = set_c;
        //this->shellset_tmpl.set4 = set_d;
        //this->quartet_diff = std::min(this->quartet_diff, uint8_t(3u));
    }

    void abstract_acc_context::set_shell_ids(int shl_a, int shl_b, int shl_c, int shl_d)
    {
        this->shellset_id[4] = shl_a & 0xFF;
        this->shellset_id[5] = shl_b & 0xFF;
        this->shellset_id[6] = shl_c & 0xFF;
        this->shellset_id[7] = shl_d & 0xFF;

        this->coeff_off(0) = this->coeff_off_base(0) + shl_a * this->nexp(0);
        this->coeff_off(1) = this->coeff_off_base(1) + shl_b * this->nexp(1);
        this->coeff_off(2) = this->coeff_off_base(2) + shl_c * this->nexp(2);
        this->coeff_off(3) = this->coeff_off_base(3) + shl_d * this->nexp(3);

        //this->shellset_tmpl.shl1 = shl_a;
        //this->shellset_tmpl.shl2 = shl_b;
        //this->shellset_tmpl.shl3 = shl_c;
        //this->shellset_tmpl.shl4 = shl_d;
        //this->quartet_diff = std::min(this->quartet_diff, uint8_t(4u));
    }

    void abstract_acc_context::finish_quartet_class()
    {
        this->quartet_class_finalizing = true;
    }

    abstract_acc_context::staging_index* abstract_acc_context::get_next_staged_atom_quartet()
    {
        *logger << "Request for next staged atom quartet, but none available\n";
        return nullptr;
    }

    bool abstract_acc_context::prefetch_cached_atom_quartet(int A, int B, int C, int D)
    {
        //*logger << "Request for cached atom quartet: A = " << A
        //          << ", B = " << B
        //          << ", C = " << C
        //          << ", D = " << D << std::endl;
        return false;
    }

    void abstract_acc_context::update_positions(Eigen::Matrix3Xd new_positions)
    {
        // Check if previous quartet class has been consumed
        if(!this->staging.empty())
        {
            auto atom_quartet = this->staging.get_current_index();
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << " Requested to change positions, but quartet staging contains atom quartet ("
                   << atom_quartet->key[0] << "," << atom_quartet->key[1] << "|" 
                   << atom_quartet->key[2] << "," << atom_quartet->key[3] << ")\n";
            ss_err << "descriptor = {" 
                   << atom_quartet->value.first_shellset_data_ptr << ", " 
                   << atom_quartet->value.first_shellset_ptr << ", " 
                   << atom_quartet->value.n_shellset_quartets << ", " 
                   << atom_quartet->prims_requested << ", " 
                   << atom_quartet->prims_computed << "}\n";
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
            return;
        }
        this->sim_box.update_positions(new_positions);
    }
    
    void abstract_acc_context::update_cell(Eigen::Matrix3d new_cell_mat)
    {
        // Check if previous quartet class has been consumed
        if(!this->staging.empty())
        {
            auto atom_quartet = this->staging.get_current_index();
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << " Requested to change cell, but quartet staging contains atom quartet ("
                   << atom_quartet->key[0] << "," << atom_quartet->key[1] << "|" 
                   << atom_quartet->key[2] << "," << atom_quartet->key[3] << ")\n";
            ss_err << "descriptor = {" 
                   << atom_quartet->value.first_shellset_data_ptr << ", " 
                   << atom_quartet->value.first_shellset_ptr << ", " 
                   << atom_quartet->value.n_shellset_quartets << ", " 
                   << atom_quartet->prims_requested << ", " 
                   << atom_quartet->prims_computed << "}\n";
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
            return;
        }
        this->sim_box.update_cell(new_cell_mat);
    }


    abstract_acc_context::atom_quartet_descriptor abstract_acc_context::get_reference_atom_quartet(const std::array<uint32_t, 4>& atom_quartet, 
                                                                                                   const am_container& am)
    {
        atom_quartet_descriptor desc{.first_shellset_data_ptr = nullptr, .first_shellset_ptr = nullptr, .n_shellset_quartets = 0};
        if(am.max() > this->master_ctx->ref_l_max)
        {
            *logger << MESSAGE_TYPE::WARNING << "Requested quartet class = " << am << " not in ref because ref only supports l_max = " << this->master_ctx->ref_l_max << "\n";
            return desc;
        }
        if(this->master_ctx->ref_index.contains(atom_quartet))
        {
            auto data_entry  = this->master_ctx->ref_index[atom_quartet];
            auto data_path   = this->master_ctx->ref_data_filenames[data_entry.first];
            auto file_offset = data_entry.second;
            
            std::ifstream data_file{data_path, std::ios::binary};
            data_file.seekg(0, std::ios::end);
            uint64_t data_file_size = data_file.tellg();

            uint64_t shellset_offset;
            uint32_t n_shellset_quartets;
            uint64_t quartet_class_offset = am.get_idx(this->master_ctx->ref_l_max) * (sizeof(shellset_offset) + sizeof(n_shellset_quartets));
            data_file.seekg(file_offset + quartet_class_offset, std::ios::beg);
            data_file.read(reinterpret_cast<char*>(&shellset_offset), sizeof(shellset_offset));
            data_file.read(reinterpret_cast<char*>(&n_shellset_quartets), sizeof(n_shellset_quartets));
            if constexpr(LOG_PRINT)
            {
                *logger << "Found atom_quartet ("
                        << atom_quartet[0] << ", " << atom_quartet[1] << "|" 
                        << atom_quartet[2] << ", " << atom_quartet[3] << "), quartet_class = ("
                        << am.a() << am.b() << "|" << am.c() << am.d() << ")"
                        << " in " << data_path << "\n";
                *logger << " -> shellsets are located at " << shellset_offset 
                        << " and data is located at " << shellset_offset + n_shellset_quartets * 8 * sizeof(uint32_t)
                        << "; in total there are " << n_shellset_quartets << " shellset_quartets present\n";
            }

            uint64_t shellset_data_end = n_shellset_quartets * (8 * sizeof(uint32_t) + am.sph_size() * sizeof(double));
            if(file_offset + shellset_offset + shellset_data_end > data_file_size)
            {
                std::stringstream ss_err;
                ss_err << MESSAGE_TYPE::ERROR << " Reading shellset_data would go out of bounds (file_offset = " << file_offset
                       << ", shellset_offset = " << shellset_offset 
                       << ", shellset_data_end = " << shellset_data_end
                       << "), but data_file_size = " << data_file_size << '\n';
                *logger << ss_err.str() << std::flush;
                if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
                return desc;
            }
         
            std::array<uint8_t, 8>* shellset_ids = new std::array<uint8_t, 8>[n_shellset_quartets];
            double* shellset_data = new double[n_shellset_quartets * am.sph_size()];

            data_file.seekg(file_offset + shellset_offset, std::ios::beg);
            data_file.read(reinterpret_cast<char*>(shellset_ids), n_shellset_quartets * 8 * sizeof(uint8_t));
            data_file.read(reinterpret_cast<char*>(shellset_data), n_shellset_quartets * am.sph_size() * sizeof(double));

            desc.first_shellset_data_ptr = (void*)shellset_data;
            desc.first_shellset_ptr = shellset_ids;
            desc.n_shellset_quartets = n_shellset_quartets;
        }
        else
        {
            std::stringstream ss_err;
            ss_err << MESSAGE_TYPE::ERROR << " did not find atom quartet"
                   << atom_quartet[0] << ", " << atom_quartet[1] << "|" 
                   << atom_quartet[2] << ", " << atom_quartet[3] << ")"
                   << " in reference index\n";
            *logger << ss_err.str() << std::flush;
            if(logger != &std::cout)std::cout << ss_err.str() << std::flush;
        }
        return desc;
    }
}
