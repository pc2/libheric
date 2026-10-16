#include <quartet_queue.h>


#include <bitset>
#include <chrono>
#include <random>
#include <thread>

#include <Eigen/Sparse>

int main()
{
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    
    libheric::quartet_queue q;

    //for(uint32_t log2_batch_size = 1; log2_batch_size <= 22; log2_batch_size++)
    //{
    //    std::cout << "Trying log2_batch_size = " << log2_batch_size << '\n';
    //    q.reshape(log2_batch_size);
    //    std::cout << "q.capacity()          = " << uint64_t(q.capacity()) << '\n';
    //    std::cout << "q.mem_usage()         = " << uint64_t(q.mem_usage()) << '\n';
    //    std::cout << "q.size()              = " << uint64_t(q.size()) << '\n';
    //    std::cout << "q.size_remaining()    = " << uint64_t(q.size_remaining()) << '\n';
    //    std::cout << "q.full()              = " << q.full() << '\n';
    //    std::cout << "q.empty()             = " << q.empty() << '\n';
    //    std::cout << "q.atom_read_idx       = " << q.atom_read_idx << '\n';
    //    std::cout << "q.atom_write_idx      = " << q.atom_write_idx << '\n';
    //    std::cout << "q.shellset_read_idx   = " << q.shellset_read_idx << '\n';
    //    std::cout << "q.shellset_write_idx  = " << q.shellset_write_idx << '\n';
    //    std::cout << "q.prim_read_idx       = " << q.prim_read_idx << '\n';
    //    std::cout << "q.prim_write_idx      = " << q.prim_write_idx << '\n';
    //    std::cout << "q.wrap_mask           = " << std::bitset<sizeof(q.wrap_mask)*8>(q.wrap_mask) << '\n';
    //    std::cout << "q.batch_size          = " << q.batch_size << '\n';
    //    std::cout << "q.num_batches         = " << q.num_batches << '\n';
    //    std::cout << '\n';

    //    bool* new_atom = new bool[q.capacity()];
    //    bool* new_shellset = new bool[q.capacity()];
    //    quartet_queue::prim_quartet_input prim_tmpl; 
    //    uint64_t n_atoms_total = 0;
    //    std::vector<uint64_t> stray_shellsets(q.capacity());
    //    for(uint32_t num_trials = 0;num_trials < 10;num_trials++)
    //    {
    //        if(!q.front_batch())
    //        {
    //            std::uniform_int_distribution<uint64_t> prim_dist(1, q.size_remaining());
    //            uint64_t n_prims = prim_dist(gen);
    //            std::uniform_int_distribution<uint64_t> shellset_dist(1, n_prims);
    //            uint64_t n_shellsets = shellset_dist(gen);
    //            std::uniform_int_distribution<uint64_t> atom_dist(1, n_shellsets);
    //            uint64_t n_atoms = atom_dist(gen);
    //            
    //            std::fill(new_atom, new_atom + n_prims, false);
    //            std::fill(new_atom, new_atom + n_atoms, true);
    //            std::random_shuffle(new_atom + 1, new_atom + n_prims); 
    //            
    //            std::fill(new_shellset, new_shellset + n_prims, false);
    //            std::fill(new_shellset, new_shellset + n_shellsets, true);
    //            std::random_shuffle(new_shellset + 1, new_shellset + n_prims);
    //           
    //            stray_shellsets.clear();
    //            for(uint64_t p = 0;p < n_prims;p++)
    //            {
    //                if(!new_atom[p] && new_shellset[p])stray_shellset.push_back(p)
    //            } 
    //            std::random_shuffle(stray_shellset.begin(), stray_shellset.end());
    //            auto strays = stary_shellset.begin();
    //            for(uint64_t p = 0;p < n_prims;p++)
    //            {
    //                if(new_atom[p] && !new_shellset[p])
    //                {
    //                    new_shellsets[p] = true;
    //                    new_shellsets[*strays] = false;
    //                    strays++;
    //                }
    //            }
    //            
    //            
    //            for(uint64_t p = 0;p < n_prims;p++)
    //            {
    //                if(new_atom[p] && !new_shellset[p])std::cout << "[WARNING] new atom without new shellset!\n"; 
    //                if(new_atom[p])q.push({p, p, p, p});
    //                if(new_shellset[p])q.push(p);
    //                q.push(prim_tmpl);
    //            }

    //        }
    //        else
    //        {
    //            auto batch_desc = q.front_batch();
    //            auto curr_atom = batch_desc.atom_ptr;

    //            auto curr_shellset = batch_desc.shellset_ptr;
    //            auto curr_prim = batch_desc.curr_prim;
    //            
    //            uint64_t curr_last = batch_desc.shellset_ptr[curr_atom->last_shellset_idx].last_prim_idx;
    //            uint64_t prim_off = curr_last - q.prim_read_idx;
    //            bool full_atom_present = prim_off < batch_desc.batch_size;
    //            while(full_atom_present)
    //            {

    //                atom_completes_batch = prim_off == batch_desc.batch_size - 1;
    //                n_atoms_used++;
    //                if(!atom_completes_batch)
    //                {
    //                    curr_atom++;
    //                    cufull_atom_present = curr_last - q.prim_read_idx < batch_desc.batch_size;  
    //                }

    //            }

    //        }
    //    }
    //    delete[] shellset_offsets;
    //    delete[] prim_offsets;
    //    //libheric::quartet_idx tmp_idx;
    //    //for(uint32_t num_trials = 0;num_trials < 10;num_trials++)
    //    //{
    //    //    if(q.front_batch() == nullptr)
    //    //    {
    //    //        std::uniform_int_distribution<uint64_t> push_dist(1, q.size_remaining());
    //    //        uint64_t num_push = push_dist(gen);
    //    //        std::cout << "Pushing " << uint64_t(num_push) << " elements\n";
    //    //        for(uint64_t i = 0;i < num_push;i++)q.push(tmp_idx);
    //    //    }
    //    //    else
    //    //    {
    //    //        std::cout << "Popping " << q.batch_size << " elements\n";
    //    //        q.pop_batch();
    //    //    }
    //    //    std::cout << "q.capacity()          = " << uint64_t(q.capacity()) << '\n';
    //    //    std::cout << "q.size()              = " << uint64_t(q.size()) << '\n';
    //    //    std::cout << "q.size_remaining()    = " << uint64_t(q.size_remaining()) << '\n';
    //    //    std::cout << "q.full()              = " << q.full() << '\n';
    //    //    std::cout << "q.empty()             = " << q.empty() << '\n';
    //    //    std::cout << "q.num_batches_ready() = " << q.num_batches_ready() << '\n';
    //    //    std::cout << "q.read_idx            = " << uint64_t(q.read_idx) << '\n';
    //    //    std::cout << "q.write_idx           = " << uint64_t(q.write_idx) << '\n';
    //    //    std::cout << '\n';
    //    //}
    //    //std::cout << std::string(50, '-') << "\n\n";
    //}

    //Eigen::SparseMatrix<double> sparse_mat(100000, 50000);
    //{
    //    std::vector<Eigen::Triplet<double>> trps;
    //    std::uniform_int_distribution<uint32_t> row_idx_dist(0, 99999);
    //    std::uniform_int_distribution<uint32_t> col_idx_dist(0, 49999);
    //    std::uniform_real_distribution<double> coeff_dist(0.0, 1.0);
    //    for(uint32_t i = 0;i < 100000000;i++)
    //    {
    //        uint32_t row_idx = row_idx_dist(gen);
    //        uint32_t col_idx = col_idx_dist(gen);
    //        double  coeff = coeff_dist(gen);
    //        trps.push_back({row_idx, col_idx, coeff });
    //    } 
    //    sparse_mat.setFromTriplets(trps.begin(), trps.end());
    //    sparse_mat.makeCompressed();
    //    std::cout << "sparse_mat.nonZeros() = " << sparse_mat.nonZeros() << '\n';
    //    std::cout << "sparse_mat.innerSize() = " << sparse_mat.innerSize() << '\n';
    //    std::cout << "sparse_mat.outerSize() = " << sparse_mat.outerSize() << '\n';
    //    std::cout << "sizeof(sparse_mat) = " << sizeof(sparse_mat) << '\n';
    //    std::cout << "size(sparse_mat) = " 
    //              << sparse_mat.nonZeros() * (sizeof(double) + sizeof(int)) + sparse_mat.outerSize() * sizeof(int)  << '\n';
    //}

    //using namespace std::chrono_literals;
    //std::this_thread::sleep_for(5min);
    std::cout << "sizeof(atom_quartet_input)     = " << sizeof(libheric::quartet_queue::atom_quartet_input) << '\n';
    std::cout << "sizeof(shellset_quartet_input) = " << sizeof(libheric::quartet_queue::shellset_quartet_input) << '\n';
    std::cout << "sizeof(prim_quartet_input)     = " << sizeof(libheric::quartet_queue::prim_quartet_input) << '\n';
    return 0;
}
