#include "logging.h"
#include "constants.h"
#include "globals.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

#include <omp.h>

namespace libheric
{
std::vector<std::ostream*> loggers;

std::ostream& operator<<(std::ostream& os, const MESSAGE_TYPE& msg_t)
{
    std::ios::fmtflags restore_flags = os.flags();
    os << std::setfill('0');
    switch(msg_t)
    {
    case MESSAGE_TYPE::TIMESTAMP:
        os << "[T=" << std::fixed << std::setprecision(3) << std::setw(15) << omp_get_wtime() << "]\n";
        break;
    case MESSAGE_TYPE::WARNING:
        os << "[WARNING on R" << std::setw(4) << globals::mpi_rank 
           << "T" << std::setw(3) << omp_get_thread_num() << "]";
        break;
    case MESSAGE_TYPE::ERROR:
        os << "[ERROR on R" << std::setw(4) << globals::mpi_rank 
           << "T" << std::setw(3) << omp_get_thread_num() << "]";
        break;
    }
    os << std::setfill(' ');
    os.flags(restore_flags);
    return os;
}

bool initialize_loggers()
{
    uint32_t num_loggers = omp_get_max_threads();
    if(loggers.size() != 0)release_loggers();
    if constexpr(LOG_PRINT)
    {
        std::string libheric_logs_root = ".libheric_logs";
        std::filesystem::create_directories(libheric_logs_root);
        if(!std::filesystem::exists(libheric_logs_root))
        {
           std::cout << MESSAGE_TYPE::ERROR << " Failed to create libheric_logs directory" << std::endl;
           return false;
        }
        std::stringstream ss_logger_base_name;
        ss_logger_base_name << libheric_logs_root << "/R" << std::setfill('0') << std::setw(4) << globals::mpi_rank;
        for(uint32_t i = 0;i < num_loggers;i++)
        {
            std::stringstream ss_logger_name;
            ss_logger_name << ss_logger_base_name.str() 
                           << "_T" << std::setfill('0') << std::setw(3) << i << ".log";
            loggers.push_back(new std::ofstream(ss_logger_name.str()));
            if(!dynamic_cast<std::ofstream*>(loggers.back())->is_open())
            {
                std::cout << MESSAGE_TYPE::ERROR << " Failed to open " << ss_logger_name.str() << std::endl;
                release_loggers();
                return false;
            }
        }
    }
    else
    {
        for(uint32_t i = 0;i < num_loggers;i++)
        {
            loggers.push_back(&std::cout);
        }
    }
    return true;
}

void release_loggers()
{
    if constexpr(LOG_PRINT)
    {
        for(size_t i = 0;i < loggers.size();i++)
        {
            std::ofstream* logger = dynamic_cast<std::ofstream*>(loggers[i]);
            if(logger->is_open() && loggers[i] != &std::cout )
            {
                logger->close();
                delete loggers[i];
            }
        }
        loggers.clear(); 
    }
}


std::ostream* get_logger()
{
    return get_logger(omp_get_thread_num());
}
std::ostream* get_logger(uint32_t thread_id)
{
    if(loggers.size() == 0)
    {
        std::cout << MESSAGE_TYPE::WARNING << " Loggers are not initialized, redirecting to std::cout\n";
        return &std::cout; 
    }
    else
    {
       if(0 <= thread_id && thread_id < omp_get_max_threads()) return loggers[thread_id];
       else return get_logger();
    }
}
}
