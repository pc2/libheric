#ifndef __LIBHERIC_GLOBALS_H__
#include <cstdint>
#include <iostream>

namespace libheric {
namespace globals {
extern uint32_t mpi_rank; // defined in libheric.cpp 
extern std::vector<std::ostream*> loggers; // defined in logging.cpp
}
}
#endif
