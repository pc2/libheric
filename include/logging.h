#ifndef __LIBHERIC_LOGGING_H__
#define __LIBHERIC_LOGGING_H__

#include <cstdint>
#include <iostream>
#include <vector>

namespace libheric
{

enum class MESSAGE_TYPE {TIMESTAMP,WARNING,ERROR};
std::ostream& operator<<(std::ostream& os, const MESSAGE_TYPE& msg_t);
bool initialize_loggers();
void release_loggers();
std::ostream* get_logger();
std::ostream* get_logger(uint32_t thread_id);

}

#endif
