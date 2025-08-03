#ifndef LAX_TRACE_H
#define LAX_TRACE_H

#include <cstdint>
#include <utility>

#include "base/type.h"

namespace LAX {
enum Request {
    REFRESH_BARRIER = 4,
    ACT4_WITH_REG_20,
    ACT4_WITH_REG_16,
    ACT4_WITH_REG_8,
    ACT4_WITH_REG_4,
    ACT4,
    REG_WRITE,
    COMP,
    PRECHARGE_WITH_RESULT_16,
    PRECHARGE_WITH_RESULT_8,
    PRECHARGE,
    RESULT
};

int64_t addr_encode(uint64_t ch, uint64_t pch, uint64_t mode,
                    uint64_t reg_writes, uint64_t order, uint64_t v_count,
                    uint64_t comps);

std::tuple<int, int, int, int, int, int, int> addr_decode(int64_t addr);

}  // namespace LAX
#endif  // !LAX_TRACES_H
