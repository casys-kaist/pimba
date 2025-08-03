#include <vector>

#include "addr_mapper/addr_mapper.h"
#include "base/base.h"
#include "dram/dram.h"
#include "lax_trace.h"
#include "memory_system/memory_system.h"

namespace Ramulator {

class LinearMapperBase : public IAddrMapper {
   public:
    IDRAM *m_dram = nullptr;

    int m_num_levels = -1;  // How many levels in the hierarchy?
    std::vector<int>
        m_addr_bits;  // How many address bits for each level in the hierarchy?
    Addr_t m_tx_offset = -1;

    int m_col_bits_idx = -1;
    int m_row_bits_idx = -1;

   protected:
    void setup(IFrontEnd *frontend, IMemorySystem *memory_system) {
        m_dram = memory_system->get_ifce<IDRAM>();

        // Populate m_addr_bits vector with the number of address bits for each
        // level in the hierachy
        const auto &count = m_dram->m_organization.count;
        m_num_levels = count.size();
        m_addr_bits.resize(m_num_levels);
        for (size_t level = 0; level < m_addr_bits.size(); level++) {
            m_addr_bits[level] = calc_log2(count[level]);
        }

        // Last (Column) address have the granularity of the prefetch size
        m_addr_bits[m_num_levels - 1] -=
            calc_log2(m_dram->m_internal_prefetch_size);

        int tx_bytes =
            m_dram->m_internal_prefetch_size * m_dram->m_channel_width / 8;
        m_tx_offset = calc_log2(tx_bytes);

        // Determine where are the row and col bits for ChRaBaRoCo and
        // LAXMapper
        try {
            m_row_bits_idx = m_dram->m_levels("row");
        } catch (const std::out_of_range &r) {
            throw std::runtime_error(
                fmt::format("Organization \"row\" not found in the spec, "
                            "cannot use linear mapping!"));
        }

        // Assume column is always the last level
        m_col_bits_idx = m_num_levels - 1;
    }
};

class LAXMapper final : public LinearMapperBase, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, LAXMapper, "LAXMapper",
                                      "LAXMapper");

   public:
    void init() override {};

    void setup(IFrontEnd *frontend, IMemorySystem *memory_system) override {
        LinearMapperBase::setup(frontend, memory_system);
    }

    void apply(Request &req) override {
        req.addr_vec.resize(m_num_levels, -1);
        auto [ch, pch, mode, reg_writes, order, v_count, comps] =
            LAX::addr_decode(req.addr);
        req.addr_vec[m_dram->m_levels("channel")] = ch;
        req.addr_vec[m_dram->m_levels("pseudochannel")] = pch;
        // NOTE: We exploit unused address for meta data
        req.addr_vec[m_dram->m_levels("bankgroup")] = order;
        req.addr_vec[m_dram->m_levels("bank")] =
            (v_count << 19) | (comps << 9) | (reg_writes << 1) | mode;
    }
};

}  // namespace Ramulator
