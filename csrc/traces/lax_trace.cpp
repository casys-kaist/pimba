#include "lax_trace.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "base/exception.h"
#include "frontend/frontend.h"

namespace Ramulator {

namespace fs = std::filesystem;

class LAXTrace : public IFrontEnd, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, LAXTrace, "LAXTrace",
                                      "LAXTrace")

   private:
    struct Trace {
        int type;
        Addr_t addr;
    };
    std::vector<Trace> m_trace;

    size_t m_trace_length = 0;
    size_t m_curr_trace_idx = 0;

    size_t m_trace_count = 0;

    Logger_t m_logger;

   public:
    size_t m_completed_trace_count = 0;

    void init() override {
        std::string trace_path_str = param<std::string>("path")
                                         .desc("Path to the full trace file.")
                                         .required();
        m_clock_ratio = param<uint>("clock_ratio").required();

        m_logger = Logging::create_logger("LAXTrace");
        m_logger->info("Loading trace file {} ...", trace_path_str);
        init_trace(trace_path_str);
        m_logger->info("Loaded {} lines.", m_trace.size());
    };

    void tick() override {
        bool reg_full = false;
        while (!reg_full && m_trace_count < m_trace_length) {
            const Trace &t = m_trace[m_curr_trace_idx];

            Request r = {t.addr, t.type};
            r.callback = [this](Request &r) {
                this->m_completed_trace_count++;
            };

            bool request_sent = m_memory_system->send(r);
            if (request_sent) {
                m_curr_trace_idx = (m_curr_trace_idx + 1) % m_trace_length;
                m_trace_count++;
            } else {
                reg_full = true;
            }
        }
    };

   private:
    void init_trace(const std::string &file_path_str) {
        fs::path trace_path(file_path_str);
        if (!fs::exists(trace_path)) {
            throw ConfigurationError("Trace {} does not exist!", file_path_str);
        }

        std::ifstream trace_file(trace_path);
        if (!trace_file.is_open()) {
            throw ConfigurationError("Trace {} cannot be opened!",
                                     file_path_str);
        }

        std::unordered_map<int, int> ordering;
        std::unordered_map<int, bool> not_first_command;
        std::string line;
        while (std::getline(trace_file, line)) {
            std::vector<std::string> tokens;
            tokenize(tokens, line, " ");
            int scheduling_mode = 0;
            int reg_writes = 0;
            int v_count = 0;
            int comps = 0;

            int type = -1;
            if (tokens[0] == "refresh_barrier") {
                type = LAX::Request::REFRESH_BARRIER;
                scheduling_mode = std::stoi(tokens[3]);
                reg_writes = std::stoi(tokens[4]);
                v_count = std::stoi(tokens[5]);
                comps = std::stoi(tokens[6]);
            } else if (tokens[0] == "act4_with_reg_20") {
                type = LAX::Request::ACT4_WITH_REG_20;
            } else if (tokens[0] == "act4_with_reg_16") {
                type = LAX::Request::ACT4_WITH_REG_16;
            } else if (tokens[0] == "act4_with_reg_8") {
                type = LAX::Request::ACT4_WITH_REG_8;
            } else if (tokens[0] == "act4_with_reg_4") {
                type = LAX::Request::ACT4_WITH_REG_4;
            } else if (tokens[0] == "act4") {
                type = LAX::Request::ACT4;
            } else if (tokens[0] == "reg_write") {
                type = LAX::Request::REG_WRITE;
            } else if (tokens[0] == "comp") {
                type = LAX::Request::COMP;
            } else if (tokens[0] == "precharge_with_result_8") {
                type = LAX::Request::PRECHARGE_WITH_RESULT_8;
            } else if (tokens[0] == "precharge_with_result_16") {
                type = LAX::Request::PRECHARGE_WITH_RESULT_16;
            } else if (tokens[0] == "precharge") {
                type = LAX::Request::PRECHARGE;
            } else if (tokens[0] == "result") {
                type = LAX::Request::RESULT;
            } else {
                throw ConfigurationError("Trace {} format invalid!",
                                         file_path_str);
            }

            int ch = std::stoi(tokens[1]);
            int pch = std::stoi(tokens[2]);

            int64_t idx = LAX::addr_encode(ch, pch, 0, 0, 0, 0, 0);

            int order = ordering[idx];
            ordering[idx] += 1;

            int64_t addr = LAX::addr_encode(ch, pch, scheduling_mode,
                                            reg_writes, order, v_count, comps);

            m_trace.push_back({type, addr});
        }

        trace_file.close();

        m_trace_length = m_trace.size();
    };

    bool is_finished() override {
        return m_completed_trace_count == m_trace_length;
    };
};

}  // namespace Ramulator

namespace LAX {
int64_t addr_encode(uint64_t ch, uint64_t pch, uint64_t mode,
                    uint64_t reg_writes, uint64_t order, uint64_t v_count,
                    uint64_t comps) {
    int64_t val = 0;
    val |= ((pch & 0x3) << 0);            // 2
    val |= ((ch & 0xFF) << 2);            // 8
    val |= ((order & 0xFFFFFFFF) << 10);  // 32
    val |= ((reg_writes & 0xFF) << 42);   // 8
    val |= ((comps & 0x3FF) << 50);       // 10
    val |= ((mode & 0x1) << 60);          // 1
    val |= ((v_count & 0x3) << 61);       // 2

    return val;
}

std::tuple<int, int, int, int, int, int, int> addr_decode(int64_t addr) {
    int pch = (addr >> 0) & 0x3;
    int ch = (addr >> 2) & 0xFF;
    int order = (addr >> 10) & 0xFFFFFFFF;
    int reg_writes = (addr >> 42) & 0xFF;
    int comps = (addr >> 50) & 0x3FF;
    int mode = (addr >> 60) & 0x1;
    int v_count = (addr >> 61) & 0x3;

    return {ch, pch, mode, reg_writes, order, v_count, comps};
}
}  // namespace LAX
