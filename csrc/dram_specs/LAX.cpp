#include <algorithm>
#include <exception>
#include <regex>
#include <string>
#include <vector>

#include "dram/dram.h"
#include "dram/lambdas.h"

namespace Ramulator {

class LAX : public IDRAM, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, LAX, "LAX", "LAX")

   public:
    inline static const std::map<std::string, Organization> org_presets = {
        //   name     density   DQ    Ch Pch  Bg Ba   Ro     Co
        {"LAX_2Gb", {2 << 10, 128, {1, 2, 4, 2, 1 << 14, 1 << 6}}},
        {"LAX_4Gb", {4 << 10, 128, {1, 2, 4, 4, 1 << 14, 1 << 6}}},
        {"LAX_8Gb", {8 << 10, 128, {1, 2, 4, 4, 1 << 15, 1 << 6}}},
    };

    inline static const std::map<std::string, std::vector<int>> timing_presets =
        {//   name       rate   nBL  nCL  nRCDRD  nRCDWR  nRP  nRAS  nRC  nWR
         //   nRTPS  nRTPL  nCWL  nCCDS  nCCDL  nRRDS  nRRDL  nWTRS  nWTRL
         //   nRTW  nFAW  nRFC  nRFCSB  nREFI  nREFISB  nRREFD  tCK_ps
         {"default", {2000, 4, 7, 7, 7, 7, 17, 19, 8,   2,    3,  2, 1,
                      2,    2, 3, 3, 4, 3, 15, -1, 160, 3900, -1, 8, 1000}},
         {"dramsim3", {2000, 2, 14, 14, 14, 14, 34, 48,  16,   4,  6, 4,   2, 4,
                       4,    6, 6,  8,  3,  30, -1, 160, 3900, -1, 8, 1000}}};

    /************************************************
     *                Organization
     ***********************************************/
    const int m_internal_prefetch_size = 2;

    inline static constexpr ImplDef m_levels = {
        "channel", "pseudochannel", "bankgroup", "bank", "row", "column",
    };

    /************************************************
     *             Requests & Commands
     ***********************************************/
    inline static constexpr ImplDef m_commands = {
        "ACT",
        "PRE",
        "PREA",
        "RD",
        "WR",
        "RDA",
        "WRA",
        "REFab",
        "REFsb",
        "REFRESH_BARRIER",
        "ACT4_WITH_REG_20",
        "ACT4_WITH_REG_16",
        "ACT4_WITH_REG_8",
        "ACT4_WITH_REG_4",
        "ACT4",
        "REG_WRITE",
        "COMP",
        "PRECHARGE_WITH_RESULT_16",
        "PRECHARGE_WITH_RESULT_8",
        "PRECHARGE",
        "RESULT",
    };

    inline static const ImplLUT m_command_scopes =
        LUT(m_commands, m_levels,
            {
                {"ACT", "row"},
                {"PRE", "bank"},
                {"PREA", "channel"},
                {"RD", "column"},
                {"WR", "column"},
                {"RDA", "column"},
                {"WRA", "column"},
                {"REFab", "channel"},
                {"REFsb", "bank"},
                {"REFRESH_BARRIER", "pseudochannel"},
                {"ACT4_WITH_REG_20", "pseudochannel"},
                {"ACT4_WITH_REG_16", "pseudochannel"},
                {"ACT4_WITH_REG_8", "pseudochannel"},
                {"ACT4_WITH_REG_4", "pseudochannel"},
                {"ACT4", "pseudochannel"},
                {"REG_WRITE", "pseudochannel"},
                {"COMP", "pseudochannel"},
                {"PRECHARGE_WITH_RESULT_16", "pseudochannel"},
                {"PRECHARGE_WITH_RESULT_8", "pseudochannel"},
                {"PRECHARGE", "pseudochannel"},
                {"RESULT", "pseudochannel"},
            });

    inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta>(
        m_commands,
        {
            // open?   close?   access?  refresh?
            {"ACT", {true, false, false, false}},
            {"PRE", {false, true, false, false}},
            {"PREA", {false, true, false, false}},
            {"RD", {false, false, true, false}},
            {"WR", {false, false, true, false}},
            {"RDA", {false, true, true, false}},
            {"WRA", {false, true, true, false}},
            {"REFab", {false, false, false, true}},
            {"REFsb", {false, false, false, true}},
            // NOTE: Set all meta to 'false' to avoid undesired
            // output. We managed all behaviors manually
            {"REFRESH_BARRIER", {false, false, false, false}},
            {"ACT4_WITH_REG_20", {false, false, false, false}},
            {"ACT4_WITH_REG_16", {false, false, false, false}},
            {"ACT4_WITH_REG_8", {false, false, false, false}},
            {"ACT4_WITH_REG_4", {false, false, false, false}},
            {"ACT4", {false, false, false, false}},
            {"REG_WRITE", {false, false, false, false}},
            {"COMP", {false, false, false, false}},
            {"PRECHARGE_WITH_RESULT_16", {false, false, false, false}},
            {"PRECHARGE_WITH_RESULT_8", {false, false, false, false}},
            {"PRECHARGE", {false, false, false, false}},
            {"RESULT", {false, false, false, false}},
        });

    inline static constexpr ImplDef m_requests = {
        "read",
        "write",
        "all-bank-refresh",
        "per-bank-refresh",
        "refresh_barrier",
        "act4_with_reg_20",
        "act4_with_reg_16",
        "act4_with_reg_8",
        "act4_with_reg_4",
        "act4",
        "reg_write",
        "comp",
        "precharge_with_result_16",
        "precharge_with_result_8",
        "precharge",
        "result",
    };

    inline static const ImplLUT m_request_translations =
        LUT(m_requests, m_commands,
            {
                {"read", "RD"},
                {"write", "WR"},
                {"all-bank-refresh", "REFab"},
                {"per-bank-refresh", "REFsb"},
                {"refresh_barrier", "REFRESH_BARRIER"},
                {"act4_with_reg_20", "ACT4_WITH_REG_20"},
                {"act4_with_reg_16", "ACT4_WITH_REG_16"},
                {"act4_with_reg_8", "ACT4_WITH_REG_8"},
                {"act4_with_reg_4", "ACT4_WITH_REG_4"},
                {"act4", "ACT4"},
                {"reg_write", "REG_WRITE"},
                {"comp", "COMP"},
                {"precharge_with_result_16", "PRECHARGE_WITH_RESULT_16"},
                {"precharge_with_result_8", "PRECHARGE_WITH_RESULT_8"},
                {"precharge", "PRECHARGE"},
                {"result", "RESULT"},
            });

    /************************************************
     *                   Timing
     ***********************************************/
    inline static constexpr ImplDef m_timings = {
        "rate",   "nBL",   "nCL",     "nRCDRD", "nRCDWR", "nRP",   "nRAS",
        "nRC",    "nWR",   "nRTPS",   "nRTPL",  "nCWL",   "nCCDS", "nCCDL",
        "nRRDS",  "nRRDL", "nWTRS",   "nWTRL",  "nRTW",   "nFAW",  "nRFC",
        "nRFCSB", "nREFI", "nREFISB", "nRREFD", "tCK_ps"};

    /************************************************
     *                 Node States
     ***********************************************/
    inline static constexpr ImplDef m_states = {"Opened", "Closed", "N/A"};

    inline static const ImplLUT m_init_states =
        LUT(m_levels, m_states,
            {
                {"channel", "N/A"},
                {"pseudochannel", "N/A"},
                {"bankgroup", "N/A"},
                {"bank", "Closed"},
                {"row", "Closed"},
                {"column", "N/A"},
            });

   public:
    struct Node : public DRAMNodeBase<LAX> {
        Node(LAX *dram, Node *parent, int level, int id)
            : DRAMNodeBase<LAX>(dram, parent, level, id) {};
    };
    std::vector<Node *> m_channels;
    std::vector<std::vector<int>> m_order_map;

    FuncMatrix<ActionFunc_t<Node>> m_actions;
    FuncMatrix<PreqFunc_t<Node>> m_preqs;
    FuncMatrix<RowhitFunc_t<Node>> m_rowhits;
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;

    uint32_t m_num_act = 0;
    uint32_t m_num_rdwr = 0;
    uint32_t m_num_comp = 0;

   public:
    void tick() override { m_clk++; };

    void init() override {
        RAMULATOR_DECLARE_SPECS();
        set_organization();
        set_timing_vals();

        set_actions();
        set_preqs();
        set_rowhits();
        set_rowopens();

        create_nodes();

        size_t chs = m_organization.count[m_levels("channel")];
        size_t pchs = m_organization.count[m_levels("pseudochannel")];

        m_order_map = decltype(m_order_map){chs, std::vector<int>(pchs, -1)};

        // custom stats
        register_stat(m_num_act).name("num_act");
        register_stat(m_num_rdwr).name("num_rdwr");
        register_stat(m_num_comp).name("num_comp");
    };

    void issue_command(int command, const AddrVec_t &addr_vec) override {
        int channel_id = addr_vec[m_levels["channel"]];

        if (command != m_commands("RD") && command != m_commands("WR") &&
            command != m_commands("REFab") && command != m_commands("REFsb")) {
            int pch_id = addr_vec[m_levels["pseudochannel"]];
            int order = addr_vec[m_levels["bankgroup"]];

            m_order_map[channel_id][pch_id] = order;
        }

        // update stats
        std::string command_name{m_commands(command)};

        // act
        if (command_name.find("ACT4") != std::string::npos) {
            m_num_act += 4;
        }

        // comp
        if (command_name.find("COMP") != std::string::npos) {
            m_num_comp += 1;
        }

        // rdwr
        std::regex reg(R"(.*REG_(.*))");
        std::smatch reg_match;
        if (std::regex_match(command_name, reg_match, reg)) {
            std::string reg_num = reg_match[1];
            try {
                m_num_rdwr += std::stoi(reg_num);
            } catch (const std::exception &e) {
                m_num_rdwr += 1;
            }
        }
        std::regex result(R"(.*RESULT_(.*))");
        std::smatch result_match;
        if (std::regex_match(command_name, result_match, result)) {
            std::string reg_num = result_match[1];
            m_num_rdwr += std::stoi(reg_num) * 8;
        }
        if (command_name.find("RESULT") != std::string::npos) {
            m_num_rdwr += 1;
        }

        m_channels[channel_id]->update_timing(command, addr_vec, m_clk);
        m_channels[channel_id]->update_states(command, addr_vec, m_clk);
    };

    int get_preq_command(int command, const AddrVec_t &addr_vec) override {
        int channel_id = addr_vec[m_levels["channel"]];
        return m_channels[channel_id]->get_preq_command(command, addr_vec,
                                                        m_clk);
    };

    bool check_ready(int command, const AddrVec_t &addr_vec) override {
        int channel_id = addr_vec[m_levels["channel"]];
        bool ready =
            m_channels[channel_id]->check_ready(command, addr_vec, m_clk);

        if (command != m_commands("RD") && command != m_commands("WR") &&
            command != m_commands("REFab") && command != m_commands("REFsb")) {
            int pch_id = addr_vec[m_levels["pseudochannel"]];
            int order = addr_vec[m_levels["bankgroup"]];

            ready = ready && (m_order_map[channel_id][pch_id] + 1 == order);

            if (command == m_commands("REFRESH_BARRIER")) {
#define V(timing) (m_timing_vals(timing))
                int meta = addr_vec[m_levels["bank"]];
                int mode = meta & 0x1;
                int reg_writes = (meta >> 1) & 0xFF;
                int comps = (meta >> 9) & 0x3FF;
                int v_count = (meta >> 19) & 0x3;
                int latency = 0;
                int col_latency = std::max(V("nBL"), V("nCCDL"));
                int entire_read_latency = v_count * 8 * col_latency;

                if (mode) {
                    int entire_reg_write_latency =
                        1 + ((reg_writes + 3) / 4) * col_latency;

                    // act to act
                    latency += 3 * std::max({V("nRRDS"), V("nFAW"),
                                             entire_reg_write_latency});

                    // last act
                    latency += std::max(
                        {V("nRCDRD"), V("nRCDWR"), entire_reg_write_latency});

                    // precharge
                    latency += std::max(V("nRP"), entire_read_latency);
                } else {
                    // act to act
                    latency += 4 * std::max({V("nRRDS"), V("nFAW")});
                    latency += reg_writes * col_latency;
                    latency += V("nRP");
                    latency += entire_read_latency;
                }

                // comp
                latency += comps * V("nCCDL");

                // comp to precharge
                latency += std::max(V("nRTPL"), V("nWR"));

                int refi = V("nREFI");
                ready = ready && ((m_clk % refi) + latency < refi);
#undef V
            }
        }

        return ready;
    };

    bool check_rowbuffer_hit(int command, const AddrVec_t &addr_vec) override {
        int channel_id = addr_vec[m_levels["channel"]];
        return m_channels[channel_id]->check_rowbuffer_hit(command, addr_vec,
                                                           m_clk);
    };

   private:
    void set_organization() {
        // Channel width
        m_channel_width =
            param_group("org").param<int>("channel_width").default_val(64);

        // Organization
        m_organization.count.resize(m_levels.size(), -1);

        // Load organization preset if provided
        if (auto preset_name =
                param_group("org").param<std::string>("preset").optional()) {
            if (org_presets.count(*preset_name) > 0) {
                m_organization = org_presets.at(*preset_name);
            } else {
                throw ConfigurationError(
                    "Unrecognized organization preset \"{}\" in {}!",
                    *preset_name, get_name());
            }
        }

        // Override the preset with any provided settings
        if (auto dq = param_group("org").param<int>("dq").optional()) {
            m_organization.dq = *dq;
        }

        for (int i = 0; i < m_levels.size(); i++) {
            auto level_name = m_levels(i);
            if (auto sz =
                    param_group("org").param<int>(level_name).optional()) {
                m_organization.count[i] = *sz;
            }
        }

        if (auto density =
                param_group("org").param<int>("density").optional()) {
            m_organization.density = *density;
        }

        // Sanity check: is the calculated channel density the same as the
        // provided one?
        size_t _density =
            size_t(m_organization.count[m_levels["pseudochannel"]]) *
            size_t(m_organization.count[m_levels["bankgroup"]]) *
            size_t(m_organization.count[m_levels["bank"]]) *
            size_t(m_organization.count[m_levels["row"]]) *
            size_t(m_organization.count[m_levels["column"]]) *
            size_t(m_organization.dq);
        _density >>= 20;
        if (m_organization.density != _density) {
            throw ConfigurationError(
                "Calculated {} channel density {} Mb does not equal the "
                "provided density {} Mb!",
                get_name(), _density, m_organization.density);
        }
    };

    void set_timing_vals() {
        m_timing_vals.resize(m_timings.size(), -1);

        // Load timing preset if provided
        bool preset_provided = false;
        if (auto preset_name =
                param_group("timing").param<std::string>("preset").optional()) {
            if (timing_presets.count(*preset_name) > 0) {
                m_timing_vals = timing_presets.at(*preset_name);
                preset_provided = true;
            } else {
                throw ConfigurationError(
                    "Unrecognized timing preset \"{}\" in {}!", *preset_name,
                    get_name());
            }
        }

        // Check for rate (in MT/s), and if provided, calculate and set tCK (in
        // picosecond)
        if (auto dq = param_group("timing").param<int>("rate").optional()) {
            if (preset_provided) {
                throw ConfigurationError(
                    "Cannot change the transfer rate of {} when using a speed "
                    "preset !",
                    get_name());
            }
            m_timing_vals("rate") = *dq;
        }
        int tCK_ps = 1E6 / (m_timing_vals("rate") / 2);
        m_timing_vals("tCK_ps") = tCK_ps;

        // Refresh timings
        // tRFC table (unit is nanosecond!)
        constexpr int tRFC_TABLE[1][4] = {
            //  2Gb   4Gb   8Gb  16Gb
            {160, 260, 350, 450},
        };

        // tRFC table (unit is nanosecond!)
        constexpr int tREFISB_TABLE[1][4] = {
            //  2Gb    4Gb    8Gb    16Gb
            {4875, 4875, 2438, 2438},
        };

        int density_id = [](int density_Mb) -> int {
            switch (density_Mb) {
                case 2048:
                    return 0;
                case 4096:
                    return 1;
                case 8192:
                    return 2;
                case 16384:
                    return 3;
                default:
                    return -1;
            }
        }(m_organization.density);

        m_timing_vals("nRFC") =
            JEDEC_rounding(tRFC_TABLE[0][density_id], tCK_ps);
        m_timing_vals("nREFISB") =
            JEDEC_rounding(tRFC_TABLE[0][density_id], tCK_ps);

        // Overwrite timing parameters with any user-provided value
        // Rate and tCK should not be overwritten
        for (int i = 1; i < m_timings.size() - 1; i++) {
            auto timing_name = std::string(m_timings(i));

            if (auto provided_timing =
                    param_group("timing").param<int>(timing_name).optional()) {
                // Check if the user specifies in the number of cycles (e.g.,
                // nRCD)
                m_timing_vals(i) = *provided_timing;
            } else if (auto provided_timing =
                           param_group("timing")
                               .param<float>(timing_name.replace(0, 1, "t"))
                               .optional()) {
                // Check if the user specifies in nanoseconds (e.g., tRCD)
                m_timing_vals(i) = JEDEC_rounding(*provided_timing, tCK_ps);
            }
        }

        // Check if there is any uninitialized timings
        for (int i = 0; i < m_timing_vals.size(); i++) {
            if (m_timing_vals(i) == -1) {
                throw ConfigurationError(
                    "In \"{}\", timing {} is not specified!", get_name(),
                    m_timings(i));
            }
        }

        // Set read latency
        m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL");

// Populate the timing constraints
#define V(timing) (m_timing_vals(timing))
        populate_timingcons(
            this,
            {
                /*** Channel ***/
                /// 2-cycle ACT command (for row commands)
                {.level = "channel",
                 .preceding = {"ACT"},
                 .following = {"ACT", "PRE", "PREA", "REFab", "REFsb"},
                 .latency = 2},

                /*** Pseudo Channel (Table 3 — Array Access Timings Counted
                   Individually Per Pseudo Channel, JESD-235C) ***/
                // RAS <-> RAS
                {.level = "pseudochannel",
                 .preceding = {"ACT"},
                 .following = {"ACT"},
                 .latency = V("nRRDS")},
                /// 4-activation window restriction
                {.level = "pseudochannel",
                 .preceding = {"ACT"},
                 .following = {"ACT"},
                 .latency = V("nFAW"),
                 .window = 4},

                /// ACT actually happens on the 2-nd cycle of ACT, so +1
                /// cycle to nRRD
                {.level = "pseudochannel",
                 .preceding = {"ACT"},
                 .following = {"REFsb"},
                 .latency = V("nRRDS") + 1},
                /// nRREFD is the latency between REFsb <-> REFsb to
                /// *different* banks
                {.level = "pseudochannel",
                 .preceding = {"REFsb"},
                 .following = {"REFsb"},
                 .latency = V("nRREFD")},
                /// nRREFD is the latency between REFsb <-> ACT to
                /// *different* banks. -1 as ACT happens on its 2nd cycle
                {.level = "pseudochannel",
                 .preceding = {"REFsb"},
                 .following = {"ACT"},
                 .latency = V("nRREFD") - 1},

                // CAS <-> CAS
                /// Data bus occupancy
                {.level = "pseudochannel",
                 .preceding = {"RD", "RDA"},
                 .following = {"RD", "RDA"},
                 .latency = V("nBL")},
                {.level = "pseudochannel",
                 .preceding = {"WR", "WRA"},
                 .following = {"WR", "WRA"},
                 .latency = V("nBL")},

                // CAS <-> CAS
                /// nCCDS is the minimal latency for column commands
                {.level = "pseudochannel",
                 .preceding = {"RD", "RDA"},
                 .following = {"RD", "RDA"},
                 .latency = V("nCCDS")},
                {.level = "pseudochannel",
                 .preceding = {"WR", "WRA"},
                 .following = {"WR", "WRA"},
                 .latency = V("nCCDS")},
                /// RD <-> WR, Minimum Read to Write, Assuming tWPRE = 1 tCK
                {.level = "pseudochannel",
                 .preceding = {"RD", "RDA"},
                 .following = {"WR", "WRA"},
                 .latency = V("nCL") + V("nBL") + 2 - V("nCWL")},
                /// WR <-> RD, Minimum Read after Write
                {.level = "pseudochannel",
                 .preceding = {"WR", "WRA"},
                 .following = {"RD", "RDA"},
                 .latency = V("nCWL") + V("nBL") + V("nWTRS")},
                /// CAS <-> PREab
                {.level = "pseudochannel",
                 .preceding = {"RD"},
                 .following = {"PREA"},
                 .latency = V("nRTPS")},
                {.level = "pseudochannel",
                 .preceding = {"WR"},
                 .following = {"PREA"},
                 .latency = V("nCWL") + V("nBL") + V("nWR")},
                /// RAS <-> RAS
                {.level = "pseudochannel",
                 .preceding = {"ACT"},
                 .following = {"ACT"},
                 .latency = V("nRRDS")},
                {.level = "pseudochannel",
                 .preceding = {"ACT"},
                 .following = {"ACT"},
                 .latency = V("nFAW"),
                 .window = 4},
                {.level = "pseudochannel",
                 .preceding = {"ACT"},
                 .following = {"PREA"},
                 .latency = V("nRAS")},
                {.level = "pseudochannel",
                 .preceding = {"PREA"},
                 .following = {"ACT"},
                 .latency = V("nRP")},
                /// RAS <-> REF
                {.level = "pseudochannel",
                 .preceding = {"ACT"},
                 .following = {"REFab"},
                 .latency = V("nRC")},
                {.level = "pseudochannel",
                 .preceding = {"PRE", "PREA"},
                 .following = {"REFab"},
                 .latency = V("nRP")},
                {.level = "pseudochannel",
                 .preceding = {"RDA"},
                 .following = {"REFab"},
                 .latency = V("nRP") + V("nRTPS")},
                {.level = "pseudochannel",
                 .preceding = {"WRA"},
                 .following = {"REFab"},
                 .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},
                {.level = "pseudochannel",
                 .preceding = {"REFab"},
                 .following = {"ACT", "REFsb"},
                 .latency = V("nRFC")},

                /*** Same Bank Group ***/
                /// CAS <-> CAS
                {.level = "bankgroup",
                 .preceding = {"RD", "RDA"},
                 .following = {"RD", "RDA"},
                 .latency = V("nCCDL")},
                {.level = "bankgroup",
                 .preceding = {"WR", "WRA"},
                 .following = {"WR", "WRA"},
                 .latency = V("nCCDL")},
                {.level = "bankgroup",
                 .preceding = {"WR", "WRA"},
                 .following = {"RD", "RDA"},
                 .latency = V("nCWL") + V("nBL") + V("nWTRL")},
                /// RAS <-> RAS
                {.level = "bankgroup",
                 .preceding = {"ACT"},
                 .following = {"ACT"},
                 .latency = V("nRRDL")},
                {.level = "bankgroup",
                 .preceding = {"ACT"},
                 .following = {"REFsb"},
                 .latency = V("nRRDL") + 1},
                {.level = "bankgroup",
                 .preceding = {"REFsb"},
                 .following = {"ACT"},
                 .latency = V("nRRDL") - 1},

                {.level = "bank",
                 .preceding = {"RD"},
                 .following = {"PRE"},
                 .latency = V("nRTPS")},

                /*** Bank ***/
                {.level = "bank",
                 .preceding = {"ACT"},
                 .following = {"ACT"},
                 .latency = V("nRC")},
                {.level = "bank",
                 .preceding = {"ACT"},
                 .following = {"RD", "RDA"},
                 .latency = V("nRCDRD")},
                {.level = "bank",
                 .preceding = {"ACT"},
                 .following = {"WR", "WRA"},
                 .latency = V("nRCDWR")},
                {.level = "bank",
                 .preceding = {"ACT"},
                 .following = {"PRE"},
                 .latency = V("nRAS")},
                {.level = "bank",
                 .preceding = {"PRE"},
                 .following = {"ACT"},
                 .latency = V("nRP")},
                {.level = "bank",
                 .preceding = {"RD"},
                 .following = {"PRE"},
                 .latency = V("nRTPL")},
                {.level = "bank",
                 .preceding = {"WR"},
                 .following = {"PRE"},
                 .latency = V("nCWL") + V("nBL") + V("nWR")},
                {.level = "bank",
                 .preceding = {"RDA"},
                 .following = {"ACT", "REFsb"},
                 .latency = V("nRTPL") + V("nRP")},
                {.level = "bank",
                 .preceding = {"WRA"},
                 .following = {"ACT", "REFsb"},
                 .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},

                // NOTE: LAX Timing Constraints
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4"},
                 .latency = V("nRRDS")},
                {.level = "pseudochannel",
                 .preceding = {"ACT4"},
                 .following = {"ACT4", "REG_WRITE"},
                 .latency = V("nRRDS")},
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4"},
                 .latency = V("nFAW")},
                {.level = "pseudochannel",
                 .preceding = {"ACT4"},
                 .following = {"ACT4", "REG_WRITE"},
                 .latency = V("nFAW")},

                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_20"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4", "COMP"},
                 .latency = 20 * V("nBL") + 2},
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_20"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4", "COMP"},
                 .latency = 20 * V("nCCDL") + 2},
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_16"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4", "COMP"},
                 .latency = 16 * V("nBL") + 2},
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_16"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4", "COMP"},
                 .latency = 16 * V("nCCDL") + 2},
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_8"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4", "COMP"},
                 .latency = 8 * V("nBL") + 2},
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_8"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4", "COMP"},
                 .latency = 8 * V("nCCDL") + 2},
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_4"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4", "COMP"},
                 .latency = 4 * V("nBL") + 2},
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_4"},
                 .following = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4", "COMP"},
                 .latency = 4 * V("nCCDL") + 2},
                {.level = "pseudochannel",
                 .preceding = {"REG_WRITE"},
                 .following = {"REG_WRITE", "COMP"},
                 .latency = V("nBL")},
                {.level = "pseudochannel",
                 .preceding = {"REG_WRITE"},
                 .following = {"REG_WRITE", "COMP"},
                 .latency = V("nCCDL")},

                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4"},
                 .following = {"COMP"},
                 .latency = V("nRCDRD")},
                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4"},
                 .following = {"COMP"},
                 .latency = V("nRCDWR")},
                {.level = "pseudochannel",
                 .preceding = {"ACT4"},
                 .following = {"COMP"},
                 .latency = V("nRCDRD")},
                {.level = "pseudochannel",
                 .preceding = {"ACT4"},
                 .following = {"COMP"},
                 .latency = V("nRCDWR")},

                {.level = "pseudochannel",
                 .preceding = {"ACT4_WITH_REG_20", "ACT4_WITH_REG_16",
                               "ACT4_WITH_REG_8", "ACT4_WITH_REG_4"},
                 .following = {"PRECHARGE_WITH_RESULT_16",
                               "PRECHARGE_WITH_RESULT_8"},
                 .latency = V("nRAS")},
                {.level = "pseudochannel",
                 .preceding = {"ACT4"},
                 .following = {"PRECHARGE"},
                 .latency = V("nRAS")},

                {.level = "pseudochannel",
                 .preceding = {"COMP"},
                 .following = {"COMP"},
                 .latency = V("nCCDL")},

                {.level = "pseudochannel",
                 .preceding = {"COMP"},
                 .following = {"PRECHARGE_WITH_RESULT_16",
                               "PRECHARGE_WITH_RESULT_8", "PRECHARGE"},
                 .latency = V("nRTPL")},
                {.level = "pseudochannel",
                 .preceding = {"COMP"},
                 .following = {"PRECHARGE_WITH_RESULT_16",
                               "PRECHARGE_WITH_RESULT_8", "PRECHARGE"},
                 .latency = V("nWR")},

                {.level = "pseudochannel",
                 .preceding = {"PRECHARGE_WITH_RESULT_16",
                               "PRECHARGE_WITH_RESULT_8", "PRECHARGE"},
                 .following = {"REFRESH_BARRIER"},
                 .latency = V("nRP")},
                {.level = "pseudochannel",
                 .preceding = {"PRECHARGE"},
                 .following = {"RESULT"},
                 .latency = V("nRP")},
                {.level = "pseudochannel",
                 .preceding = {"PRECHARGE_WITH_RESULT_16"},
                 .following = {"REFRESH_BARRIER"},
                 .latency = 16 * V("nCCDL") + 1},
                {.level = "pseudochannel",
                 .preceding = {"PRECHARGE_WITH_RESULT_16"},
                 .following = {"REFRESH_BARRIER"},
                 .latency = 16 * V("nBL") + 1},
                {.level = "pseudochannel",
                 .preceding = {"PRECHARGE_WITH_RESULT_8"},
                 .following = {"REFRESH_BARRIER"},
                 .latency = 8 * V("nCCDL") + 1},
                {.level = "pseudochannel",
                 .preceding = {"PRECHARGE_WITH_RESULT_8"},
                 .following = {"REFRESH_BARRIER"},
                 .latency = 8 * V("nBL") + 1},
                {.level = "pseudochannel",
                 .preceding = {"RESULT"},
                 .following = {"RESULT"},
                 .latency = V("nCCDL")},
                {.level = "pseudochannel",
                 .preceding = {"RESULT"},
                 .following = {"RESULT"},
                 .latency = V("nBL")},
            });
#undef V
    };

    void set_actions() {
        m_actions.resize(m_levels.size(),
                         std::vector<ActionFunc_t<Node>>(m_commands.size()));

        // Channel Actions
        m_actions[m_levels["channel"]][m_commands["PREA"]] =
            Lambdas::Action::Channel::PREab<LAX>;

        // Bank actions
        m_actions[m_levels["bank"]][m_commands["ACT"]] =
            Lambdas::Action::Bank::ACT<LAX>;
        m_actions[m_levels["bank"]][m_commands["PRE"]] =
            Lambdas::Action::Bank::PRE<LAX>;
        m_actions[m_levels["bank"]][m_commands["RDA"]] =
            Lambdas::Action::Bank::PRE<LAX>;
        m_actions[m_levels["bank"]][m_commands["WRA"]] =
            Lambdas::Action::Bank::PRE<LAX>;
    };

    void set_preqs() {
        m_preqs.resize(m_levels.size(),
                       std::vector<PreqFunc_t<Node>>(m_commands.size()));

        // Channel Actions
        m_preqs[m_levels["channel"]][m_commands["REFab"]] =
            Lambdas::Preq::Channel::RequireAllBanksClosed<LAX>;

        // Bank actions
        m_preqs[m_levels["bank"]][m_commands["REFsb"]] =
            Lambdas::Preq::Bank::RequireBankClosed<LAX>;
        m_preqs[m_levels["bank"]][m_commands["RD"]] =
            Lambdas::Preq::Bank::RequireRowOpen<LAX>;
        m_preqs[m_levels["bank"]][m_commands["WR"]] =
            Lambdas::Preq::Bank::RequireRowOpen<LAX>;

        // LAX
        auto id = [](auto, int cmd, auto, auto) { return cmd; };
        m_preqs[m_levels["pseudochannel"]][m_commands["REFRESH_BARRIER"]] = id;
        m_preqs[m_levels["pseudochannel"]][m_commands["ACT4_WITH_REG_20"]] = id;
        m_preqs[m_levels["pseudochannel"]][m_commands["ACT4_WITH_REG_16"]] = id;
        m_preqs[m_levels["pseudochannel"]][m_commands["ACT4_WITH_REG_8"]] = id;
        m_preqs[m_levels["pseudochannel"]][m_commands["ACT4_WITH_REG_4"]] = id;
        m_preqs[m_levels["pseudochannel"]][m_commands["ACT4"]] = id;
        m_preqs[m_levels["pseudochannel"]][m_commands["REG_WRITE"]] = id;
        m_preqs[m_levels["pseudochannel"]][m_commands["COMP"]] = id;
        m_preqs[m_levels["pseudochannel"]]
               [m_commands["PRECHARGE_WITH_RESULT_16"]] = id;
        m_preqs[m_levels["pseudochannel"]]
               [m_commands["PRECHARGE_WITH_RESULT_8"]] = id;
        m_preqs[m_levels["pseudochannel"]][m_commands["PRECHARGE"]] = id;
        m_preqs[m_levels["pseudochannel"]][m_commands["RESULT"]] = id;
    };

    void set_rowhits() {
        m_rowhits.resize(m_levels.size(),
                         std::vector<RowhitFunc_t<Node>>(m_commands.size()));

        m_rowhits[m_levels["bank"]][m_commands["RD"]] =
            Lambdas::RowHit::Bank::RDWR<LAX>;
        m_rowhits[m_levels["bank"]][m_commands["WR"]] =
            Lambdas::RowHit::Bank::RDWR<LAX>;
    }

    void set_rowopens() {
        m_rowopens.resize(m_levels.size(),
                          std::vector<RowhitFunc_t<Node>>(m_commands.size()));

        m_rowopens[m_levels["bank"]][m_commands["RD"]] =
            Lambdas::RowOpen::Bank::RDWR<LAX>;
        m_rowopens[m_levels["bank"]][m_commands["WR"]] =
            Lambdas::RowOpen::Bank::RDWR<LAX>;
    }

    void create_nodes() {
        int num_channels = m_organization.count[m_levels["channel"]];
        for (int i = 0; i < num_channels; i++) {
            Node *channel = new Node(this, nullptr, 0, i);
            m_channels.push_back(channel);
        }
    };
};

}  // namespace Ramulator
