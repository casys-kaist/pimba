#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "base/base.h"
#include "base/config.h"
#include "example/example_ifce.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"
#include "yaml-cpp/emitter.h"
#include "yaml-cpp/node/parse.h"

int main(int argc, char *argv[]) {
    // Parse command line arguments
    argparse::ArgumentParser program("LAXSim", "0.1");
    program.add_argument("-f", "--config_file")
        .metavar("path-to-configuration-file")
        .help("Path to a YAML configuration file.")
        .required();
    program.add_argument("-o", "--output_file")
        .metavar("path-to-output-file")
        .help("Path to a output file")
        .required();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        spdlog::error(err.what());
        std::cerr << program;
        std::exit(1);
    }

    // config
    std::string config_file_path = *program.present("-f");
    YAML::Node config =
        Ramulator::Config::parse_config_file(config_file_path, {});

    // modules
    auto frontend = Ramulator::Factory::create_frontend(config);
    auto memory_system = Ramulator::Factory::create_memory_system(config);

    frontend->connect_memory_system(memory_system);
    memory_system->connect_frontend(frontend);

    // Get the relative clock ratio between the frontend and memory system
    int frontend_tick = frontend->get_clock_ratio();
    int mem_tick = memory_system->get_clock_ratio();

    int tick_mult = frontend_tick * mem_tick;

    for (uint64_t i = 0;; i++) {
        if (((i % tick_mult) % mem_tick) == 0) {
            frontend->tick();
        }

        if (frontend->is_finished()) {
            break;
        }

        if ((i % tick_mult) % frontend_tick == 0) {
            memory_system->tick();
        }
    }

    // Finalize the simulation. Recursively print all statistics from all
    // components
    frontend->finalize();
    memory_system->finalize();

    // Output parse
    YAML::Emitter m_emitter;
    m_emitter << YAML::BeginMap;
    memory_system->m_impl->print_stats(m_emitter);
    m_emitter << YAML::EndMap;

    YAML::Node m_out = YAML::Load(m_emitter.c_str());
    uint64_t cycles =
        m_out["MemorySystem"]["memory_system_cycles"].as<uint64_t>();
    uint64_t num_act = m_out["MemorySystem"]["DRAM"]["num_act"].as<uint64_t>();
    uint64_t num_comp =
        m_out["MemorySystem"]["DRAM"]["num_comp"].as<uint64_t>();
    uint64_t num_rdwr =
        m_out["MemorySystem"]["DRAM"]["num_rdwr"].as<uint64_t>();

    std::string output_file_path = *program.present("-o");
    std::ofstream ofs(output_file_path);
    ofs << "cycles: " << cycles << "\n";
    ofs << "num_act: " << num_act << "\n";
    ofs << "num_comp: " << num_comp << "\n";
    ofs << "num_rdwr: " << num_rdwr << "\n";

    return 0;
}
