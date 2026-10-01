// SPDX-License-Identifier: GPL-3.0-or-later
#include "../lib/gfx/pipeline_cache.hpp"
#include "../lib/internal.hpp"
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
int main() {
    auto directory = std::filesystem::temp_directory_path() /
        ("bluewake-pipeline-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    std::string path = directory.string();
    aurora::g_config.cachePath = path.c_str();
    aurora::g_config.userPath = path.c_str();
    aurora::g_config.resourcesPath = path.c_str(); // empty synthetic cache, no bundled game pipelines
    for (unsigned i = 0; i < 1000; ++i) {
        aurora::gfx::initialize_pipeline_cache();
        if (i % 2) std::this_thread::yield();
        aurora::gfx::shutdown_pipeline_cache();
    }
    std::filesystem::remove_all(directory);
}
