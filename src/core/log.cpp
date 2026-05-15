#include "vox2tet/core/log.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace vox2tet::log {

namespace {

std::mutex g_log_mutex;

// Single hold-pointer for cout's original rdbuf during redirection.
std::ofstream*           g_redirect_file = nullptr;
std::streambuf*          g_saved_cout    = nullptr;

std::string current_timestamp() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto t   = system_clock::to_time_t(now);
    const auto us  = duration_cast<microseconds>(now.time_since_epoch()) % 1'000'000;

    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char hms[16];
    std::strftime(hms, sizeof(hms), "%H:%M:%S", &tm_buf);

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%s:%06lld", hms, static_cast<long long>(us.count()));
    return buf;
}

}  // namespace

void print_now(std::string_view text, const char* source_file_path) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    std::string base = std::filesystem::path(source_file_path).filename().string();
    std::cout << current_timestamp() << ": " << base << ":\t" << text << '\n';
    std::cout.flush();
}

PrintRedirect open_redirect(const std::string& path) {
    PrintRedirect r;
    std::lock_guard<std::mutex> lk(g_log_mutex);
    if (g_redirect_file) {
        // Already redirected — close the previous file first.
        std::cout.flush();
        std::cout.rdbuf(g_saved_cout);
        g_redirect_file->close();
        delete g_redirect_file;
        g_redirect_file = nullptr;
        g_saved_cout    = nullptr;
    }
    g_redirect_file = new std::ofstream(path);
    if (!g_redirect_file->is_open()) {
        delete g_redirect_file;
        g_redirect_file = nullptr;
        std::cerr << "WARN: failed to open print redirect: " << path << '\n';
        return r;
    }
    g_saved_cout = std::cout.rdbuf(g_redirect_file->rdbuf());
    r.active     = true;
    return r;
}

void close_redirect(PrintRedirect& r) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    if (!r.active) return;
    if (g_redirect_file) {
        std::cout.flush();
        std::cout.rdbuf(g_saved_cout);
        g_redirect_file->close();
        delete g_redirect_file;
        g_redirect_file = nullptr;
        g_saved_cout    = nullptr;
    }
    r.active = false;
}

}  // namespace vox2tet::log
