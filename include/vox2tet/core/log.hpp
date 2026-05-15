#pragma once

// Tiny logger:
//   HH:MM:SS:ffffff: <source_file>:\t<text>
// The source file is captured at the call site via __FILE__.

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace vox2tet::log {

void print_now(std::string_view text, const char* source_file_basename);

// Convenience: pulls the basename out of __FILE__.
#define VOX2TET_PRINT(text) ::vox2tet::log::print_now((text), __FILE__)

// std::ostringstream-style helper:
//   VOX2TET_LOG() << "count: " << n;
class StreamLogger {
public:
    explicit StreamLogger(const char* file) : file_(file) {}
    ~StreamLogger() { print_now(buf_.str(), file_); }

    template <typename T>
    StreamLogger& operator<<(const T& v) { buf_ << v; return *this; }

private:
    const char* file_;
    std::ostringstream buf_;
};

#define VOX2TET_LOG() ::vox2tet::log::StreamLogger(__FILE__)

// Redirect global cout-equivalent output to a file (mirrors
// Cglobal.init_print / close_print).
struct PrintRedirect {
    bool         active        = false;
    std::ostream* original_out = nullptr;
    void* file_handle          = nullptr;
};

PrintRedirect open_redirect(const std::string& file_path);
void          close_redirect(PrintRedirect& r);

}  // namespace vox2tet::log
