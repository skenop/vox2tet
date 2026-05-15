#pragma once

// Filesystem-path utilities used by the I/O and pipeline layers
// (basename / extension / make-folder helpers, etc.).

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vox2tet::paths {

// Ensure that the directory containing `file_path_or_base` exists.
// When `do_dir_name` is true, treat `file_path_or_base` as a base file path
// (no extension) and create the parent directory; otherwise create the
// path itself as a directory. Matches `utils.createFolder`.
void create_folder(const std::string& file_path_or_base, bool do_dir_name = true);

// Strip the extension from a file path (keeps the directory and stem).
std::string base_file_path(const std::string& path);

// Return the extension including the leading dot.
std::string file_extension(const std::string& path, bool to_upper = false);

// File name with extension.
std::string file_name(const std::string& path);

// Stem (file name without extension).
std::string base_file_name(const std::string& path);

// Directory portion (absolute).
std::string base_folder(const std::string& path);

// If `path` exists, append the modification time to its stem and rename.
// Mirrors `utils.rename_if_exist`.
void rename_if_exist(const std::vector<std::string>& path_list);

// All files in the list exist.
bool is_exist_files(const std::vector<std::string>& path_list);

// Locate an executable on $PATH (and $PATHEXT on Windows). `nullopt`
// when not found.
std::optional<std::string> which(const std::string& program);

}  // namespace vox2tet::paths
