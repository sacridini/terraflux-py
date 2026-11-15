// SPDX-License-Identifier: GPL-3.0-or-later
// Environment switches: TERRAFLUX_<NAME> first, else the name landschaft used (LANDSCHAFT_<NAME>).
#pragma once

#include <cstdlib>
#include <string>

namespace tfenv {

// True when TERRAFLUX_<name> or LANDSCHAFT_<name> is set (to anything, even empty).
inline bool is_set(const char* name) {
    const std::string n(name);
    return std::getenv(("TERRAFLUX_" + n).c_str()) != nullptr || std::getenv(("LANDSCHAFT_" + n).c_str()) != nullptr;
}

}  // namespace tfenv
