#pragma once

#include "persistent_decode.hpp"

#include <string_view>

namespace rwkv7::persistent_decode {

void run_persistent_decode_self_test(const Profile &profile,
                                     std::string_view profile_bytes);

} // namespace rwkv7::persistent_decode
