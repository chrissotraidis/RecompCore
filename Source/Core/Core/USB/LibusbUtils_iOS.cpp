// Copyright 2026 SunPad project
// SPDX-License-Identifier: GPL-2.0-or-later

// iOS stub for LibusbUtils: iOS has no USB host access (no libusb/IOKit on
// the device SDK). The API is kept linkable with no-op implementations so the
// rest of the core (e.g. the Wii USB host scanner) can be built unchanged.

#include "Core/USB/LibusbUtils.h"

#include <optional>
#include <string>
#include <utility>

namespace LibusbUtils
{
class Context::Impl
{
};

Context::Context() = default;
Context::~Context() = default;
Context::operator libusb_context*() const
{
  return nullptr;
}
bool Context::IsValid() const
{
  return false;
}
int Context::GetDeviceList(const GetDeviceListCallback& /*callback*/) const
{
  return -1;
}

std::pair<int, ConfigDescriptor> MakeConfigDescriptor(libusb_device* /*device*/,
                                                      u8 /*config_num*/)
{
  return {-1, ConfigDescriptor(nullptr, nullptr)};
}

const char* ErrorWrap::GetStrError() const
{
  return "USB unavailable on iOS";
}
const char* ErrorWrap::GetName() const
{
  return "LIBUSB_ERROR_NO_DEVICE";
}

std::optional<std::string> GetStringDescriptor(libusb_device_handle* /*dev_handle*/,
                                               uint8_t /*desc_index*/)
{
  return std::nullopt;
}
}  // namespace LibusbUtils
