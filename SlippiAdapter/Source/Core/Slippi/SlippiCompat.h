#pragma once
#include "Common/Config/Config.h"
#include "Common/FileUtil.h"
#include "Common/CommonPaths.h"
#include "Core/Config/MainSettings.h"
#include "Core/Slippi/SlippiConfig.h"
#include <filesystem>
namespace Config {
// Main.Slippi

// Netplay Settings
inline const Info<int> SLIPPI_ONLINE_DELAY{{System::Main, "Slippi", "OnlineDelay"}, 2};
inline const Info<bool> SLIPPI_ENABLE_SPECTATOR{{System::Main, "Slippi", "EnableSpectator"}, true};
inline const Info<int> SLIPPI_SPECTATOR_LOCAL_PORT{{System::Main, "Slippi", "SpectatorLocalPort"}, 51441};
inline const Info<bool> SLIPPI_SAVE_REPLAYS{{System::Main, "Slippi", "SaveReplays"}, true};
inline const Info<Slippi::Chat> SLIPPI_ENABLE_QUICK_CHAT{{System::Main, "Slippi", "EnableQuickChat"},
                                                  Slippi::Chat::ON};
inline const Info<bool> SLIPPI_FORCE_NETPLAY_PORT{{System::Main, "Slippi", "ForceNetplayPort"}, false};
inline const Info<int> SLIPPI_NETPLAY_PORT{{System::Main, "Slippi", "NetplayPort"}, 2626};
inline const Info<bool> SLIPPI_FORCE_LAN_IP{{System::Main, "Slippi", "ForceLanIP"}, false};
inline const Info<std::string> SLIPPI_LAN_IP{{System::Main, "Slippi", "LanIP"}, ""};
inline const Info<bool> SLIPPI_REPLAY_MONTHLY_FOLDERS{{System::Main, "Slippi", "ReplayMonthlyFolders"},
                                               true};
inline const Info<std::string> SLIPPI_REPLAY_DIR{{System::Main, "Slippi", "ReplayDir"},
                                          ""};
inline const Info<bool> SLIPPI_ENABLE_FRAME_INDEX{{System::Main, "Slippi", "EnableFrameIndex"}, false};
inline const Info<bool> SLIPPI_BLOCKING_PIPES{{System::Main, "Slippi", "BlockingPipes"}, false};
inline const Info<bool> SLIPPI_ENABLE_JUKEBOX{{System::Main, "Slippi", "EnableJukebox"}, true};
inline const Info<int> SLIPPI_JUKEBOX_VOLUME{{System::Main, "Slippi", "JukeboxVolume"}, 100};

inline const Info<bool> SLIPPI_ENABLE_RANK_LOCAL{{System::Main, "Slippi", "ShowLocalRankInfo"}, true};
inline const Info<bool> SLIPPI_ENABLE_RANK_OPP{{System::Main, "Slippi", "ShowOpponentRankInfo"}, true};

// Playback Settings
inline const Info<bool> SLIPPI_ENABLE_SEEK{{System::Main, "Slippi", "EnableSeek"}, true};

}
namespace SlippiCompat {
inline Slippi::Config& RuntimeConfig() { static Slippi::Config value{Melee::Version::NTSC}; return value; }
inline const std::string& Version() { static const std::string value="3.6.4+meleepad.probe"; return value; }
inline std::string UserDirectory() { return File::GetUserPath(D_USER_IDX) + "Slippi/"; }
inline u64 FileModificationTime(const std::string& path) {
  std::error_code error; const auto time=std::filesystem::last_write_time(path,error);
  return error ? 0 : static_cast<u64>(time.time_since_epoch().count());
}
}

std::string TruncateLengthChar(const std::string&, int);
std::string ConvertStringForGame(const std::string&, int);

// Isolated boot probe settings: never persisted as an app preference.
#include "Core/HW/EXI/EXI_Device.h"
#include <array>
#include <atomic>
namespace SlippiCompat {
inline std::string boot_iso_path;
inline bool boot_enabled = false;
inline constexpr auto device_type = static_cast<ExpansionInterface::EXIDeviceType>(0x100);
inline std::array<std::atomic<u64>, 256> command_counts{};
inline std::atomic<u64> device_creations{0};
inline u32 loaded_gct_address = 0;
inline u32 loaded_gct_size = 0;
inline std::atomic<u64> gct_handler_invalidations{0};
}

// Test-only synchronous callbacks on the emulated CPU thread.
#include <functional>
namespace SlippiCompat {
inline std::function<void(u8, const u8*, u32)> game_frame_observer;
}
