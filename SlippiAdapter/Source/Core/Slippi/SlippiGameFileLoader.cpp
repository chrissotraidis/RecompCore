#include "SlippiGameFileLoader.h"

#include "Common/FileUtil.h"
#include "Common/IOFile.h"
#include "Common/Logging/Log.h"
#include "Core/Boot/Boot.h"
#include "Core/Config/ConfigManager.h"
#include "Core/Core.h"
#include "Core/HW/DVD/DVDThread.h"
#include "Core/System.h"

SlippiGameFileLoader::SlippiGameFileLoader(std::string directory)
    : resource_directory(directory.empty() ? File::GetSysDirectory() : directory)
{
  if (!resource_directory.ends_with("/")) resource_directory += "/";
}

static std::string getFilePath(const std::string& file_name, const std::string& dir_path)
{

  std::string file_path = dir_path + "GameFiles/GALE01/" + file_name;  // TODO: Handle other games?

  if (File::Exists(file_path))
  {
    return file_path;
  }

  file_path = file_path + ".diff";
  if (File::Exists(file_path))
  {
    return file_path;
  }

  return "";
}

u32 SlippiGameFileLoader::LoadFile(Core::System& system, std::string file_name, std::string& data)
{
  if (file_cache.count(file_name))
  {
    data = file_cache[file_name];
    return static_cast<u32>(data.size());
  }

  if (grpsx_strings.count(file_name) && system.GetDVDThread().HasDisc())
  {
    std::vector<u8> buf;
    if (!system.GetDVDThread().ReadFile(file_name, buf)) { data.clear(); return 0; }
    std::string contents(buf.begin(), buf.end());

    file_cache[file_name] = contents;
    data = file_cache[file_name];
    INFO_LOG_FMT(SLIPPI, "Preloaded Transformation: {} -> {}", file_name.c_str(), static_cast<u32>(data.size()));
    return static_cast<u32>(data.size());
  }

  INFO_LOG_FMT(SLIPPI, "Loading file: {}", file_name.c_str());


  std::string game_file_path = getFilePath(file_name, resource_directory);
  if (game_file_path.empty())
  {
    file_cache[file_name] = "";
    data = "";
    return 0;
  }

  std::string file_contents;
  if (!File::ReadFileToString(game_file_path, file_contents)) { data.clear(); return 0; }

  // A delta requires its base disc file and a successful decode. Never return
  // undecoded delta bytes when the disc is unavailable.
  if (game_file_path.ends_with(".diff"))
  {
    std::vector<u8> buf;
    INFO_LOG_FMT(SLIPPI, "Will process diff");
    if (!system.GetDVDThread().ReadFile(file_name, buf)) { data.clear(); return 0; }
    std::string diff_contents = file_contents;
    file_contents.clear();
    if (!decoder.Decode((char*)buf.data(), buf.size(), diff_contents, &file_contents))
    { data.clear(); return 0; }
  }

  file_cache[file_name] = file_contents;
  data = file_cache[file_name];
  INFO_LOG_FMT(SLIPPI, "File size: {}", static_cast<u32>(data.size()));
  return static_cast<u32>(data.size());
}
