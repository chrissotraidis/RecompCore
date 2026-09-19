// Derived from Project Slippi StringUtil.cpp, GPL-2.0-or-later.
#include "Common/StringUtil.h"
#include <unordered_map>
void ConvertNarrowSpecialSHIFTJIS(std::string& input)
{
  // Melee doesn't correctly display special characters in narrow form We need to convert them to
  // wide form. I couldn't find a library to do this so for now let's just do it manually
  static std::unordered_map<char, char16_t> specialCharConvert = {
      {'!', (char16_t)0x8149},  {'"', (char16_t)0x8168}, {'#', (char16_t)0x8194},
      {'$', (char16_t)0x8190},  {'%', (char16_t)0x8193}, {'&', (char16_t)0x8195},
      {'\'', (char16_t)0x8166}, {'(', (char16_t)0x8169}, {')', (char16_t)0x816a},
      {'*', (char16_t)0x8196},  {'+', (char16_t)0x817b}, {',', (char16_t)0x8143},
      {'-', (char16_t)0x817c},  {'.', (char16_t)0x8144}, {'/', (char16_t)0x815e},
      {':', (char16_t)0x8146},  {';', (char16_t)0x8147}, {'<', (char16_t)0x8183},
      {'=', (char16_t)0x8181},  {'>', (char16_t)0x8184}, {'?', (char16_t)0x8148},
      {'@', (char16_t)0x8197},  {'[', (char16_t)0x816d}, {'\\', (char16_t)0x815f},
      {']', (char16_t)0x816e},  {'^', (char16_t)0x814f}, {'_', (char16_t)0x8151},
      {'`', (char16_t)0x814d},  {'{', (char16_t)0x816f}, {'|', (char16_t)0x8162},
      {'}', (char16_t)0x8170},  {'~', (char16_t)0x8160},
  };

  int pos = 0;
  while (pos < input.length())
  {
    auto c = input[pos];
    if ((u8)(0x80 & (u8)c) == 0x80)
    {
      // This is a 2 char rune, move to next
      pos += 2;
      continue;
    }

    bool hasConversion = specialCharConvert.count(c);
    if (!hasConversion)
    {
      pos += 1;
      continue;
    }

    // Remove previous character
    input.erase(pos, 1);

    // Add new chars to pos to replace
    auto newChars = (char*)&specialCharConvert[c];
    input.insert(input.begin() + pos, 1, newChars[0]);
    input.insert(input.begin() + pos, 1, newChars[1]);
  }
}

std::string TruncateLengthChar(const std::string& input, int length)
{
  auto units = UTF8ToUTF16(input);
  size_t end = 0; int count = 0;
  while (end < units.size() && count < length) {
    auto ch = units[end++];
    if (ch >= 0xd800 && ch <= 0xdbff && end < units.size() && units[end] >= 0xdc00 && units[end] <= 0xdfff) ++end;
    ++count;
  }
  units.resize(end);
  return UTF16ToUTF8(units);
}

std::string ConvertStringForGame(const std::string& input, int length)
{
  auto utf8 = TruncateLengthChar(input, length);
  auto shiftJis = UTF8ToSHIFTJIS(utf8);
  ConvertNarrowSpecialSHIFTJIS(shiftJis);

  // Make fixed size
  shiftJis.resize(length * 2 + 1);
  return shiftJis;
}

