// SPDX-License-Identifier: GPL-2.0-or-later
#include "Core/Config/StaticRecompSettings.h"
#include "Core/System.h"

namespace Config
{
const Info<bool> MAIN_STATICRECOMP_MODULE{{System::Main, "Core", "StaticRecompModule"}, true};
const Info<u32> MAIN_STATICRECOMP_IDLE_PC{{System::Main, "Core", "StaticRecompIdlePC"}, 0};
const Info<u32> MAIN_STATICRECOMP_SECONDARY_IDLE_PC{
    {System::Main, "Core", "StaticRecompSecondaryIdlePC"}, 0};
const Info<u32> MAIN_STATICRECOMP_CALLER_IDLE_PC{
    {System::Main, "Core", "StaticRecompCallerIdlePC"}, 0};
const Info<u32> MAIN_STATICRECOMP_CALLER_IDLE_LR{
    {System::Main, "Core", "StaticRecompCallerIdleLR"}, 0};
// A yield function's return point, the stack offset where its caller's return address is saved,
// and the callers that call it only from wait loops (comma-separated hex).
const Info<u32> MAIN_STATICRECOMP_YIELD_IDLE_PC{
    {System::Main, "Core", "StaticRecompYieldIdlePC"}, 0};
const Info<u32> MAIN_STATICRECOMP_YIELD_IDLE_CALLER_OFFSET{
    {System::Main, "Core", "StaticRecompYieldIdleCallerOffset"}, 0};
const Info<std::string> MAIN_STATICRECOMP_YIELD_IDLE_CALLERS{
    {System::Main, "Core", "StaticRecompYieldIdleCallers"}, ""};
const Info<std::string> MAIN_STATICRECOMP_FALLBACK_RANGES{
    {System::Main, "Core", "StaticRecompFallbackRanges"}, ""};
}  // namespace Config
