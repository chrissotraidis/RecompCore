#pragma once
#include "completion-recorder.h"

namespace galaxypad {
// Runtime configures before boot and exports only after both workers join.
inline CompletionRecorder completion_recorder;
inline void ObserveCompletion(void*, bool before) noexcept {
  completion_recorder.RecordGraphics(before ? CompletionRecorder::Kind::BeforeNotify :
                                              CompletionRecorder::Kind::AfterNotify);
}
} // namespace galaxypad
