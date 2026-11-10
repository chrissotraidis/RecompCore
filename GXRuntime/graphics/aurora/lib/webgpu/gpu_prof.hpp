#pragma once

#include <string_view>

#include <webgpu/webgpu_cpp.h>

namespace aurora::webgpu::gpu_prof {

void initialize();
void shutdown();

void frame_begin(const wgpu::CommandEncoder& encoder);
void frame_end(const wgpu::CommandEncoder& encoder);
void after_submit();
const wgpu::PassTimestampWrites* pass_writes(std::string_view name);
// Prefixes the names of the passes and zones recorded until it is cleared (an
// empty prefix), so [gpu-prof] tells apart work of the same name ("EFB" in a
// real frame and in an in-between frame's replay). Render worker only.
void set_name_prefix(std::string_view prefix);

// Bytes the frame's staging copies move, by kind, for DOL_AURORA_GPU_PROF's
// [gpu-prof] lines (KB per frame).
enum class CopyKind : uint8_t { Vertices, Uniforms, Indices, Storage, Textures, Count };
void count_copy(CopyKind kind, uint64_t bytes);

class Zone {
public:
  Zone(const wgpu::CommandEncoder& encoder, std::string_view name);
  ~Zone();
  Zone(const Zone&) = delete;
  Zone& operator=(const Zone&) = delete;

private:
  const wgpu::CommandEncoder* m_encoder = nullptr;
  uint32_t m_endQuery = 0;
};

} // namespace aurora::webgpu::gpu_prof
