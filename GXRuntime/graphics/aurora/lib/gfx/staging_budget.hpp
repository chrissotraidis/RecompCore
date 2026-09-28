#pragma once

#include <cstddef>

namespace aurora::gfx {

struct StagingUse {
  size_t verts = 0;
  size_t indices = 0;
  size_t uniforms = 0;
  size_t storage = 0;
};

struct StagingRequest {
  size_t verts = 0;
  size_t indices = 0;
  size_t uniforms = 0;
  size_t secondUniforms = 0;
  size_t storage = 0;
};

struct StagingLimits {
  StagingUse capacity{};
  size_t uniformAlignment = 1;
  size_t storageAlignment = 1;
  size_t finishUniformHeadroom = 0;
};

constexpr size_t staging_project(size_t current, size_t length,
                                 size_t alignment) {
  const size_t aligned = alignment <= 1
                             ? current
                             : ((current + alignment - 1) / alignment) * alignment;
  return aligned + length;
}

constexpr StagingUse staging_project(const StagingUse& current,
                                     const StagingRequest& request,
                                     const StagingLimits& limits) {
  StagingUse projected{
      .verts = staging_project(current.verts, request.verts, 4),
      .indices = staging_project(current.indices, request.indices, 4),
      .uniforms = staging_project(current.uniforms, request.uniforms,
                                  limits.uniformAlignment),
      .storage = staging_project(current.storage, request.storage,
                                 limits.storageAlignment),
  };
  if (request.secondUniforms != 0) {
    projected.uniforms = staging_project(projected.uniforms,
                                         request.secondUniforms,
                                         limits.uniformAlignment);
  }
  projected.uniforms += limits.finishUniformHeadroom;
  return projected;
}

constexpr bool staging_fits(const StagingUse& current,
                            const StagingRequest& request,
                            const StagingLimits& limits) {
  const StagingUse projected = staging_project(current, request, limits);
  return projected.verts <= limits.capacity.verts &&
         projected.indices <= limits.capacity.indices &&
         projected.uniforms <= limits.capacity.uniforms &&
         projected.storage <= limits.capacity.storage;
}

} // namespace aurora::gfx
