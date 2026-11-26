#pragma once

namespace aurora::gfx {

struct StagingCopyPlan {
  bool copyBuffersAtOperation = true;
  bool copyTexturesAtOperation = true;
  bool submitBuffersBeforeRender = false;
};

constexpr StagingCopyPlan staging_copy_plan(bool batchBuffers) noexcept {
  return {
      .copyBuffersAtOperation = !batchBuffers,
      .copyTexturesAtOperation = true,
      .submitBuffersBeforeRender = batchBuffers,
  };
}

constexpr bool needs_incremental_staging_copy(const StagingCopyPlan& plan, bool buffersPending,
                                              bool texturesPending) noexcept {
  return (plan.copyBuffersAtOperation && buffersPending) || (plan.copyTexturesAtOperation && texturesPending);
}

} // namespace aurora::gfx
