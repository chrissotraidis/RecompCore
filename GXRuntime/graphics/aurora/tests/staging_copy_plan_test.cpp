#include "gfx/staging_copy_plan.hpp"

#include <cstdio>

namespace {
int failures = 0;

void check(bool condition, const char* expression, int line) {
  if (condition)
    return;
  std::fprintf(stderr, "staging_copy_plan_test:%d: %s\n", line, expression);
  ++failures;
}
} // namespace

#define CHECK(expression) check((expression), #expression, __LINE__)

int main() {
  const auto normal = aurora::gfx::staging_copy_plan(false);
  CHECK(normal.copyBuffersAtOperation);
  CHECK(normal.copyTexturesAtOperation);
  CHECK(!normal.submitBuffersBeforeRender);
  CHECK(aurora::gfx::needs_incremental_staging_copy(normal, true, false));
  CHECK(aurora::gfx::needs_incremental_staging_copy(normal, false, true));
  CHECK(!aurora::gfx::needs_incremental_staging_copy(normal, false, false));

  const auto batched = aurora::gfx::staging_copy_plan(true);
  CHECK(!batched.copyBuffersAtOperation);
  CHECK(batched.copyTexturesAtOperation);
  CHECK(batched.submitBuffersBeforeRender);
  CHECK(!aurora::gfx::needs_incremental_staging_copy(batched, true, false));
  CHECK(aurora::gfx::needs_incremental_staging_copy(batched, false, true));
  CHECK(aurora::gfx::needs_incremental_staging_copy(batched, true, true));

  if (failures == 0)
    std::puts("staging_copy_plan_test: all checks passed");
  return failures == 0 ? 0 : 1;
}
