#include "BackendBinding.hpp"

#import <Foundation/Foundation.h>
#include <SDL3/SDL_metal.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>

namespace aurora::webgpu::utils {
// One Metal view per window, reused by every surface. The surface is released
// while an iOS app is in the background and created again on return; creating
// a view each time stacked a new one above everything already in the window,
// which on iPadOS covered the touch controls after a home-screen round trip,
// and never removed the old views. The view lives as long as its window.
static constexpr const char* kAuroraMetalViewProperty = "aurora.metal_view";

std::shared_ptr<wgpu::ChainedStruct> SetupWindowAndGetSurfaceDescriptorCocoa(SDL_Window* window) {
  const SDL_PropertiesID props = SDL_GetWindowProperties(window);
  SDL_MetalView view = SDL_GetPointerProperty(props, kAuroraMetalViewProperty, nullptr);
  if (view == nullptr) {
    view = SDL_Metal_CreateView(window);
    if (view != nullptr)
      SDL_SetPointerProperty(props, kAuroraMetalViewProperty, view);
  }
  if (view == nullptr)
    return nullptr;
  std::shared_ptr<wgpu::SurfaceSourceMetalLayer> desc = std::make_shared<wgpu::SurfaceSourceMetalLayer>();
  desc->layer = SDL_Metal_GetLayer(view);
  return std::move(desc);
}
} // namespace aurora::webgpu::utils
