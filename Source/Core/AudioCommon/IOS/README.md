# Maintained iOS audio variant

This is the exact v8 mixer used by the GalaxyPad build 13 device host. CMake
selects it only for iOS; `AudioCommon/Mixer.h` selects its layout for iPhoneOS
and iPhoneSimulator consumers. Desktop retains its existing mixer.

DMA is limited to 32 kHz. An unsupported rate explicitly disables the tempo
path until the stream is recreated. This preserves an accepted device policy;
it does not establish new device or listening acceptance.

Frozen source SHA-256 provenance:

- Mixer.cpp: `102a876d8efe784e2c37c4cadae46eadc4479f0776797a8388d16d31215565e1`
- Mixer.h: `4e14a7db9b84f372b65ac660c8653c1f7207f243d36e2bfe9b84b5ed81ee0c27`
- ../AudioTempo.h: `eff12bf32da6aa3fb6ccf073117f1b0001df1c23e3ef3f5baf21bd1036fbba5a`

The deployed host's file-backed Mach-O sections match the build 13 prepared
host, linked against the frozen core archive SHA-256
`7fe0ea02261edb7d89853e6d83f8a687b5783cf3f50b0b67671591d6effb4e5a`.
Rebuild every Mixer header consumer when changing variants; never swap a lone
Mixer object into an archive built with a different layout.

Opt-in audio search experiment (September 15, 2026):
`GALAXYPAD_AUDIO_BATCHED_SEARCH=1` evaluates four independent correlation
candidates together. Each candidate retains its original sample accumulation
order, double accumulators and ascending tie selection; incomplete batches use
the scalar path. It changes no class layout, queue policy or search range.
The default remains scalar pending physical comparison. The frozen hashes above
identify the original accepted inputs, not this optional source extension.
Batched-search baseline AudioTempo.h SHA-256: `373a6a3c9db469a386e963c77c9d36060becf22da245a555b69b39444e7463ed`.
GalaxyPad's differential PCM/accounting/sanitizer test covers both variants.
This addition was implemented with AI assistance; no upstream submission is made.

## Optional sample-energy cache

`GALAXYPAD_AUDIO_CACHED_ENERGY=1` computes repeated search sample energies once
per search, retaining the float expressions, ordered double accumulation and
ascending tie-breaking. It works with scalar or batched search, allocates no
heap memory and does not alter class layout. Default remains disabled. The
additional fixed stack scratch is 3.5 KiB. Cached-energy revision AudioTempo.h SHA-256: `4830d3f0568cb8faf50c7c2fb7720f50b9958bbed37dc310fddbbd4898035871`.
Host component timing and exact PCM/accounting validation are recorded in
GalaxyPad docs/PERFORMANCE-CONTINUATION-2026-09-16.md; no iPhone FPS claim.

## Opt-in low-speed continuity repair

`GALAXYPAD_AUDIO_LOW_SPEED=1` lowers the analysis floor from 0.60 to 0.45,
reduces the queue control target by one hop, and leaves unity bypass one hop
earlier when measured supply falls. This addresses persistent starvation below
36 FPS and short gaps on a sharp slowdown. Tested supply coverage begins at
0.55; 0.45 is correction headroom, not a supported sustained speed claim.
It preserves normal-speed PCM, input accounting, bounded storage and real stall
reporting. Existing 0.67-1.0 mixer latency checks retain their 120 ms limit; new
0.55-0.60 coverage uses a separate 160 ms wall-age limit, excluding hardware
output latency. No default promotion or game-FPS gain is implied.

Current AudioTempo.h SHA-256: `404bbaba9c08ba96b2f7708078e9a6d444660fb4be68200d34a8fab5c0d5561f`.
