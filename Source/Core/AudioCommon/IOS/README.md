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
Current AudioTempo.h SHA-256: `373a6a3c9db469a386e963c77c9d36060becf22da245a555b69b39444e7463ed`.
GalaxyPad's differential PCM/accounting/sanitizer test covers both variants.
This addition was implemented with AI assistance; no upstream submission is made.
