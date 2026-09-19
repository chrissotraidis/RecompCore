# MeleePad Slippi adapter

Optional source adapter for MeleePad, derived from Project Slippi Dolphin commit
`41a7a3a110ed52999486ae1901c8fbb9a63d4f13` and this RecompCore fork.
Original copyright and license headers are retained. Slippi Dolphin's COPYING
is included here; open-vcdiff and semver retain their own notices.

The Source directory is the maintained adapter used by the iOS target. It
contains native static-recompilation adaptations, matchmaking mode gates,
replay support and bounded transport diagnostics from the owner-tested build 26.
The application supplies the slippi-direct-probe and diagnostics interfaces.
Source updates are ordinary reviewed commits, without replaying patches.

The Externals directory is copied verbatim from the pinned Slippi donor
(with unused third_party build/test dependencies omitted). No game image,
generated game module, account or replay is included. This port is independently
maintained and is not endorsed by Project Slippi.
