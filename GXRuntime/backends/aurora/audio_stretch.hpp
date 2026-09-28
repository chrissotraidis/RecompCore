// SPDX-License-Identifier: GPL-3.0-or-later
// Time stretching for the audio output while the guest runs slower than real
// time (the Dolphin option of the same name, without its SoundTouch).
//
// When the game produces 32 kHz audio at, say, 77 percent of real time, the
// device drains the queue and plays silence between pushes: the heavy Outset
// view on the iPad simulator starved 39 percent of its pushes. Stretching the
// output in time - same pitch, slower tempo - turns that stutter into
// continuous sound that runs as slow as the game does. At full speed the
// stretcher is idle and samples pass through untouched.
//
// The method is synchronous overlap-add (SOLA): output a 40 ms sequence of
// input, then find where in the next 15 ms of input the waveform best matches
// the last 8 ms written, crossfade there, and advance the input by less than
// the output (by the stretch factor), so material repeats seamlessly.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gx_aurora {

class AudioStretcher {
public:
    // Frames at the stream rate; defaults suit 32 kHz (40 / 8 / 15 ms).
    explicit AudioStretcher(int sequence = 1280, int overlap = 256, int seek = 480)
        : seq_(sequence), ovl_(overlap), seek_(seek), mid_(static_cast<size_t>(overlap) * 2) {}

    bool active() const { return active_; }

    // Adds stereo frames and appends stretched output to out. stretch >= 1 is
    // output length over input length; 1 leaves the stretcher (flushing what
    // it holds, continuously) and passes input through bit-exactly.
    void process(const int16_t* in, size_t frames, double stretch, std::vector<int16_t>& out) {
        if (stretch <= 1.0005) {
            if (active_)
                leave(out);
            out.insert(out.end(), in, in + frames * 2);
            return;
        }
        if (!active_) {
            active_ = true;
            first_ = true;
            skip_fract_ = 0.0;
            in_.clear();
        }
        in_.insert(in_.end(), in, in + frames * 2);
        const double nominal_skip = static_cast<double>(seq_ - ovl_) / stretch;
        while (static_cast<int>(in_.size() / 2) >= seq_ + seek_) {
            int offset = 0;
            if (!first_) {
                offset = best_offset();
                // Crossfade the previous tail into the matched position.
                for (int i = 0; i < ovl_; ++i) {
                    const float w = static_cast<float>(i) / static_cast<float>(ovl_);
                    for (int c = 0; c < 2; ++c) {
                        const float a = mid_[static_cast<size_t>(i) * 2 + c];
                        const float b = in_[static_cast<size_t>(offset + i) * 2 + c];
                        out.push_back(clamp16(a * (1.0f - w) + b * w));
                    }
                }
            } else {
                out.insert(out.end(), in_.begin(), in_.begin() + ovl_ * 2);
                first_ = false;
            }
            // The body of the sequence, then keep its last ovl_ frames.
            const size_t body_begin = static_cast<size_t>(offset + ovl_) * 2;
            const size_t body_end = static_cast<size_t>(offset + seq_ - ovl_) * 2;
            out.insert(out.end(), in_.begin() + body_begin, in_.begin() + body_end);
            std::memcpy(mid_.data(), in_.data() + body_end, static_cast<size_t>(ovl_) * 2 * sizeof(int16_t));
            // Where the input continues after the tail just kept, before the skip.
            next_after_mid_ = offset + seq_;
            skip_fract_ += nominal_skip;
            const int skip = static_cast<int>(skip_fract_);
            skip_fract_ -= skip;
            in_.erase(in_.begin(), in_.begin() + static_cast<size_t>(skip) * 2);
            next_after_mid_ -= skip;
        }
    }

    // Drops all state (a new stream, or a flush without output).
    void reset() {
        active_ = false;
        in_.clear();
    }

private:
    static int16_t clamp16(float v) {
        v = std::nearbyint(v);
        return static_cast<int16_t>(std::clamp(v, -32768.0f, 32767.0f));
    }

    // The position in [0, seek_) whose next ovl_ frames best match the kept
    // tail, by normalised cross-correlation of the channel sum.
    int best_offset() const {
        double best = -1e300;
        int best_at = 0;
        for (int off = 0; off < seek_; off += 2) {
            double corr = 0.0, norm = 0.0;
            for (int i = 0; i < ovl_; i += 2) {
                const size_t m = static_cast<size_t>(i) * 2;
                const size_t n = static_cast<size_t>(off + i) * 2;
                const double a = static_cast<double>(mid_[m]) + mid_[m + 1];
                const double b = static_cast<double>(in_[n]) + in_[n + 1];
                corr += a * b;
                norm += b * b;
            }
            const double score = corr / std::sqrt(norm + 1.0);
            if (score > best) {
                best = score;
                best_at = off;
            }
        }
        return best_at;
    }

    // Back to pass-through without a gap or a jump: the kept tail, then the
    // input exactly where it continues after that tail.
    void leave(std::vector<int16_t>& out) {
        active_ = false;
        if (first_) {
            out.insert(out.end(), in_.begin(), in_.end());
        } else {
            out.insert(out.end(), mid_.begin(), mid_.begin() + ovl_ * 2);
            const size_t from = static_cast<size_t>(std::max(next_after_mid_, 0)) * 2;
            if (from < in_.size())
                out.insert(out.end(), in_.begin() + from, in_.end());
        }
        in_.clear();
    }

    int seq_, ovl_, seek_;
    bool active_ = false;
    bool first_ = true;
    double skip_fract_ = 0.0;
    int next_after_mid_ = 0;
    std::vector<int16_t> in_;
    std::vector<int16_t> mid_;
};

}  // namespace gx_aurora
