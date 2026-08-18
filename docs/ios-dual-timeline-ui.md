# iOS dual-timeline UI: implementation specification

This document is the implementation context for the iOS demo screen of on-device Hibiki 1B translation. It specifies one screen, the **dual timeline**, in two input modes.

It is a UI contract only. Artifact loading, the streaming schedule, cache ownership, sampling, and finalization are specified in [the core implementation reference](./core-library.md); this document never restates them. Domain terms are defined in [`CONTEXT.md`](../CONTEXT.md) and are used here exactly as written there — in particular **source frame**, **generation step**, **text frame**, **target audio frame**, **complete audio frame**, **inference session**, and **model time**.

> **Provenance:** this design won a four-variant HTML prototype driven by a real run of `assets/long-form/2.wav`. The prototype is the primary source for visual intent and lives on a throwaway branch, not on `main`. Where this document and the prototype disagree, this document wins — it corrects three things the prototype got wrong, each marked **Prototype correction** below.

## The one idea

Two lanes on one clock. The source lane and the target lane share a single horizontal time axis, so the **offset** between French going in and English coming out is the thing the screen shows. Every other element is subordinate to that reading.

This is why the design won: it is the only shape that shows the seconds of context Hibiki banks before it starts speaking, instead of apologising for them. The screen reports that lag by drawing it, not by explaining it.

The two-frame scheduler delay is a separate quantity and is not legible on this axis — 160 ms is under half a percent of a whole-run width. It appears only as numbers: the target lane's own frame count, and the playhead sitting at target audio frame `t − 2`. Do not conflate the two latencies in any label.

## Determinate and indeterminate

Input mode is not a cosmetic switch. It changes whether the screen has a **total**, and every row below reads that one fact:

| Element | File mode (determinate) | Live mode (indeterminate) |
| --- | --- | --- |
| Time axis | Whole run, `0 ..< totalGenerationSteps` | Rolling window, last 150 frames |
| Progress | `step 300 / 552` | `step 300` |
| Throughput | `1.8× faster than real time` — a result | `22.4 steps/s · needs 12.5` — a health reading |
| Bottom control | Progress track, then scrubber | Keeping-up gauge + Stop |
| Un-decoded text | Dimmed ahead of the playhead | Absent |

Read the last row as a rule, not a style: **a live session must never render a text frame that has not been sampled.** In file mode the transcript ahead of the playhead exists because the run finished; in live mode it does not exist at all, and drawing a placeholder for it is a correctness bug.

The same number carries opposite meaning across the two columns. `stepsPerSecond / 12.5` is a retrospective benchmark in file mode and a live margin in live mode. Label it differently in each; do not share one string.

## Screen anatomy

Top to bottom, one column, no horizontal scrolling:

```text
┌─────────────────────────────────────────┐
│ TRANSLATION RUN           [File|Live]   │  mode is a segmented control
│ 2.wav ▾   /   Microphone                │  file name opens the picker
├─────────────────────────────────────────┤
│  Audio 43.7s │ Wall clock 24.6s │ 1.8×  │  three metrics, mode-dependent
├─────────────────────────────────────────┤
│ FRENCH IN         level only — never…   │
│ ▁▃█▅▂▇█▃▁▂▅█▆▃▁▁▄▇█▅▂▁    (source)  ▕   │  playhead at source frame t
│ 0s   5s   10s   15s   20s   …     now   │  one shared ruler
│ ENGLISH OUT           429 / 550 frames  │
│ ▁▁▂▆█▄▁▃█▅▂▁▃▇█▄▁▂▆█▃▁    (target)  ▕   │  playhead at target frame t−2
├─────────────────────────────────────────┤
│ …the Global Citizens Festival, now at   │  caption ribbon, live word lit
│ its seventh edition, will see tens of   │
│ thousands of people run towards the …   │
├─────────────────────────────────────────┤
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░  step 430 / 552    │  scrubber (file) or gauge (live)
│ [       Stop / Start listening       ]  │  live mode only
└─────────────────────────────────────────┘
```

The two lanes and the ruler between them are one unit. They must share exact horizontal insets and one x-mapping function; a half-point mismatch destroys the only comparison the screen exists to make.

## Mapping model time to x

One function, used by both lanes, the ruler, and the playhead:

```swift
/// Maps a model-timeline frame index to a point on the shared axis.
/// `window` is a Range<Int> of generation steps; it may start below zero in live mode.
func x(forFrame frame: Int, in width: CGFloat) -> CGFloat {
    CGFloat(frame - window.lowerBound) / CGFloat(window.count) * width
}
```

`window` is the only thing that differs by mode:

- **File mode:** `0 ..< totalGenerationSteps`. Fixed for the whole run, so bars never move.
- **Live mode:** `(currentStep - 150) ..< currentStep`. 150 frames is 12.0 s at the 80 ms model clock. The playhead is pinned to the right edge and the bars translate leftward.

In the first 12 seconds of a live session the window's lower bound is negative. That is intentional — the axis keeps a constant 12-second scale instead of stretching — so the mapping must accept it while the draw loop clamps to frames that exist. Do not index a level buffer with the raw lower bound.

Both lanes map against **generation steps**, not their own array lengths. A source frame count and a target audio frame count differ (the tail adds steps; the delay removes two), and mapping each lane against its own length shears the two axes by about one percent — enough to break alignment at the resolution this screen is read at.

## Level, not waveform

Each lane draws one **level** per frame: the RMS of that frame's 1,920 samples, mapped to `0…1`.

```swift
func level(rms: Float) -> Float {
    guard rms > 0 else { return 0 }
    let dbfs = 20 * log10(rms)              // -inf … 0
    return min(max((dbfs + 50) / 50, 0), 1) // -50 dBFS floor
}
```

**Prototype correction.** The prototype normalised each lane against the peak of the whole file, which a live session cannot know — its peak is in the future. Worse, a running peak makes the lane rescale itself while someone is speaking. Use the fixed dBFS mapping above in **both** modes: it needs no lookahead, keeps the two modes visually comparable, and spends the lane's height on the 0.1–0.5 RMS band where speech actually sits. (The prototype's `sqrt` curve was a patch over this same problem; drop it.)

Source levels are computed from the PCM chunk feeding each source frame. Target levels are computed from the PCM of each complete audio frame. Both are cheap — one pass over 1,920 floats — and belong on the same background actor as the session, never on the main thread.

### Column aggregation

A lane is about 331 pt wide. A 60-second run is 750 frames, so frames outnumber points and one-bar-per-frame is sub-pixel. Aggregate to columns before drawing:

```swift
// framesPerColumn = max(1, ceil(window.count / Int(width)))
// column level = max(levels in that column)   // max, not mean
```

Use **max**, not mean: speech is sparse at 80 ms granularity and averaging flattens it into a uniform band. Draw the whole lane as a single `Path` inside a SwiftUI `Canvas` — one filled rect per column, mirrored about the lane's centre line. One `Shape` view per bar will not hold 12.5 Hz.

## Update cadence

The session produces at most one `StepResult` per **generation step**, so the screen has new information 12.5 times per second. Publish at that rate and no faster.

- Run the inference session on a dedicated background actor. Never in an audio callback, and never on the main actor — see the real-time I/O guidance in [the core reference](./core-library.md).
- Coalesce each step's outcome into one immutable snapshot and hand that to the main actor.
- Drive redraws from the snapshot, not from a `TimelineView(.animation)` running at display rate. A 60 Hz redraw over 12.5 Hz data burns four fifths of its frames producing identical pixels.
- The playhead may interpolate between steps for smoothness; the bars may not — a bar appearing before its frame exists is a lie about model time.

## View state

The screen renders from one value type. Everything in it is derived from `StepResult`s the session already returns.

```swift
struct TimelineSnapshot {
    enum Mode { case file(FileSource), live }
    enum Phase { case idle, loadingArtifacts, warmingUp, translating, finalizing, finished, failed(String) }

    let mode: Mode
    let phase: Phase

    // model timeline
    let generationStep: Int          // text frame index of the newest step
    let totalGenerationSteps: Int?   // nil in live mode — the indeterminate switch
    let completeAudioFrames: Int     // generationStep - 2, floored at 0

    // lanes: ring buffer in live mode, full arrays in file mode
    let sourceLevels: [Float]
    let targetLevels: [Float]

    // text
    let textFrames: [(frame: Int, piece: String)]   // sampled only
    let liveWordRange: Range<Int>?                  // last few, for highlight

    // metrics
    let modelTime: TimeInterval      // generationStep * 0.08
    let computeTime: TimeInterval    // wall clock spent translating
    let stepsPerSecond: Double       // rolling, measured
}
```

`totalGenerationSteps` being `Optional` is deliberate and load-bearing: it makes the determinate/indeterminate split a compile-time concern instead of a styling afterthought. Every element in the "Progress" and "Bottom control" rows of the table above reads it, and there is no sensible default to substitute when it is `nil`.

Derive `modelTime` from the frame index and the 80 ms clock — never from wall clock. `CONTEXT.md` keeps **model time** and processing time distinct, and this screen displays both side by side (`34.4s model · 19.2s compute`); computing one from the other collapses the distinction the screen is built to show.

## Phases and controls

The two phases before `translating` are not decoration — the artifact bundle is about 3.99 GB and loading it is visible time. All seven phases need a rendered state:

| Phase | Screen state |
| --- | --- |
| `idle` | Lanes empty, metrics dashed, control is Start / Start listening |
| `loadingArtifacts` | Lanes empty, control disabled, with a determinate progress if the loader supplies bytes |
| `warmingUp` | Same, control shows "Preparing" |
| `translating` | Lanes filling, playhead advancing, control is Stop |
| `finalizing` | Silence-tail finalization — six further silent frames; lanes still advance, control disabled |
| `finished` | File mode: timeline becomes scrubbable. Live mode: summary, control resets to Start |
| `failed` | Message in place of the caption ribbon, lanes frozen at last frame, control offers Retry |

**Prototype correction.** The prototype let you scrub at any time, because it was replaying a finished run. A live inference session only moves forward — seeking backwards would mean re-running it. So the bottom track has two distinct states in file mode: a **read-only progress track while `translating`**, and a **scrubber once `finished`**, over the retained levels and text plus the written output audio. Do not ship a draggable knob during translation.

In live mode the track is never a scrubber. It is the keeping-up gauge: a bar for measured `stepsPerSecond` against a fixed marker at 12.5, the rate needed to hold real time.

## Visual tokens

The winning variant is light and typographic — an instrument panel, not a chat app. It must read in both colour schemes; the tokens below are the light values.

| Token | Value | Use |
| --- | --- | --- |
| `surface` | `#F7F6F3` | Screen background |
| `ink` | `#14141A` | Primary text, source lane fill |
| `inkMuted` | `#6B6558` | Elapsed transcript |
| `hairline` | `#E2DED4` | Metric dividers, ruler baseline |
| `pending` | `#DDD8CB` | Lane past the playhead (file mode only) |
| `target` | `#2F7D68` | Target lane fill, healthy margin |
| `playhead` | `#C8632F` | Playhead, gauge threshold |
| `highlight` | `#F5E2B8` | Live word background |

Type: system font throughout. Metrics use **tabular figures** — a metric that shifts width as it counts reads as jitter. Set the ruler and the `step / compute` row in a monospaced face; they are instrument readouts.

Lane height 74 pt each, ruler 26 pt between them, one shared inset of 20 pt.

## What this screen must not show

Each of these is a constraint from the model, not a preference:

- **A French transcript.** Hibiki is speech-to-speech and never emits source text. The source lane is level only, and it says so in its own header. Adding on-device French ASR to fill the gap would show a second model's opinion on the same axis as Hibiki's output and invite reading one as the other.
- **Un-sampled text.** See the live-mode rule above.
- **A total in live mode.** No frame count, no percentage, no indeterminate spinner standing in for one.
- **Per-step phase timings or memory**, unless the session was built with timing measurement enabled. Blank is honest; a plausible-looking number is not.

## Open decisions

Both block a shippable live mode and neither is a UI question. Raise them rather than picking silently.

1. **Live output is a feedback loop.** With the microphone open and English playing from the speaker, the microphone hears the translation and feeds it back as source audio. This needs headphones, echo cancellation, or half-duplex operation. Nothing in this screen solves it.
2. **Live capture does not exist yet.** `InferenceSession.push_pcm` already buffers arbitrary-length PCM chunks into 80 ms source frames and `finish()` zero-pads the partial tail, so the session contract is ready for a microphone. The capture path, and the iOS-native session it feeds, are separate work. Until that lands, live mode has no data source.

## Conformance criteria

The screen is done when every item holds. These are checkable; treat any that cannot be checked as not done.

**Alignment and mapping**
- Both lanes, the ruler, and the playhead derive x from one function and one `window`; no lane maps against its own array length.
- At generation step `t`, the source lane fills to frame `t` and the target lane to frame `t − 2`.
- No target audio frame renders for the first two generation steps.

**Modes**
- Every element in the determinate/indeterminate table renders its own column's form; none reads a default when `totalGenerationSteps` is `nil`.
- No text frame renders in live mode before it is sampled.
- Live mode holds exactly 150 frames of levels and pins the playhead to the right edge.

**Levels**
- Level comes from the fixed dBFS mapping, with no dependence on a peak the session has not yet seen.
- Columns aggregate by max, and a 750-frame run draws without dropping below the update cadence.

**Timing and threading**
- `modelTime` is derived from the frame index and the 80 ms clock, never from wall clock.
- No inference, level computation, or PCM handling runs on the main actor or in an audio callback.
- The screen redraws at the step rate, not the display rate, and sustains 12.5 Hz for a 120-second session — the stated version-one duration limit — with bounded memory.

**Phases**
- All seven phases render, including `finalizing` and `failed`.
- The bottom track is read-only while `translating` and a scrubber only once `finished`, and is never a scrubber in live mode.

**Presentation**
- Both colour schemes pass contrast on every token pair above.
- Metrics use tabular figures, and no metric changes width while counting.
- Dynamic Type at accessibility sizes reflows the transcript and metrics without clipping; the lanes keep their aspect.
- VoiceOver reads the lanes as a summary value — elapsed model time, words emitted, and current margin — rather than announcing bars.
