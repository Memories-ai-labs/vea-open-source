
# Final Cut Pro XML (FCPXML) — A Practical, Comprehensive Authoring Guide

> **Purpose.** This guide is a practical “developer-facing” reference for **manually generating FCPXML** from code. It focuses on **correct syntax**, **time math**, and **coverage of the most common edit constructs** (cuts, trims, connected clips, audio roles, transitions, titles, retimes, multicam, captions, markers, metadata, formats, color spaces, 360/VR, etc.). It is written to be embedded in a repository and used by LLMs or generators that convert higher-level edit commands to FCPXML.

> **Scope / Versioning.** FCPXML is versioned (e.g., **1.10, 1.11, 1.12, 1.13**). Prefer the **latest your target can import** (Final Cut Pro 10.8.1 introduced **1.13**). Resolve frequently exports **1.9/1.10**. Use the **lowest common** version when round-tripping with other NLEs.


---

## 0) Quick Reference: Minimal Valid Files

### 0.1 Minimal timeline with one clip on the primary storyline
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
  <resources>
    <format id="f1080p24" name="FFVideoFormatRateUndefined" width="1920" height="1080" frameDuration="1/24s"/>
    <asset id="a1" name="shot_001.mov" hasVideo="1" hasAudio="1" start="0s" duration="240/24s">
      <media-rep kind="original-media" src="file:///Volumes/Show/shot_001.mov"/>
    </asset>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Example Project">
        <sequence format="f1080p24" tcStart="3600/1s" tcFormat="NDF" duration="240/24s">
          <spine>
            <asset-clip ref="a1" name="shot_001.mov" start="0s" duration="240/24s" offset="3600/1s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

### 0.2 Primary + connected clip (overlay) with a cross dissolve
```xml
<spine>
  <asset-clip ref="a1" start="0s" duration="5s" offset="3600/1s" name="A">
    <!-- Connected overlay above the spine -->
    <clip name="Overlay" lane="2" start="0s" duration="5s" offset="0/1s" enabled="1">
      <video ref="a2" start="0s" duration="5s"/>
    </clip>
  </asset-clip>

  <!-- Transition overlaps tail of A and head of B -->
  <transition name="Cross Dissolve" duration="1s"/>

  <asset-clip ref="a3" start="0s" duration="5s" name="B"/>
</spine>
```

---

## 1) Anatomy of an FCPXML Document

```
fcpxml
├─ resources          # Global registry of things referenced by ID
│  ├─ format          # Video format presets / sequence formats
│  ├─ asset           # Media references (files, duration, channels)
│  ├─ effect?         # (Optional) Motion template references for titles/generators/effects
│  └─ media?          # (Optional) defines multicam or compound media by ID
└─ library
   └─ event
      └─ project
         └─ sequence  # A timeline (project) — defines tcStart, format, duration
            └─ spine  # The primary storyline (no “tracks” in FCP)
               ├─ asset-clip / clip / sync-clip / ref-clip / multicam clip
               ├─ title / caption / gap / audition
               └─ transition
```

**Key properties** appear on elements as XML attributes:
- **`id`**: unique within the file (on resources).
- **`ref`**: references a resource by `id` (clips, titles, filters).
- **`start`**: source in-point **within the referenced media**.
- **`duration`**: length of the instance.
- **`offset`**: timeline position **within the current container** (e.g., sequence start at `tcStart`).
- **`enabled`**: `1` or `0` for visible/audible.
- **`tcFormat`**: `DF` or `NDF` (drop/non-drop). On `sequence` or per-clip if needed.
- **`format`**: references a `<format>` resource (frame rate, raster size, color space).

---

## 2) Time, Units, and Math (crucial)

**Every time-like attribute** (e.g., `start`, `duration`, `offset`, `tcStart`, `frameDuration`) is a **rational number of seconds** with **`s` suffix**:  
- Integer seconds: `5s`  
- Rational frames: `1001/30000s` (≈ 29.97 fps), `1/24s` (24 fps), etc.

**Working rule:** represent frame-accurate positions as **fractions** so you avoid rounding drift.  
**Sequence time zero** is its `tcStart`. Child offsets are **relative to their parent container’s zero**.

**Examples**
- 24 fps timeline: 1 frame = `1/24s`.
- 29.97 timeline: 1 frame = `1001/30000s`.
- A 10-second clip beginning at 01:00:05:00 (24 fps):  
  `offset="3605/1s"` and `duration="240/24s"`.

---

## 3) `resources`: Formats, Assets, Effects

### 3.1 Formats
```xml
<format id="f1" name="FFVideoFormat720p2997" width="1280" height="720" frameDuration="1001/30000s"
        colorSpace="Rec. 709"/>
```
- **`frameDuration`** sets the frame rate.  
- `colorSpace` may be a named preset (e.g., `Rec. 709`, `Rec. 2020`, `Rec. 2020 HLG`, `Rec. 2020 PQ`).  
- Optional fields for stereoscopic/360: `projection`, `stereoscopic`, `heroEye` (when applicable to VR/3D).

### 3.2 Assets (media)
```xml
<asset id="a1" name="clip.mov" hasVideo="1" hasAudio="1"
       start="0s" duration="73073/1875s" audioSources="1" audioChannels="8">
  <media-rep kind="original-media" src="file:///path/to/clip.mov"/>
</asset>
```
- **`start`/`duration`** describe the **available range** of the source media.
- `audioSources`, `audioChannels` describe the file’s audio layout.
- If the **format** is known for the asset, set `format="f1"` to hint raster/framerate.

### 3.3 Effects (Motion templates), Titles & Generators
FCP titles/generators/effects are **Motion templates**. To instantiate a template (e.g., a **Basic Title**), include an **`effect`** in `resources` and **reference it**:
```xml
<resources>
  <effect id="eTitleBasic" name="Basic Title" uid="...optional..." />
</resources>

<!-- Later in the timeline -->
<title ref="eTitleBasic" name="Title Here" start="0s" duration="3s" offset="3600/1s">
  <text>My Title</text>
</title>
```
> If you don’t know the `effect` UID, you can often rely on `name` to match an installed template. Behavior varies by environment; template discovery differs across systems.

---

## 4) `library` → `event` → `project` → `sequence`

### 4.1 Sequence (timeline) header
```xml
<sequence format="f1080p24" tcStart="3600/1s" tcFormat="NDF" duration="240/24s">
  <spine>…</spine>
</sequence>
```
- `format` references the timeline format resource.
- `tcStart="3600/1s"` → 01:00:00:00 timecode start.
- `duration` is the **total timeline length**.

### 4.2 Primary Storyline (`spine`)
A **linear container** that holds the **main narrative**: `asset-clip`, `clip`, `sync-clip`, `title`, `gap`, `transition`, `audition`.

---

## 5) Placing Media — Clips and Connected Clips

### 5.1 Primary clips (`asset-clip`)
```xml
<asset-clip ref="a1" name="Shot A" start="100/24s" duration="240/24s" offset="3600/1s" enabled="1">
  <!-- per-clip transforms or effects here -->
</asset-clip>
```
- `start`/`duration`: **trim** within source `a1`.
- `offset`: position in the current timeline container (here, the spine).

### 5.2 Connected clips / Lanes (`clip` with `lane`)
```xml
<clip name="Overlay" lane="2" start="0s" duration="5s" offset="0/1s">
  <video ref="a2" start="0s" duration="5s"/>
</clip>
```
- A `clip` **inside** an `asset-clip` (or spine) with `lane="2"` sits **above** the primary storyline (like a V2 overlay).
- Connected **audio-only** can be created with `<audio>` (see §7).

### 5.3 Compound or referenced clips
- **`ref-clip`**: reference a clip defined elsewhere.
- **`sync-clip`**: A/V that’s been synchronized (e.g., dual-system audio).
- **`mc-clip` / `multicam`**: see §10.

---

## 6) Edits, Cuts, Transitions

### 6.1 Butt cuts (hard cuts)
Just place adjacent clips with back-to-back `offset` and `duration` values.

### 6.2 Transitions
```xml
<spine>
  <asset-clip ref="a1" duration="5s"/>
  <transition name="Cross Dissolve" duration="1s"/>
  <asset-clip ref="a3" duration="5s"/>
</spine>
```
- A `transition` **overlaps the tail of the previous** element and **head of the next** for `duration`.
- You can add filter parameters within a transition (advanced).

### 6.3 Gap clips (slugs)
```xml
<gap name="Black" duration="1s"/>
```
Use gaps for silence/black or to create timing padding.

---

## 7) Audio: Components, Roles, J/L Cuts, Levels, Pan

### 7.1 Audio components inside a clip
```xml
<audio ref="a1" start="0s" duration="5s" role="dialogue" srcCh="1,2"/>
```
- `srcCh` maps **source channels** used by this component.  
- `role` assigns an **audio role** (e.g., `dialogue`, `effects`, `music`, hierarchical like `dialogue.interview`).

### 7.2 J/L cuts (split edits)
Do split edits by **extending audio independently** on an `asset-clip` with:
```xml
<asset-clip ref="a1" start="10s" duration="4s" audioStart="9s" audioDuration="6s"/>
```
- `audioStart`/`audioDuration` override the audio’s in/out relative to the video range, enabling J/L.

### 7.3 Volume & pan
```xml
<adjust-volume amount="-6dB"/>
<adjust-panner mode="stereo" amount="0.5"/>
```
- Gains are typically in dB; automation requires keyframes via `<param>` (advanced).

### 7.4 Roles on containers
- `asset-clip` supports `audioRole`/`videoRole` defaults.
- Child `audio`/`video` can set their own `role`.

---

## 8) Video Compositing: Transform, Conform, Blend, Crop/Distort

### 8.1 Transform (`adjust-transform`)
```xml
<adjust-transform position="320 180" scale="0.5 0.5" rotation="0" anchor="0 0"/>
```
- Coordinates are **viewer pixels**; anchor modifies the transform origin.

### 8.2 Spatial Conform (`adjust-conform`)
```xml
<adjust-conform type="fit"/>   <!-- fit | fill | none -->
```

### 8.3 Blend/Opacity (`adjust-blend`)
```xml
<adjust-blend amount="0.8" mode="screen"/>
```

### 8.4 Distort/Corners (picture-in-picture, keystone)
```xml
<adjust-corners topLeft="-100 -50" topRight="100 -50" botLeft="-100 50" botRight="100 50"/>
```

### 8.5 Stabilization, Rolling Shutter
```xml
<adjust-stabilization type="automatic"/>
<adjust-rollingShutter amount="low"/>
```

### 8.6 Orientation / 360 / VR
```xml
<adjust-orientation tilt="0" pan="0" roll="0" fieldOfView="90"/>
<adjust-360-transform coordinates="spherical" latitude="10" longitude="15"/>
```

---

## 9) Titles, Captions, and Text

### 9.1 Title (Motion template)
```xml
<title ref="eTitleBasic" name="Lower Third" start="0s" duration="3s" offset="3605/1s">
  <text>Speaker Name</text>
  <text-style-def id="ts1"><text-style font="Helvetica" fontSize="72"/></text-style-def>
</title>
```

### 9.2 Captions
- **CEA-608/708** and **iTT** captions are supported. A simple iTT-style example:
```xml
<caption role="subtitle" start="0s" duration="3s">
  <text>Welcome to the show.</text>
</caption>
```

---

## 10) Multicam & Sync

### 10.1 Multicam resources
```xml
<resources>
  <media id="m1">
    <multicam angleSet="Cameras">
      <!-- angles defined inside the multicam -->
    </multicam>
  </media>
</resources>
```
Use **`mc-clip`** or `multicam` with **`mc-angle`** in the timeline to pick angles. Angle changes appear as cut points on the spine. For simpler dual-system workflows, use **`sync-clip`**.

---

## 11) Retiming (Speed Changes) and Optical Flow

### 11.1 Constant-speed retime via `timeMap`
```xml
<video ref="a1" start="0s" duration="10s">
  <timeMap preservesPitch="1">
    <timept time="0s"  value="0s"  interp="linear"/>
    <timept time="10s" value="5s"  interp="linear"/>
  </timeMap>
</video>
```
This **stretches** 5 seconds of source over 10 seconds of output (**50% speed**).  
To reverse, make `value` decrease over time (e.g., `value="5s"` → `0s`).  
You can choose sampling methods like `frame-blending` or `optical-flow` at the `timeMap` level (advanced).

### 11.2 Conform frame rate
```xml
<conform-rate srcFrameRate="29.97" frameSampling="optical-flow"/>
```

---

## 12) Markers, Keywords, Ratings, Notes

```xml
<marker start="12/24s" value="Cut Here" duration="0s"/>
<keyword start="0s" duration="10s" value="Interview, Favorite"/>
<rating value="favorite" start="0s" duration="10s"/>
<note>Any free-form note</note>
```

Use **markers** for edit navigation and automation. **Keywords** tag ranges. **Ratings** can be `favorite` or `reject`.

---

## 13) Color Management, Color Spaces, and LUTs

- Specify `colorSpace` at the **format** or override per **asset** via `colorSpaceOverride`.
- Some workflows use `customLUTOverride` to point to a LUT identifier managed by FCP.  
- HDR projects may use `Rec. 2020 HLG` or `Rec. 2020 PQ`. Always match delivery.

---

## 14) Roles (Audio/Video Organization)

Roles classify media for **organization and export** (e.g., role-based stems):
- **Primary types:** `dialogue`, `effects`, `music`, `titles`, `video`.
- **Hierarchical:** `dialogue.interview`, `dialogue.vo`, etc.
- Set on containers (`asset-clip` with `audioRole`/`videoRole`) and children (`audio role="..."`, `video role="..."`).

---

## 15) Bundles vs. Single Files

FCP can package an XML plus referenced sidecars as a **bundle** (`.fcpxmld`) or export a standalone `.fcpxml`. When interchanging with Resolve/Premiere, use **`.fcpxml`** unless your target supports bundles. Keep **relative paths** and source media **reachable**.

---

## 16) Validation, Versioning, and Interchange Tips

- The **`version`** on the root `<fcpxml>` must match what your target FCP can import (e.g., `1.10`, `1.12`, `1.13`).  
- Resolve typically **imports/exports** 1.9–1.10 reliably.  
- When you see **“DTD validation failed”**, check:
  1) Element/attribute spelling,  
  2) Required attributes present (`ref`, `id`, `start/duration/offset`),  
  3) Rational time values with `s` suffix,  
  4) Version mismatches (newer XML into older FCP),  
  5) Missing `resources` referenced by `ref`.

**Round-trip strategy**
- Author to **the lowest version** needed by all tools in the pipeline.  
- Normalize channel mappings and roles early (`srcCh`, `audioSources`, etc.).  
- Prefer **deterministic frame fractions** (never float seconds).

---

## 17) Cookbook: Common Tasks

### 17.1 Cut a long clip into three shots on the spine
```xml
<spine>
  <asset-clip ref="a1" start="0s"    duration="5s" offset="3600/1s"/>
  <asset-clip ref="a1" start="5s"    duration="4s" />
  <asset-clip ref="a1" start="10s"   duration="6s" />
</spine>
```

### 17.2 Add a picture-in-picture overlay at top-right
```xml
<clip lane="2" start="0s" duration="6s" offset="0/1s" enabled="1">
  <video ref="a2" start="0s" duration="6s"/>
  <adjust-transform position="1660 140" scale="0.35 0.35" anchor="0 0"/>
</clip>
```

### 17.3 J-cut (audio leads picture by 12 frames at 24 fps)
```xml
<asset-clip ref="a3" start="20/24s" duration="120/24s" audioStart="8/24s" audioDuration="132/24s"/>
```

### 17.4 Fade-in from black (video) and fade-up audio
Use a **transition** (or keyframe opacity via filters) and **volume ramp**:
```xml
<asset-clip ref="a1" start="0s" duration="5s">
  <adjust-blend amount="0.0"/>
  <!-- then animate amount to 1.0 with <param> keyframes (advanced) -->
</asset-clip>
```
Or simply place a **`transition name="Cross Dissolve"`** at head/tail.

### 17.5 Cross dissolve between shots
```xml
<asset-clip ref="a1" duration="5s"/>
<transition name="Cross Dissolve" duration="12/24s"/>
<asset-clip ref="a2" duration="4s"/>
```

### 17.6 Lower-third title
```xml
<title ref="eTitleBasic" name="LT" start="0s" duration="3s" offset="3610/1s">
  <text>Speaker Name</text>
  <text-style-def id="ts1"><text-style font="SF Pro" fontSize="72"/></text-style-def>
  <adjust-transform position="120 920"/>
</title>
```

### 17.7 Basic retime (50% slow)
```xml
<video ref="a1" start="0s" duration="10s">
  <timeMap preservesPitch="1">
    <timept time="0s"  value="0s"  interp="linear"/>
    <timept time="10s" value="5s"  interp="linear"/>
  </timeMap>
</video>
```

### 17.8 Insert markers and keywords
```xml
<marker start="48/24s" value="Action"/>
<keyword start="48/24s" duration="5s" value="B-Roll"/>
```

### 17.9 Multicam cut from Angle A to B at 10s
*(Skeleton — angle selection is represented by cuts)*
```xml
<spine>
  <mc-clip ref="m1" start="0s" duration="10s"/>
  <mc-clip ref="m1" start="10s" duration="5s"/>
</spine>
```

---

## 18) Robustness Rules for Generators/LLMs

1. **Always declare**: `<?xml ...?>`, `<!DOCTYPE fcpxml>`, `<fcpxml version="X.YY">`.
2. **Populate `resources` first**, then reference by `ref`.
3. **Use rational seconds** with `s` suffix for every time value.
4. **Compute `offset` precisely**: it is **parent-relative**, not global.
5. **Keep IDs unique** and **stable** across passes.
6. **Never reference missing IDs**; validate after generation.
7. **Prefer known template names** for titles/transitions; include `<effect>` when needed.
8. **Normalize roles** (`audioRole`, `videoRole`, child `role`) early.
9. **Choose version** based on the importing app; **downshift** if round-tripping with Resolve.
10. **Test-import** into Final Cut Pro; re-export to verify structural integrity.

---

## 19) Troubleshooting Cheatsheet

- **“DTD validation failed / not the correct format”**  
  → Version mismatch or missing required attributes; verify `version=`, `ref=`, rational times.

- **Audio out of sync after retime**  
  → Confirm `preservesPitch` and `timeMap` interpolation; avoid fractional-frame drift.

- **Transition didn’t apply**  
  → Transition must sit **between two story elements** and its `duration` must be positive.

- **Title template missing**  
  → Ensure referenced `effect` exists on the target system; consider using generic names.

- **Connected clip not visible**  
  → Check `lane` (overlay above spine) and `adjust-blend`/opacity.

---

## 20) Reference Snippets You Can Reuse

### 20.1 Complete skeleton with most elements
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
  <resources>
    <format id="f1" name="FFVideoFormat1080p24" width="1920" height="1080" frameDuration="1/24s" colorSpace="Rec. 709"/>
    <asset  id="a1" name="A.mov" hasVideo="1" hasAudio="1" start="0s" duration="12s">
      <media-rep kind="original-media" src="file:///path/A.mov"/>
    </asset>
    <asset  id="a2" name="B.mov" hasVideo="1" hasAudio="1" start="0s" duration="12s">
      <media-rep kind="original-media" src="file:///path/B.mov"/>
    </asset>
    <effect id="eTitleBasic" name="Basic Title"/>
  </resources>

  <library>
    <event name="Event">
      <project name="Project">
        <sequence format="f1" tcStart="3600/1s" tcFormat="NDF" duration="24s">
          <spine>
            <asset-clip ref="a1" name="A" start="0s" duration="10s" offset="3600/1s">
              <adjust-transform position="0 0" scale="1 1" rotation="0" anchor="0 0"/>
              <adjust-volume amount="-6dB"/>
              <clip name="Overlay" lane="2" start="0s" duration="5s" offset="0/1s">
                <video ref="a2" start="0s" duration="5s"/>
              </clip>
            </asset-clip>

            <transition name="Cross Dissolve" duration="1s"/>

            <asset-clip ref="a2" name="B" start="0s" duration="10s">
              <adjust-conform type="fit"/>
            </asset-clip>

            <title ref="eTitleBasic" name="LT" start="0s" duration="3s" offset="3618/1s">
              <text>Hello!</text>
              <text-style-def id="ts1"><text-style font="Helvetica" fontSize="72"/></text-style-def>
            </title>

            <caption role="subtitle" start="0s" duration="3s">
              <text>Subtitle text.</text>
            </caption>

            <marker start="48/24s" value="Note"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

---

## 21) Practical Notes

- **Connected clip timing**: `offset` inside a child `clip` is **relative to its parent container’s local zero** (commonly the parent `asset-clip` start on the spine).  
- **Audio J/L**: `audioStart/audioDuration` live on the **`asset-clip`**, not the child `audio` element.  
- **Transitions**: Only legal **between** story elements; duration must be <= heads/tails.  
- **Retime**: Prefer `timeMap` for ramps; `conform-rate` for frame-rate matching.  
- **Imports from Resolve**: Expect additional `clip` wrappers and lane usage; versions commonly **1.9–1.10**.  
- **IDs**: Treat as **stable** so that later passes can patch/extend the same XML reliably.

---

## 22) Suggested LLM Prompts for Code Generators

- “Given a list of cuts with absolute timeline positions (seconds), generate `asset-clip` elements with correct `offset` and `duration` for FCPXML 1.10. Use `1/24s` fractions for a 24 fps sequence.”  
- “Create a connected `clip` in `lane=2` that overlays B-roll with a transform (scale 0.35, top-right corner).”  
- “Insert cross dissolves (`transition` with `name="Cross Dissolve"`) of 12 frames between each adjacent primary `asset-clip`.”  
- “Produce an L-cut: set `audioStart` 12 frames earlier and `audioDuration` 12 frames longer than the video range.”  
- “Emit `timeMap` for 50% slow motion over the range [t0, t1] using linear interpolation.”

---

## 23) References & Further Reading

- Apple — **Final Cut Pro release notes** (versioning, FCPXML updates).  
- Apple — **FCPXML Reference / Creating FCPXML / Bundle Reference** (schema overview, bundles).  
- Apple — **Legacy FCPXML DTDs** (older but useful for semantics; see `timeMap`/`timept` examples).  
- FCP Cafe — **Demystifying Final Cut Pro XMLs**, curated FCPXML resources.

*(Keep these URLs with the repo so LLMs can find them.)*
