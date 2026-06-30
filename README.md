# Virtual Marble Run

An interactive projector game for kids. A camera watches a wall; whatever the
kids stick, draw, or place on it (sticky notes, paper shapes, toys, hands) is
detected by background differencing and turned into collision geometry. Marbles
drop continuously from the top center and bounce/roll off the detected objects
with real 2D physics until they leave the play area.

## Disclaimer

This project is provided "as is", without warranty of any kind. By using this
software, you agree that you do so at your own risk. The authors and
contributors take no liability for any damages, losses, injuries, or other
claims arising from use, misuse, or inability to use this project.

## How it works

```
 Camera ─▶ Vision (background diff ─▶ contours)  ─▶ Homography warp ─▶ Physics ─▶ Projector
 (thread)        (thread, ~10 Hz)     camera-space        cam→display     (pymunk)   (pygame, 60 Hz)
```

- **Camera** ([`marblerun/camera.py`](marblerun/camera.py)) reads frames on a
  background thread so rendering never stalls.
- **Vision** ([`marblerun/vision.py`](marblerun/vision.py)) snapshots the empty
  wall once, then flags any region that differs from it as an object and traces
  its outline. A *fixed* reference background is used on purpose so objects that
  sit still don't fade away.
- **Calibration** ([`marblerun/calibration.py`](marblerun/calibration.py))
  builds a homography mapping camera pixels to projector pixels, so a detected
  object lines up with where the marble actually bounces.
- **Physics** ([`marblerun/physics.py`](marblerun/physics.py)) turns each object
  outline into static segments and drops dynamic marbles that collide with them.
- **Display** ([`marblerun/display.py`](marblerun/display.py)) renders the
  fullscreen projector image.

## Requirements

- Python 3.10+ (tested on 3.11)
- A webcam and a projector
- Dependencies in [`requirements.txt`](requirements.txt)

```bash
python -m pip install -r requirements.txt
# If you hit a permissions error writing to the system Scripts folder:
python -m pip install --user -r requirements.txt
```

## Hardware setup

1. Point the projector at a flat wall and project the game image onto it.
2. Place the camera so it sees the *entire* projected area (roughly head-on is
   best; extreme angles reduce accuracy).
3. Keep the camera and projector still once calibrated. Moving either one
   requires recalibrating.
4. Steady, even lighting helps differencing. Avoid the projector light washing
   out the camera's view of objects if possible (dimmer projector content / a
   reasonably lit room is a good balance).

## Running

```bash
python main.py             # fullscreen on the projector
python main.py --windowed  # windowed, useful while setting up
python main.py --check      # offline self-test, no camera/projector needed
```

### First-run sequence

Calibration happens entirely in the one projector window - no separate camera
window:

1. Launch the game. If there's no saved calibration it enters **calibration**
   mode: it briefly projects a 3x3 grid of numbered registration marks, captures
   a single still frame of them, then shows that frozen camera image inside the
   same window.
2. **Click each numbered mark in order** on that image. A **magnifier loupe**
   follows your cursor for pixel-precise clicks; the **arrow keys nudge** the
   most recent point, **R** undoes the last point, and **G** re-grabs a fresh
   frame if needed. Press **Enter** to confirm. Using 9 marks (instead of 4
   corners) lets the alignment be solved by least squares, so it is far more
   accurate. The resulting reprojection error (in pixels) is shown in the HUD - a
   few px is excellent. The mapping is saved to `calibration.json` for next time.
3. Put some tape on the wall, then press **L** to learn its color: a frozen
   camera frame appears; click a few spots on the tape (the green overlay
   previews what will be detected), then press **Enter**. This trains detection
   on the tape as the camera actually sees it under your projector light, which
   is the most reliable option. (No background capture is needed for `learned`
   or `tape` modes; for the background-based modes, press **C** with the wall
   empty instead.)
4. Marbles will begin bouncing off the tape. Add/move tape freely; press **L**
   again any time to relearn if the lighting changes.

## Controls

| Key | Action |
| --- | --- |
| `C` | Capture/refresh the background (do on an empty wall) |
| `K` | Re-run calibration (registration grid) |
| `L` | Learn the tape color (sample it by clicking, with live preview) |
| `O` | Toggle the object-outline overlay (what physics sees) |
| `M` | Cycle detection mode (learned / tape / combo / lab / saturation / gray) |
| `D` | Toggle the camera difference (mask) debug window |
| `X` | Clear all marbles |
| `H` | Hide/show the on-screen info text |
| `G` | Lock/freeze object geometry (physics keeps running, no new objects) |
| `P` / `Space` | Pause/resume physics and marble spawning |
| `B` | Cycle marble color scheme (Rainbow / Pinks & Purples / Reds & Yellows / Black & White) |
| `F11` | Toggle fullscreen |
| `Esc` / `Q` | Quit |

## Tuning

All knobs live in [`config.py`](config.py). The most useful:

- `DETECTION_MODE` - how objects are found. Press **M** at runtime to cycle:
  - `learned` (default) - detects the actual tape color(s) you sampled with the
    **L** step. Most robust under real projector lighting because it is trained
    on exactly what the camera sees. Needs no background.
  - `tape` - detects fixed green/blue hues. Needs no background.
  - `combo` - color background diff + saturation (background-based).
  - `lab` - color background diff (detects different color at similar brightness).
  - `saturation` - finds any colorful region directly; needs no background.
  - `gray` - original grayscale diff (weak for colored objects on grey walls).
- Learned colors are saved to `tape_colors.json` and reloaded on launch. Tuning
  knobs: `LEARN_H_MARGIN` (hue width around a sample), `LEARN_S_MARGIN` /
  `LEARN_V_MARGIN` (how much darker/less-saturated still counts),
  `LEARN_S_FLOOR` / `LEARN_V_FLOOR` (hard floors that keep the grey wall out),
  and `LEARN_HUE_CLUSTER` (samples within this many hue degrees merge into one
  band). The **L** preview overlay (green) shows exactly what will be detected.
- `TAPE_HUE_RANGES` - which colors `tape` mode detects, as OpenCV HSV hue ranges
  (H is 0..179). Defaults: green `(35,85)` and blue `(90,130)`.
- `TAPE_MIN_SATURATION` / `TAPE_MIN_VALUE` - raise to reject washed-out or dark
  pixels; lower if vivid tape is being missed.
- `COLOR_DIFF_THRESHOLD` / `LAB_L_WEIGHT` - sensitivity of the color diff; lower
  `LAB_L_WEIGHT` makes detection care more about color than brightness.
- `SATURATION_THRESHOLD` - lower detects less-vivid tape; raise to ignore the
  faintly-colored wall.
- `DIFF_THRESHOLD` - lower detects fainter objects but picks up more noise.
- `MIN_OBJECT_AREA` - raise to ignore small specks; lower to detect tiny items.
- `MORPH_KERNEL` - larger smooths/merges blobs and removes speckle.
- `CONTOUR_EPSILON_FRAC` - higher = coarser outlines (cheaper physics).
- `SPAWN_INTERVAL` / `MAX_MARBLES` - how many marbles and how often.
- `OBJECT_SOLID` - `True` builds detected shapes as solid filled bodies (via
  convex decomposition) so marbles can never get stuck inside them; `False` makes
  them hollow outlines. `SOLID_DECOMP_TOLERANCE` trades accuracy for fewer pieces.
- `GRAVITY`, `MARBLE_ELASTICITY`, `OBJECT_ELASTICITY` - feel of the bouncing.
- `SIDE_WALLS` - if `True`, marbles bounce off screen edges instead of falling off.
- `CAMERA_INDEX` / `CAMERA_BACKEND` - change if the wrong camera opens.

## Troubleshooting

- **Wrong camera opens**: change `CAMERA_INDEX` in `config.py` (try 1, 2, ...).
- **Objects don't line up with bounces**: recalibrate (`K`) and watch the
  reprojection error in the HUD - click each mark carefully using the magnifier
  and arrow-key nudge until it reads only a few pixels. Make sure neither the
  camera nor projector moved since calibrating.
- **Everything is detected as an object / lots of noise**: re-capture the
  background (`C`) on an empty wall, raise `DIFF_THRESHOLD`, or raise
  `MIN_OBJECT_AREA`. Use `D` to inspect the mask.
- **Objects fade away after sitting still**: this build uses a fixed background
  on purpose, so that shouldn't happen; if it does, you likely re-captured the
  background with objects present - press `C` again on an empty wall.
- **Marbles tunnel through thin objects**: increase `SEGMENT_THICKNESS` or
  `PHYSICS_SUBSTEPS`, or reduce `GRAVITY`.

## Project layout

```
MarbleRun/
├── main.py              orchestration + state machine + offline self-test
├── config.py            all tunables
├── requirements.txt
├── README.md
├── calibration.json     created after first calibration
├── tape_colors.json     created after learning tape color (L)
└── marblerun/
    ├── camera.py        threaded webcam capture
    ├── calibration.py   4-corner homography (compute/save/load/warp)
    ├── vision.py        background differencing + contour detection
    ├── physics.py       pymunk world, marbles, static object outlines
    └── display.py       pygame fullscreen rendering + overlays
```

Thank you!