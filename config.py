"""Central configuration for Virtual Marble Run.

Every tunable lives here so you can tweak behaviour without touching logic.
Values are intentionally conservative; adjust to your room, camera and projector.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "calibration.json")
TAPE_COLOR_FILE = os.path.join(PROJECT_ROOT, "tape_colors.json")

# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
# Index passed to cv2.VideoCapture. 0 is usually the built-in/first webcam.
CAMERA_INDEX = 0
# Requested capture resolution. The driver may pick the closest supported mode.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
# On Windows the DirectShow backend tends to open faster and more reliably.
# Set to None to let OpenCV choose automatically.
CAMERA_BACKEND = "dshow"  # one of: "dshow", "msmf", None

# --------------------------------------------------------------------------- #
# Display / projector
# --------------------------------------------------------------------------- #
# Logical canvas the physics world and renderer use (the projector's native res).
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
# Start the projector window in fullscreen. Toggle with F11 at runtime.
FULLSCREEN = True
TARGET_FPS = 60
BACKGROUND_COLOR = (12, 12, 20)

# --------------------------------------------------------------------------- #
# Vision: background differencing + contour extraction
# --------------------------------------------------------------------------- #
# How often the vision thread re-analyses the wall (frames per second).
VISION_FPS = 10
# Gaussian blur kernel applied before differencing to suppress sensor noise.
BLUR_KERNEL = 5

# --- Detection mode -------------------------------------------------------- #
# Which signal(s) to use to find objects:
#   "gray"       - grayscale background diff (original; weak for same-brightness
#                  colored objects on grey walls)
#   "lab"        - color background diff in CIE Lab (detects "different color,
#                  same brightness"; great for colored tape on drywall)
#   "saturation" - find colorful regions directly (colored tape is saturated,
#                  grey drywall is not). Needs no background.
#   "tape"       - detect fixed colors (green/blue tape) by hue. Needs no
#                  background, so peeling tape off never leaves a phantom object.
#   "learned"    - detect the actual tape color(s) sampled during setup (press
#                  L to learn). Most robust under real projector lighting because
#                  it is trained on what the camera actually sees.
#   "combo"      - union of "lab" and "saturation" (background-based)
DETECTION_MODE = "learned"

# --- "tape" mode: which colors count -------------------------------------- #
# OpenCV HSV hue is 0..179. Each (low, high) range is a color to detect; the
# union of all ranges is what registers. Defaults target green and blue tape.
TAPE_HUE_RANGES = [
    (35, 85),    # green
    (90, 130),   # blue
]
# Pixels must be at least this saturated / bright to count (rejects grey wall,
# shadows and washed-out highlights).
TAPE_MIN_SATURATION = 70
TAPE_MIN_VALUE = 50

# --- "learned" mode: tape colors sampled during setup --------------------- #
# Filled in by the L (learn) step or loaded from TAPE_COLOR_FILE. Each entry is
# an HSV inRange band [h_lo, h_hi, s_min, v_min] (hue 0..179). Empty until learnt.
LEARNED_COLORS: list = []
# Patch size (px) sampled around each click when learning a color.
LEARN_PATCH = 9
# Margins applied around sampled values to build the detection band.
LEARN_H_MARGIN = 12      # hue half-width (+/-) around the sample
LEARN_S_MARGIN = 70      # how far below the sample's saturation still counts
LEARN_V_MARGIN = 70      # how far below the sample's value still counts
LEARN_S_FLOOR = 45       # never accept pixels less saturated than this (wall)
LEARN_V_FLOOR = 35       # never accept pixels darker than this
# Samples whose hue is within this many degrees are merged into one band.
LEARN_HUE_CLUSTER = 16

# Grayscale-diff threshold (used by "gray").
DIFF_THRESHOLD = 30
# Color-diff threshold for the Lab distance (used by "lab"/"combo").
COLOR_DIFF_THRESHOLD = 22
# How much the lightness (L) channel counts vs. color (a/b). <1 makes detection
# care more about hue/color than brightness, so shadows/projector light matter
# less. Range ~0..1.
LAB_L_WEIGHT = 0.4
# Saturation detection (used by "saturation"/"combo"), HSV S and V are 0..255.
SATURATION_THRESHOLD = 55   # pixels more saturated than this are "colorful"
SATURATION_MIN_VALUE = 45   # ignore very dark pixels (noisy hue/saturation)
# Morphology kernel size (px) used to clean the foreground mask.
MORPH_KERNEL = 5
# Contours smaller than this area (in camera pixels) are ignored as noise.
MIN_OBJECT_AREA = 800
# Douglas-Peucker simplification epsilon as a fraction of contour perimeter.
# Larger -> fewer, coarser points -> cheaper physics, less detail.
CONTOUR_EPSILON_FRAC = 0.01
# A detected change is only pushed to physics if the object set moved/grew by
# more than this fraction of total outline length (debounce against jitter).
REBUILD_CHANGE_FRAC = 0.04

# --------------------------------------------------------------------------- #
# Physics (pymunk). Units are display pixels; gravity is px/s^2.
# --------------------------------------------------------------------------- #
GRAVITY = 1600.0
PHYSICS_SUBSTEPS = 2  # sub-steps per rendered frame for stability
MARBLE_RADIUS = 14.0
MARBLE_MASS = 1.0
MARBLE_FRICTION = 0.6
MARBLE_ELASTICITY = 0.55  # bounciness 0..1
OBJECT_FRICTION = 0.7
OBJECT_ELASTICITY = 0.6
# Static object outlines are inflated by the marble radius? No - we just give
# the segments a small thickness so fast marbles do not tunnel through.
SEGMENT_THICKNESS = 3.0

# --- Collision geometry source -------------------------------------------- #
# When True, the detected white pixels (the mask) are warped into display space
# and used directly as the hard surfaces marbles hit (pixel-accurate edges).
# When False, the older coarse-polygon path is used instead.
COLLISION_FROM_MASK = True
# Contour simplification for mask surfaces, in display pixels. 0 = full detail
# (most accurate, most segments); 2-4 keeps the shape while cutting segment
# count for performance.
MASK_COLLISION_EPSILON = 2.0
# Ignore mask blobs smaller than this (in display-space pixels) as noise.
MASK_MIN_OBJECT_AREA = 1200
# When True, detected shapes are built as SOLID filled bodies (triangulated)
# so marbles can never get trapped inside them. When False, shapes are hollow
# outlines (segments) that balls can fall into.
OBJECT_SOLID = True
# Reserved tolerance knob for solid-shape building (display px). Coarser contour
# simplification is controlled by MASK_COLLISION_EPSILON above.
SOLID_DECOMP_TOLERANCE = 2.0
# Safety caps so noisy/complex detections can never explode the physics world
# (a common cause of slow-downs and instability over long runs).
MAX_STATIC_SHAPES = 1500   # total collision shapes across all objects
MAX_CONTOUR_VERTS = 120    # vertices per detected shape (simplified beyond this)

# --------------------------------------------------------------------------- #
# Marble spawning + lifecycle
# --------------------------------------------------------------------------- #
# Seconds between marble drops. Lower = more marbles.
SPAWN_INTERVAL = 0.45
# Marbles drop in across this fraction of the play-area width, centered on the
# top-center. e.g. 0.40 = spread over the middle 40% of the width.
SPAWN_WIDTH_FRAC = 0.40
# Vertical start position (px from top).
SPAWN_Y = -20
# Small random initial horizontal velocity range (px/s).
SPAWN_VX_RANGE = 60
# Hard cap so the simulation never explodes with too many bodies.
MAX_MARBLES = 250
# Margin (px) beyond the screen edges before a marble is culled.
CULL_MARGIN = 120
# Optional invisible walls on the left/right edges. If False, marbles fall off
# the sides and are culled. If True, they bounce off the screen edges.
SIDE_WALLS = False

# Selectable marble color schemes, cycled at runtime with the B key.
MARBLE_SCHEMES = (
    ("Rainbow", (
        (255, 99, 132),
        (54, 162, 235),
        (255, 206, 86),
        (75, 192, 192),
        (153, 102, 255),
        (255, 159, 64),
        (46, 204, 113),
    )),
    ("Pinks and Purples", (
        (255, 105, 180),
        (255, 145, 200),
        (218, 112, 214),
        (186, 85, 211),
        (147, 112, 219),
        (138, 43, 226),
        (199, 21, 133),
    )),
    ("Reds and Yellows", (
        (255, 40, 40),
        (255, 80, 0),
        (255, 140, 0),
        (255, 180, 0),
        (255, 215, 0),
        (255, 240, 90),
        (220, 20, 60),
    )),
    ("Black and White", (
        (245, 245, 245),
        (30, 30, 30),
        (200, 200, 200),
        (70, 70, 70),
        (255, 255, 255),
        (120, 120, 120),
    )),
)
# Which scheme to start with (index into MARBLE_SCHEMES). 0 = Rainbow.
MARBLE_SCHEME_INDEX = 0
# The active palette (kept in sync when the scheme is cycled).
MARBLE_COLORS = MARBLE_SCHEMES[MARBLE_SCHEME_INDEX][1]
