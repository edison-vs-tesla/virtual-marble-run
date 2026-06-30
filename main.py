"""Virtual Marble Run - entry point and orchestration.

Pipeline each frame (RUN state):
  camera (thread) -> vision (thread): background diff -> camera-space polygons
  -> warp through homography H -> display-space polygons
  -> physics static geometry -> marbles fall/bounce -> render on projector.

Run it:
    python main.py            # launch the game (projector + camera)
    python main.py --windowed # same, but in a window (handy for setup)
    python main.py --check     # offline self-test, no camera/projector needed

Controls (while running):
    C        capture/refresh background (do this on an EMPTY wall first!)
    K        (re)calibrate projector-to-camera mapping
    L        learn the tape color
    O        toggle object-outline overlay (shows what physics sees)
    D        toggle the camera difference debug window
    M        cycle detection mode
    X        clear all marbles
    H        hide/show the on-screen info text
    G        lock/freeze object geometry (physics keeps running)
    P / Space pause/resume physics and spawning
    B        cycle marble color scheme
    F11      toggle fullscreen
    ESC / Q  quit
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import sys
import threading
import time

import numpy as np

import config
from marblerun import calibration

log = logging.getLogger("marblerun")


def setup_logging() -> None:
    """Log to console and a file, and capture crashes that have no traceback.

    - Python exceptions (main thread and worker threads) are written to
      `marblerun.log`.
    - `faulthandler` writes a low-level stack to `marblerun_fault.log` even for
      native crashes (segfaults / aborts inside C libraries), which otherwise
      produce no error message at all.
    """
    log_path = os.path.join(config.PROJECT_ROOT, "marblerun.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_path)],
    )

    fault_path = os.path.join(config.PROJECT_ROOT, "marblerun_fault.log")
    # Kept open for the process lifetime so the handler can write during a crash.
    fault_file = open(fault_path, "w")
    faulthandler.enable(file=fault_file, all_threads=True)

    def _excepthook(exc_type, exc, tb):
        log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        log.critical("Uncaught thread exception in %s",
                     args.thread.name if args.thread else "?",
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    threading.excepthook = _thread_excepthook
    log.info("Logging to %s ; fault log %s", log_path, fault_path)


# --------------------------------------------------------------------------- #
# Live game
# --------------------------------------------------------------------------- #
def run_game(windowed: bool = False) -> int:
    import cv2

    from marblerun import vision as vision_mod
    from marblerun.camera import CameraThread
    from marblerun.display import Display
    from marblerun.physics import World
    from marblerun.vision import VisionThread

    display = Display(fullscreen=not windowed and config.FULLSCREEN)
    camera = CameraThread().start()
    vision = VisionThread(camera).start()
    world = World()

    H = calibration.load_homography()
    calibrated = H is not None
    if H is None:
        # Fallback so detections still reach physics before calibration.
        H = calibration.default_homography(camera.resolution)
    calib_error: float | None = None
    grid_points, grid_labels = calibration.display_grid_points()

    # Load any previously learned tape colors so "learned" mode works on launch.
    saved_colors = vision_mod.load_learned_colors()
    if saved_colors:
        config.LEARNED_COLORS = saved_colors

    show_outlines = True
    show_diff = False
    show_hud = True          # H toggles the on-screen info text
    geometry_locked = False  # G freezes detected objects (physics keeps running)
    paused = False           # P pauses physics + spawning
    scheme_index = config.MARBLE_SCHEME_INDEX  # B cycles marble color schemes
    world.set_palette(config.MARBLE_SCHEMES[scheme_index][1])
    last_vision_version = -1
    spawn_accum = 0.0
    empty_streak = 0  # consecutive empty detections (hysteresis vs. flicker)

    def _grab_reference():
        """Project the registration marks and capture one frozen camera frame.

        Returns a fresh frame that contains the projected marks on the wall, so
        we can let the operator click them inside this same window (no second
        window, no feedback loop since the captured frame is a still image).
        """
        import pygame
        display.clear()
        display.draw_calibration_markers(grid_points, grid_labels)
        display.draw_center_text(["CALIBRATION", "Capturing the projected marks..."])
        display.flip()
        pygame.event.pump()
        time.sleep(0.5)  # let the projector paint and the camera expose
        frame = None
        for _ in range(8):  # flush stale buffered frames
            _id, f = camera.read()
            if f is not None:
                frame = f
            time.sleep(0.03)
        return frame

    def do_calibration() -> None:
        """Unified in-window calibration: capture marks, then click them here."""
        nonlocal H, calibrated, calib_error
        import pygame

        n = len(grid_points)
        frame = _grab_reference()
        if frame is None:
            return
        cam_h, cam_w = frame.shape[:2]
        frame_surf = display.frame_to_surface(frame)

        # Letterbox the camera image into the projector window and remember the
        # transform so window clicks can be mapped back to camera pixels.
        scale = min(display.width / cam_w, display.height / cam_h)
        disp_w, disp_h = int(cam_w * scale), int(cam_h * scale)
        ox = (display.width - disp_w) // 2
        oy = (display.height - disp_h) // 2
        fitted = pygame.transform.smoothscale(frame_surf, (disp_w, disp_h))

        def win_to_cam(mx, my):
            cx = (mx - ox) / scale
            cy = (my - oy) / scale
            return [max(0.0, min(cam_w - 1, cx)), max(0.0, min(cam_h - 1, cy))]

        clicks: list[list[float]] = []
        pygame.mouse.set_visible(True)
        while True:
            display.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if len(clicks) < n:
                        clicks.append(win_to_cam(*event.pos))
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return  # cancel, keep previous calibration
                    if event.key in (pygame.K_r, pygame.K_BACKSPACE) and clicks:
                        clicks.pop()
                    elif event.key == pygame.K_g:
                        f2 = _grab_reference()
                        if f2 is not None:
                            frame_surf = display.frame_to_surface(f2)
                            fitted = pygame.transform.smoothscale(
                                frame_surf, (disp_w, disp_h))
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and len(clicks) == n:
                        cam_pts = np.array(clicks, dtype=np.float32)
                        Hn = calibration.compute_homography(cam_pts, grid_points)
                        calib_error = calibration.reprojection_error(Hn, cam_pts, grid_points)
                        calibration.save_homography(Hn)
                        H = Hn
                        calibrated = True
                        return
                    elif clicks and event.key == pygame.K_LEFT:
                        clicks[-1][0] -= 1
                    elif clicks and event.key == pygame.K_RIGHT:
                        clicks[-1][0] += 1
                    elif clicks and event.key == pygame.K_UP:
                        clicks[-1][1] -= 1
                    elif clicks and event.key == pygame.K_DOWN:
                        clicks[-1][1] += 1

            display.screen.fill((0, 0, 0))
            display.screen.blit(fitted, (ox, oy))
            for i, (cx, cy) in enumerate(clicks):
                wx, wy = int(ox + cx * scale), int(oy + cy * scale)
                pygame.draw.circle(display.screen, (0, 255, 0), (wx, wy), 7)
                pygame.draw.circle(display.screen, (0, 255, 0), (wx, wy), 13, 1)
                lbl = display.font.render(grid_labels[i], True, (0, 255, 0))
                display.screen.blit(lbl, (wx + 12, wy - 12))

            if len(clicks) < n:
                msg = f"Click mark {grid_labels[len(clicks)]}   ({len(clicks)}/{n})"
            else:
                msg = "ENTER confirm | R undo | arrows nudge | G re-grab | ESC cancel"
            display.draw_hud([
                "CALIBRATION - click each projected mark as seen in this image",
                msg,
            ])

            mx, my = pygame.mouse.get_pos()
            cam_xy = win_to_cam(mx, my)
            display.draw_loupe(frame_surf, cam_xy, (mx, my))
            pygame.draw.line(display.screen, (0, 255, 255), (mx - 12, my), (mx + 12, my), 1)
            pygame.draw.line(display.screen, (0, 255, 255), (mx, my - 12), (mx, my + 12), 1)
            display.flip()

    def learn_tape_colors() -> None:
        """Sample the tape's actual color by clicking it, with a live mask preview.

        Projects the normal dark game background, freezes a camera frame of the
        wall+tape under that lighting, then lets the operator click tape spots to
        learn their color. The green overlay previews exactly what would be
        detected, so you can confirm the tape (and not the wall) is captured.
        """
        import pygame

        # Show the real gameplay background while we capture, so the learned
        # color matches what the camera sees during play.
        display.clear()
        display.draw_center_text(["LEARN TAPE COLOR", "Capturing..."])
        display.flip()
        pygame.event.pump()
        time.sleep(0.4)
        frame = None
        for _ in range(8):
            _id, f = camera.read()
            if f is not None:
                frame = f
            time.sleep(0.03)
        if frame is None:
            return

        cam_h, cam_w = frame.shape[:2]
        frame_surf = display.frame_to_surface(frame)
        scale = min(display.width / cam_w, display.height / cam_h)
        disp_w, disp_h = int(cam_w * scale), int(cam_h * scale)
        ox = (display.width - disp_w) // 2
        oy = (display.height - disp_h) // 2
        fitted = pygame.transform.smoothscale(frame_surf, (disp_w, disp_h))

        def win_to_cam(mx, my):
            return (int((mx - ox) / scale), int((my - oy) / scale))

        samples: list = []
        ranges: list = []
        overlay = None
        last_n = -1
        show_preview = True

        while True:
            display.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    cx, cy = win_to_cam(*event.pos)
                    if 0 <= cx < cam_w and 0 <= cy < cam_h:
                        samples.append(vision_mod.sample_hsv(frame, cx, cy))
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key in (pygame.K_r, pygame.K_BACKSPACE) and samples:
                        samples.pop()
                    elif event.key == pygame.K_c:
                        samples.clear()
                    elif event.key == pygame.K_p:
                        show_preview = not show_preview
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and ranges:
                        config.LEARNED_COLORS = ranges
                        vision_mod.save_learned_colors(ranges)
                        config.DETECTION_MODE = "learned"
                        return

            ranges = vision_mod.build_ranges_from_samples(samples)
            if len(samples) != last_n:
                last_n = len(samples)
                overlay = None
                if ranges:
                    mask = vision_mod._learned_mask(frame, ranges)
                    rgba = np.zeros((cam_h, cam_w, 4), dtype=np.uint8)
                    rgba[mask > 0] = (0, 255, 0, 110)
                    osurf = pygame.image.frombuffer(rgba.tobytes(), (cam_w, cam_h), "RGBA")
                    overlay = pygame.transform.scale(osurf, (disp_w, disp_h))

            display.screen.fill((0, 0, 0))
            display.screen.blit(fitted, (ox, oy))
            if show_preview and overlay is not None:
                display.screen.blit(overlay, (ox, oy))

            display.draw_hud([
                "LEARN TAPE COLOR - click on the tape (green/blue)",
                f"samples: {len(samples)}   bands: {len(ranges)}   "
                f"preview(P): {'on' if show_preview else 'off'}",
                "ENTER save | click to add | R undo | C clear | ESC cancel",
            ])
            mx, my = pygame.mouse.get_pos()
            cx, cy = win_to_cam(mx, my)
            cx = max(0, min(cam_w - 1, cx))
            cy = max(0, min(cam_h - 1, cy))
            display.draw_loupe(frame_surf, (cx, cy), (mx, my))
            pygame.draw.line(display.screen, (0, 255, 255), (mx - 12, my), (mx + 12, my), 1)
            pygame.draw.line(display.screen, (0, 255, 255), (mx, my - 12), (mx, my + 12), 1)
            display.flip()

    # If there is no saved calibration, do it once up front.
    if not calibrated:
        do_calibration()

    import pygame

    log.info("Entering main loop (detection=%s, solid=%s)",
             config.DETECTION_MODE, config.OBJECT_SOLID)
    running = True
    last_health = time.time()
    try:
        while running:
            dt = display.tick()

            # Periodic health log to catch slow leaks / runaway growth.
            now = time.time()
            if now - last_health >= 10.0:
                last_health = now
                log.info(
                    "health: fps=%.0f marbles=%d static_shapes=%d objects=%d mode=%s",
                    display.clock.get_fps(), len(world.marbles),
                    len(world._static_shapes), len(world.static_polygons),
                    config.DETECTION_MODE,
                )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_c:
                        vision.capture_background()
                    elif event.key == pygame.K_k:
                        do_calibration()
                    elif event.key == pygame.K_l:
                        learn_tape_colors()
                    elif event.key == pygame.K_o:
                        show_outlines = not show_outlines
                    elif event.key == pygame.K_d:
                        show_diff = not show_diff
                        if not show_diff:
                            try:
                                cv2.destroyWindow("camera diff")
                            except Exception:
                                pass
                    elif event.key == pygame.K_x:
                        world.clear_marbles()
                    elif event.key == pygame.K_h:
                        show_hud = not show_hud
                    elif event.key == pygame.K_g:
                        geometry_locked = not geometry_locked
                    elif event.key in (pygame.K_p, pygame.K_SPACE):
                        paused = not paused
                    elif event.key == pygame.K_b:
                        scheme_index = (scheme_index + 1) % len(config.MARBLE_SCHEMES)
                        world.set_palette(config.MARBLE_SCHEMES[scheme_index][1])
                    elif event.key == pygame.K_m:
                        modes = ["learned", "tape", "combo", "lab", "saturation", "gray"]
                        i = modes.index(config.DETECTION_MODE) if config.DETECTION_MODE in modes else 0
                        config.DETECTION_MODE = modes[(i + 1) % len(modes)]
                    elif event.key == pygame.K_F11:
                        display.toggle_fullscreen()

            # Spawn marbles on a fixed interval (not while paused).
            if not paused:
                spawn_accum += dt
                if spawn_accum >= config.SPAWN_INTERVAL:
                    spawn_accum = 0.0
                    world.spawn_marble()

            # Pull the newest detected objects and (if changed) rebuild geometry.
            # Skipped while paused or when the geometry is locked/frozen, so no
            # new objects are added and existing ones stay put.
            version, polys_cam = vision.get_objects()
            if not paused and not geometry_locked and version != last_vision_version:
                last_vision_version = version
                if config.COLLISION_FROM_MASK:
                    # Warp the detected white pixels into display space and use
                    # them directly as the hard surfaces (pixel-accurate).
                    mask_cam = vision.get_mask()
                    has_pixels = mask_cam is not None and cv2.countNonZero(mask_cam) > 0
                    if has_pixels:
                        empty_streak = 0
                        mask_disp = cv2.warpPerspective(
                            mask_cam, H, (display.width, display.height),
                            flags=cv2.INTER_NEAREST,
                        )
                        world.maybe_rebuild_from_mask(mask_disp)
                    else:
                        empty_streak += 1
                        if empty_streak >= 3:
                            world.maybe_rebuild_from_mask(
                                np.zeros((display.height, display.width), dtype=np.uint8)
                            )
                elif polys_cam:
                    empty_streak = 0
                    polys_disp = [calibration.warp_points(H, p) for p in polys_cam]
                    world.maybe_rebuild_static(polys_disp)
                else:
                    # Only clear after a few consecutive empty frames so a single
                    # dropped detection does not make objects flicker away.
                    empty_streak += 1
                    if empty_streak >= 3:
                        world.maybe_rebuild_static([])

            if not paused:
                world.step(dt)

            # Render.
            display.clear()
            if show_outlines:
                display.draw_object_outlines(
                    world.static_polygons, filled=config.OBJECT_SOLID
                )
            display.draw_marbles(world.marbles)
            if show_hud:
                display.draw_hud([
                    f"FPS {display.clock.get_fps():4.0f}   marbles {len(world.marbles)}",
                    f"background: {'set' if vision.has_background else ('not needed' if config.DETECTION_MODE in vision_mod.NO_BACKGROUND_MODES else 'NOT SET - press C on empty wall')}",
                    f"calibration: {('calibrated (err %.1fpx)' % calib_error) if (calibrated and calib_error is not None) else ('calibrated' if calibrated else 'default/uncalibrated - press K')}",
                    f"detections(cam): {len(polys_cam)}   ->   objects(physics): {len(world.static_polygons)}",
                    f"detection: {config.DETECTION_MODE} (M)"
                    + (("  learned colors: %d%s" % (
                        len(config.LEARNED_COLORS),
                        " - press L to learn!" if not config.LEARNED_COLORS else ""))
                       if config.DETECTION_MODE == "learned" else "")
                    + f"   outlines: {'ON' if show_outlines else 'off'}",
                    f"balls: {config.MARBLE_SCHEMES[scheme_index][0]} (B)",
                    "K calibrate | L learn | C bg | O outlines | D diff | M mode | X clear",
                    "H hide text | G lock geometry | P pause | B ball colors | F11 | ESC",
                ])
            # Compact status badges remain visible even when the HUD text is off.
            badges = []
            if paused:
                badges.append(("PAUSED", (255, 210, 80)))
            if geometry_locked:
                badges.append(("GEOMETRY LOCKED", (120, 200, 255)))
            display.draw_badges(badges)
            display.flip()

            if show_diff:
                mask = vision.get_mask()
                if mask is not None:
                    view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    # Green = detections that passed the area filter (these are
                    # what gets sent to physics). If blobs are visible in white
                    # but have no green outline, raise/lower MIN_OBJECT_AREA.
                    for p in polys_cam:
                        cv2.polylines(view, [p.astype(np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(
                        view, f"accepted: {len(polys_cam)}  (min_area={config.MIN_OBJECT_AREA})",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                    cv2.imshow("camera diff", view)
                cv2.waitKey(1)
    finally:
        vision.stop()
        camera.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        display.quit()
    return 0


# --------------------------------------------------------------------------- #
# Offline self-test (no hardware)
# --------------------------------------------------------------------------- #
def run_check() -> int:
    """Exercise vision, calibration and physics with synthetic data."""
    import cv2

    from marblerun.physics import World
    from marblerun import vision

    print("[check] building synthetic camera frames ...")
    bg = np.full((480, 640, 3), 30, dtype=np.uint8)
    frame = bg.copy()
    cv2.rectangle(frame, (200, 150), (360, 300), (220, 220, 220), -1)
    bg_gray = vision._prepare(bg, config.BLUR_KERNEL)

    polys, mask = vision.detect_objects(frame, bg_gray, mode="lab")
    assert polys, "expected to detect the synthetic rectangle"
    print(f"[check] detected {len(polys)} object(s); first has {len(polys[0])} points")

    print("[check] color detection: blue + green tape on grey drywall ...")
    wall = np.full((480, 640, 3), 180, dtype=np.uint8)  # neutral grey wall
    tape = wall.copy()
    cv2.rectangle(tape, (80, 120), (260, 180), (200, 60, 40), -1)   # blue (BGR)
    cv2.rectangle(tape, (360, 280), (560, 340), (40, 180, 60), -1)  # green (BGR)
    wall_bg = vision._blur_bgr(wall, config.BLUR_KERNEL)
    # The whole point: grayscale diff is weak here, color modes are strong.
    gray_polys, _ = vision.detect_objects(tape, wall_bg, mode="gray")
    sat_polys, _ = vision.detect_objects(tape, None, mode="saturation")
    combo_polys, _ = vision.detect_objects(tape, wall_bg, mode="combo")
    print(f"[check]   gray -> {len(gray_polys)}, saturation -> {len(sat_polys)}, "
          f"combo -> {len(combo_polys)} objects")
    assert len(sat_polys) >= 2, "saturation mode should find both tape strips"
    assert len(combo_polys) >= 2, "combo mode should find both tape strips"

    print("[check] learned-color mode: sample tape, detect by learned color ...")
    # Sample the blue and green tape patches, build bands, then detect.
    s_blue = vision.sample_hsv(tape, 170, 150)
    s_green = vision.sample_hsv(tape, 460, 310)
    learned = vision.build_ranges_from_samples([s_blue, s_green])
    config.LEARNED_COLORS = learned
    learned_polys = vision.detect_objects(tape, None, mode="learned")[0]
    wall_only = vision.detect_objects(wall, None, mode="learned")[0]
    print(f"[check]   learned {len(learned)} band(s); tape -> {len(learned_polys)}, "
          f"bare wall -> {len(wall_only)}")
    assert len(learned_polys) >= 2, "learned mode should detect both sampled tapes"
    assert len(wall_only) == 0, "learned mode must not fire on the bare grey wall"
    config.LEARNED_COLORS = []

    print("[check] tape mode ignores background and tape removal ...")
    # Calibrate background WITH tape present, then remove the tape.
    tape_bg = vision._blur_bgr(tape, config.BLUR_KERNEL)
    removed = wall.copy()  # tape peeled off -> just grey wall
    # Background diff would falsely flag the revealed wall...
    combo_after = vision.detect_objects(removed, tape_bg, mode="combo")[0]
    # ...but tape mode keys on color, so grey wall registers nothing.
    tape_present = vision.detect_objects(tape, None, mode="tape")[0]
    tape_removed = vision.detect_objects(removed, None, mode="tape")[0]
    print(f"[check]   with tape: tape-mode={len(tape_present)} ; "
          f"after removal: tape-mode={len(tape_removed)}, combo-diff={len(combo_after)}")
    assert len(tape_present) >= 2, "tape mode should detect green+blue tape"
    assert len(tape_removed) == 0, "tape mode must NOT flag the revealed wall"

    print("[check] computing homography (camera 640x480 -> display) ...")
    cam_pts = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)
    disp_pts = calibration.display_target_points()
    H = calibration.compute_homography(cam_pts, disp_pts)
    warped = [calibration.warp_points(H, p) for p in polys]
    print(f"[check] warped object centroid -> {warped[0].mean(axis=0).round(1)}")

    print("[check] running physics: rebuild static + drop marbles ...")
    world = World()
    world.maybe_rebuild_static(warped)
    assert world.static_polygons, "static geometry should exist after rebuild"

    print("[check] mask-based collision: white pixels -> hard surfaces ...")
    disp_mask = np.zeros((config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH), dtype=np.uint8)
    cv2.rectangle(disp_mask, (700, 500), (1200, 620), 255, -1)
    cv2.circle(disp_mask, (960, 800), 90, 255, -1)
    mworld = World()
    assert mworld.maybe_rebuild_from_mask(disp_mask), "mask rebuild should run"
    assert len(mworld.static_polygons) >= 2, "expected 2 mask surfaces"
    # Unchanged mask should be skipped by the debounce.
    assert not mworld.maybe_rebuild_from_mask(disp_mask), "unchanged mask should skip"
    for _ in range(120):
        mworld.spawn_marble()
        mworld.step(1.0 / 60.0)
    print(f"[check] mask surfaces: {len(mworld.static_polygons)}, "
          f"marbles resting on them: {len(mworld.marbles)}")

    print("[check] solid fill of a concave (U-shaped) object ...")
    from marblerun.physics import _convex_pieces
    u_shape = np.array(
        [[0, 0], [300, 0], [300, 300], [220, 300], [220, 80],
         [80, 80], [80, 300], [0, 300]], dtype=float)
    pieces = _convex_pieces(u_shape, config.SOLID_DECOMP_TOLERANCE)
    assert len(pieces) >= 2, "concave shape should split into multiple convex solids"
    print(f"[check]   U-shape decomposed into {len(pieces)} convex solid pieces")
    for _ in range(40):
        world.spawn_marble()
        world.step(1.0 / 60.0)
    # Run long enough for some marbles to fall off and be culled.
    for _ in range(600):
        world.step(1.0 / 60.0)
    print(f"[check] marbles still alive after settling: {len(world.marbles)}")

    # Homography round-trip sanity: warp then inverse-warp returns the source.
    Hinv = np.linalg.inv(H)
    back = calibration.warp_points(Hinv, warped[0])
    err = float(np.abs(back - polys[0]).max())
    assert err < 1e-2, f"homography round-trip error too high: {err}"
    print(f"[check] homography round-trip max error {err:.2e}")
    print("[check] OK - all offline checks passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Virtual Marble Run")
    parser.add_argument("--windowed", action="store_true", help="run in a window")
    parser.add_argument("--check", action="store_true", help="offline self-test")
    args = parser.parse_args()

    if args.check:
        return run_check()

    setup_logging()
    try:
        return run_game(windowed=args.windowed)
    except Exception:
        log.critical("run_game crashed", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
