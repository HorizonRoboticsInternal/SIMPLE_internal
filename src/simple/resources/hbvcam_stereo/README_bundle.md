# Unitree G1 with the HBVCAM stereo fisheye head camera

A G1 (29 DoF, Dex3 hands) with the HBVCAM-F2439GS-2 dual fisheye camera on its face: robot models with the camera
frames, the camera's calibration, and the code that turns pinhole renders into the camera's fisheye images.

## Contents

| file | what it is |
|---|---|
| `g1_29dof_with_hand_hbvcam_stereo.urdf` | the robot with the two eye frames and a visual-only camera body (see *Frames*) |
| `g1_29dof_with_hand_hbvcam_stereo.xml` | the same robot as MuJoCo MJCF, plus five 90° cameras per eye for fisheye rendering |
| `meshes/` | the 50 STL meshes both models use |
| `calibration/stereo_calib.yaml` | 2026-09-10 stereo calibration, OpenCV fisheye model, 1280 × 720 per eye |
| `calibration/calibration_report.txt` | its summary: errors, baseline, K/D, R/T |
| `hbvcam_fisheye.py` | fisheye rays, fisheye from pinhole renders, remap of single images, MuJoCo helpers, self-test |
| `docs/HBVCAM-F2439GS-2_V11_datasheet.pdf` | the camera datasheet (drawing on page 4) |
| `SHA256SUMS` | checksums of everything above |

Requirements: Python 3.9+, `numpy`, `opencv-python`; `mujoco` 3.2 or newer for rendering and the MJCF checks.
Tested with Python 3.10, MuJoCo 3.3.6 and OpenCV 4.11.

```bash
python hbvcam_fisheye.py selftest                  # maths vs OpenCV, MJCF cameras vs URDF
MUJOCO_GL=egl python hbvcam_fisheye.py render      # out/hbvcam_sbs_raw.png and _rect.png (2560 x 720)
```

## The camera

HBVCAM-F2439GS-2 V11: two AR0234 1/2.6" global-shutter sensors, 3.0 µm pixels, fixed-focus fisheye lenses (EFL 3.15 mm,
F2.35, 180° diagonal), lens centres 60 mm apart on an 80 × 25 × 18.3 mm board, USB 2.0 UVC. Used as one
**2560 × 720 MJPG** stream (up to 60 fps): left eye in columns 0–1279, right eye in 1280–2559.

Calibration (`calibration/stereo_calib.yaml`, 25 of 35 board pairs):

| | left | right |
|---|---|---|
| fx, fy (px) | 471.20, 474.36 | 474.31, 477.61 |
| cx, cy (px) | 771.05, 450.16 | 717.45, 450.57 |
| k1…k4 | −0.03587, 0.00030, −0.00297, 0.00036 | −0.03488, −0.00018, −0.00324, 0.00048 |
| field of view in the 1280 × 720 image | 155° × 89° | 163° × 89° |

Stereo: baseline 60.52 mm, RMS 0.117 px. Rectified pair (R1/R2, P1/P2): f 493.6 px, 104.7° × 72.2°, vertical error
median 0.08 px. The principal points are well off the image centre, so the 180° lens circle cuts the left edge and the
corners of each image (black there). `E` and `F` in the yaml are zeros: OpenCV's fisheye calibration does not compute
them, so do not use them.

## Frames

Added to the URDF under `torso_link` (fixed joints), in the MJCF as sites on `torso_link`:

| link / site | xyz (m) | rpy (rad) | axes |
|---|---|---|---|
| `head_stereo_left_optical_frame` | 0.067974825 0.030355827 0.426216236 | −2.169288919 0 −1.570796327 | optical: x right, y down, z forward |
| `head_stereo_right_optical_frame` | 0.067572276 −0.030355827 0.426718221 | −2.169646752 −0.002785440 −1.576804238 | optical |
| `head_stereo_link` (URDF only) | 0.0677735505 0 0.4264672285 | 0 0.598492592 0 | body: x forward, y left, z up |

Both eyes look straight ahead, 34.3° below the torso x axis; with the robot standing (pelvis 0.793 m) they are about
1.27 m above the floor. The eye origins sit about 1 mm in front of the `head_link` face; `head_stereo_link` is a visual
box-and-cylinder model of the camera placed with its lens fronts at the eye origins (1 g, no collision). The URDF eye
frames are 60.72 mm apart with 0.37° of toe-in; the calibration measured 60.52 mm and 0.66°.

**The numbers are relative to `torso_link` and are used unchanged.** Note that G1 models with the rev_1_0 waist (for
example `g1_29dof_wholebody_dex3.xml`) define `torso_link` 44 mm above the pelvis instead of 54 mm, while the head mesh
does not move; the same numbers on such a model put the lenses about 9 mm in front of the face instead of 1 mm.

## Using it

**URDF** (ROS TF, pinocchio, IK): load it with `meshes/` next to it. A URDF cannot carry intrinsics; take them from
`calibration/stereo_calib.yaml`.

**MuJoCo fisheye images.** MuJoCo cameras are pinhole only. The MJCF has five 90° cameras per eye,
`head_stereo_{left,right}_{front,left,right,up,down}`, at the eye origin; `render_hbvcam()` renders them and assembles
each eye's fisheye image. Add your own scene with `mujoco.MjSpec` (see `demo_spec()`), or copy the cameras into another
model with `add_hbvcam_cameras(body, eye_poses())`.

```python
import mujoco, numpy as np, hbvcam_fisheye as hf
model = hf.demo_spec().compile()                      # or your own spec with the robot in it
data = mujoco.MjData(model); mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, 1536, 1536)                 # square, <= 2048 (offscreen buffer set in the MJCF)
eyes = hf.render_hbvcam(model, data, r, hf.load_calib(), rectified=False)
frame = np.hstack([eyes["left"], eyes["right"]])       # 720 x 2560 x 3, like the real stream
```

How: for every fisheye pixel, normalise with K, invert θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸) (Newton) to get the ray's
angle θ, take the face render the ray points into and read its pixel (bilinear); black past θ = 90°. Rectified images
use rays from P1/P2 turned by R1ᵀ/R2ᵀ. This is the inverse of `cv2.fisheye.projectPoints` (the self-test checks it to
10⁻³ px), so points land where the real camera with these intrinsics would put them.

**Remap an existing pinhole image** (for example frames rendered with an older head camera):

```bash
# SIMPLE's default head camera (640 x 360, 110° horizontal = 77.55° vertical), turned to the HBVCAM left eye
python hbvcam_fisheye.py remap head_stereo_left.png --fovy 77.55 --source simple_head --out left_fisheye.png
# keep the source camera's pose and only change the lens
python hbvcam_fisheye.py remap image.png --fovy 42.56 --out image_fisheye.png
```

`pinhole_to_fisheye()` does the same in code. One pinhole image only covers part of the fisheye's view: everything
outside it stays black (SIMPLE's 110° camera fills about 58 % of the lens circle at its own pose, 49 % when turned
13.3° to the HBVCAM eye). The translation between the two cameras is ignored, and the source's lower angular
resolution (224 px/rad for SIMPLE's camera vs 471 px/rad for the HBVCAM at the centre) makes the result softer.

## Limits

- One projection centre per eye; a real fisheye's entrance pupil shifts by a few mm with angle (matters only very close).
- No vignetting, blur, chromatic aberration, exposure or MJPG compression; add them yourself if a policy needs them.
- Five-face renders cost five camera renders per eye per frame.
