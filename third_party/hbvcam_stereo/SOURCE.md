# HBVCAM stereo head camera for the G1 (vendored)

Copied unchanged on 2026-09-29 from `docs/_scan/g1_head_stereo_camera/dist/g1_hbvcam_stereo/` (built by
`docs/_scan/g1_head_stereo_camera/pack_bundle.py`); the bundle's own README is `README_bundle.md`.

| file | sha256 |
|---|---|
| `g1_29dof_with_hand_hbvcam_stereo.xml` | `639b9c998bdb55ab…` |
| `calibration/stereo_calib.yaml` | `01e41b85900d03d0…` |
| `calibration/calibration_report.txt` | `cb980cc7b6878ac1…` |
| `README_bundle.md` | `b2f6593be096afd4…` |

- `g1_29dof_with_hand_hbvcam_stereo.xml`: SIMPLE's `g1_29dof_with_hand.xml` (same bodies, masses, joints, actuators)
  plus the eye-frame sites, five 90° cameras per eye and a visual-only camera body. Its meshes are the stock ones in
  `data/robots/g1_sonic/meshes/` (identical files), so `robot_variants.py` installs it there.
- `calibration/stereo_calib.yaml`: the 2026-09-10 stereo calibration (OpenCV fisheye, 1280 × 720 per eye). The teleop uses its
  rectified left eye (R1, P1) as a pinhole camera.

- `hbvcam_fisheye.py` + `g1_29dof_with_hand_hbvcam_stereo.urdf`: the bundle's fisheye helper (same layout as the
  bundle, so its paths work); the teleop's `--stream-camera fisheye` uses its face definitions and fisheye ray model.
