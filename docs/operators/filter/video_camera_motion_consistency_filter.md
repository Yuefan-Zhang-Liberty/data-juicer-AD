# video_camera_motion_consistency_filter

Filter to keep samples whose video camera motion is consistent and trackable, using Shi-Tomasi corners + Lucas-Kanade optical flow + RANSAC homography fitting.

用于保留相机运动一致且可追踪的视频样本的过滤器，使用 Shi-Tomasi 角点 + Lucas-Kanade 光流 + RANSAC 单应矩阵拟合实现。

Type 算子类型: **filter**

Tags 标签: cpu, video

## Algorithm 算法

For each pair of adjacent sampled frames:

1. **Shi-Tomasi corner detection** on the grayscale previous frame (`cv2.goodFeaturesToTrack`).
2. **Lucas-Kanade optical flow** tracks those corners into the current frame (`cv2.calcOpticalFlowPyrLK`); only points with `status == 1` are kept.
3. **RANSAC homography** is fitted to the tracked correspondences (`cv2.findHomography`); the fraction of RANSAC inliers (`inlier_ratio`) and mean reprojection error of inliers (`warp_error`) are recorded.
4. Mean displacement of the RANSAC-inlier correspondences gives a per-step **velocity** estimate `(dx, dy)`.

Per-video aggregation:
- `mean_inlier_ratio = mean(inlier_ratios)` — how well a single rigid camera-motion model fits each frame pair.
- `motion_smoothness = 1 / (1 + std(acceleration) / (mean_velocity + 1e-3))` — how smooth the motion trajectory is over the clip; normalized by the clip's own mean velocity for speed-invariant comparison.
- `camera_motion_consistency = mean_inlier_ratio × motion_smoothness` — the primary score (both factors in [0, 1], so the product is too).
- Videos that yield no valid tracked frame pair (corrupted, too short, unreadable) receive a sentinel score of `-1` and are excluded by any non-negative `min_consistency`.

The five computed stats are cached in `Fields.stats` under the keys listed below, so downstream analysis can inspect them without re-running the operator.

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|---|---|---|---|
| `min_consistency` | `float` | `0.05` | Minimum `camera_motion_consistency` score to keep a sample. Default is calibrated for real nuScenes front-camera clips (~0.10–0.25 due to foreground vehicles breaking the homography assumption). |
| `max_consistency` | `float` | `sys.float_info.max` | Maximum score (effectively unbounded by default). |
| `frame_field` | `Optional[str]` | `None` | Field name of pre-extracted frames to use instead of the video field. |
| `sampling_fps` | `float` | `2` | Frames-per-second rate at which to sample frame pairs from the video. |
| `original_fps` | `Optional[float]` | `None` | Original FPS of pre-extracted frames (only used when `frame_field` is set). |
| `size` | `int \| tuple \| None` | `None` | Resize frames before tracking. Int → smaller edge matched to this value. |
| `max_size` | `Optional[int]` | `None` | Maximum allowed longer edge after resize. |
| `divisible` | `int` | `1` | Dimension divisibility constraint for resizing. |
| `max_corners` | `int` | `200` | Maximum Shi-Tomasi corners to detect per frame. |
| `quality_level` | `float` | `0.01` | Shi-Tomasi minimum accepted corner quality, relative to the best corner. |
| `min_distance` | `int` | `7` | Minimum Euclidean distance between detected corners (pixels). |
| `ransac_reproj_threshold` | `float` | `3.0` | Maximum reprojection error (pixels) for a point pair to be a RANSAC inlier. |
| `min_track_points` | `int` | `8` | Minimum tracked correspondences needed to fit a homography for a frame pair. |
| `any_or_all` | `str` | `"any"` | Multi-video strategy: `"any"` keeps the sample if any video passes; `"all"` requires all to pass. |

## 📊 Stats keys emitted

| key | description |
|---|---|
| `video_camera_motion_consistency` | Primary score: `mean_inlier_ratio × motion_smoothness`. `-1` for hard failures. |
| `video_motion_smoothness` | `1 / (1 + std(acceleration) / mean_velocity)`. `-1` if < 2 valid frame pairs. |
| `video_mean_inlier_ratio` | Mean fraction of RANSAC inliers across all sampled frame pairs. |
| `video_mean_warp_error` | Mean reprojection error (pixels) of RANSAC inliers. |
| `video_max_motion_jerk` | Maximum absolute jerk (third derivative of position). |

## ⚠️ Known limitations / 已知限制

The operator assumes a **planar rigid-background motion model**. It degrades gracefully on scenes dominated by large, independently-moving foreground objects (e.g. a vehicle filling most of the frame), because RANSAC will fit whichever motion — background or foreground — has the most trackable corners, not necessarily the camera's own motion. This simplification is appropriate for dashcam/autonomous-driving footage where the background fills most of the frame; it is not appropriate for close-up tracking shots.

该算子假设背景满足**平面刚体运动模型**。当画面中存在大量独立运动的前景目标（如近距离大型车辆）时，RANSAC 可能拟合前景运动而非相机自身运动，导致性能下降。该简化对于行车记录仪/自动驾驶场景（背景占据画面大部分）是合理的，但不适用于近距离跟拍镜头。

## Example YAML

```yaml
process:
  - video_camera_motion_consistency_filter:
      min_consistency: 0.1
      max_consistency: 1.0
      sampling_fps: 2
      max_corners: 200
      ransac_reproj_threshold: 3.0
      any_or_all: any
```
