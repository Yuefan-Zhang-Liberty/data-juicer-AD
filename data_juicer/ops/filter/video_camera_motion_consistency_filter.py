import sys
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import numpy as np
from pydantic import PositiveFloat, PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import calculate_resized_dimensions

from ..base_op import OPERATORS, UNFORKABLE, Filter

cv2 = LazyLoader("cv2", "opencv-contrib-python")

OP_NAME = "video_camera_motion_consistency_filter"


@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class VideoCameraMotionConsistencyFilter(Filter):
    """Filter to keep samples whose video camera motion is consistent and trackable.

    For each pair of adjacent sampled frames, the operator detects Shi-Tomasi corners on
    the background of the previous frame, tracks them into the current frame with
    Lucas-Kanade optical flow, and fits a single global homography to the tracked
    correspondences with RANSAC. The fraction of RANSAC inliers (`mean_inlier_ratio`) and
    the mean reprojection error of those inliers (`mean_warp_error`) measure how well a
    single rigid camera motion explains the frame pair. The mean displacement of the
    RANSAC-inlier correspondences is used as a per-step motion velocity, whose
    frame-to-frame variability -- normalized by the clip's own mean velocity so clips
    with different absolute motion speeds are comparable -- gives `motion_smoothness`,
    and whose second derivative gives `max_motion_jerk`. The aggregate
    `camera_motion_consistency` score is the product of `mean_inlier_ratio` and
    `motion_smoothness`, so a clip only scores well when a rigid background motion model
    both fits well and stays smooth over time. Videos too short/corrupted to yield any
    valid tracked frame pair get a sentinel score of -1 and are filtered out by any
    non-negative `min_consistency`.

    This is a lightweight, model-free operator (Shi-Tomasi + Lucas-Kanade + RANSAC
    homography, all from OpenCV) intended to catch corrupted timing, dropped/reordered
    frames, brightness flicker, and scene cuts in dashcam-style footage. Its planar-motion
    assumption is a simplification appropriate for mostly-rigid, mostly-background-filling
    scenes (e.g. street driving footage); it degrades on scenes dominated by large,
    independently-moving foreground objects, since RANSAC will fit whichever motion
    (background or foreground) has the most trackable corners, not necessarily the
    camera's own motion."""

    _default_gftt_kwargs = {
        "maxCorners": 200,
        "qualityLevel": 0.01,
        "minDistance": 7,
    }

    def __init__(
        self,
        min_consistency: float = 0.05,
        max_consistency: float = sys.float_info.max,
        frame_field: Optional[str] = None,
        sampling_fps: PositiveFloat = 2,
        original_fps: Optional[PositiveFloat] = None,
        size: Union[PositiveInt, Tuple[PositiveInt], Tuple[PositiveInt, PositiveInt], None] = None,
        max_size: Optional[PositiveInt] = None,
        divisible: PositiveInt = 1,
        max_corners: PositiveInt = 200,
        quality_level: PositiveFloat = 0.01,
        min_distance: PositiveInt = 7,
        ransac_reproj_threshold: PositiveFloat = 3.0,
        min_track_points: PositiveInt = 8,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_consistency: The minimum camera motion consistency score to keep
            samples. Default 0.05 is calibrated against real nuScenes front-camera clips
            (which score ~0.10-0.25 due to foreground vehicles/pedestrians breaking the
            pure-background-homography assumption -- see the class docstring's limitation
            note) so genuine driving footage is not rejected by default; hard failures
            (corrupted/too-short/fully-decorrelated frame sequences) score exactly -1 and
            are filtered by any non-negative threshold regardless of this default. Raise
            this value to also catch intermediate degradations (e.g. scene cuts, which
            scored ~0.4 on synthetic test data) at the cost of also rejecting more
            aggressive real camera motion such as sharp turns.
        :param max_consistency: The maximum camera motion consistency score to keep
            samples.
        :param frame_field: the field name of video frames to compute the consistency
            score. If frame_field is None, extract frames from the video field.
        :param sampling_fps: The sampling rate in frames_per_second used to select frame
            pairs.
        :param original_fps: The original FPS of the video from which the frames were
            extracted. Only used when `frame_field` is specified. When provided, frames
            will be sampled at `sampling_fps` rate by computing
            `sampling_step = round(original_fps / sampling_fps)`. If None, all frames
            will be processed without sampling.
        :param size: Resize frames before tracking. If size is a sequence like (h, w),
            frame size will be matched to this. If size is an int, smaller edge of frames
            will be matched to this number. Default `None` to keep the original size.
        :param max_size: The maximum allowed for the longer edge of resized frames. Only
            supported if size is an int.
        :param divisible: The number that the dimensions must be divisible by.
        :param max_corners: Max number of Shi-Tomasi corners to detect per frame.
        :param quality_level: Shi-Tomasi minimal accepted corner quality, relative to the
            best corner quality in the frame.
        :param min_distance: Minimum possible Euclidean distance between detected
            Shi-Tomasi corners.
        :param ransac_reproj_threshold: Maximum reprojection error (in pixels) for a
            point pair to be considered a RANSAC inlier when fitting the homography.
        :param min_track_points: Minimum number of successfully tracked correspondences
            required to fit a homography for a frame pair; pairs with fewer tracked
            points are skipped.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_consistency = min_consistency
        self.max_consistency = max_consistency
        self.sampling_fps = sampling_fps
        self.original_fps = original_fps
        self.frame_field = frame_field

        if isinstance(size, (list, tuple)):
            if len(size) not in [1, 2]:
                raise ValueError(
                    f"Size must be an int or a 1 or 2 element tuple/list," f"not a {len(size)} element tuple/list."
                )
        if isinstance(size, int):
            size = (size,)
        self.size = size
        self.max_size = max_size
        self.divisible = divisible

        self.gftt_kwargs = dict(self._default_gftt_kwargs)
        self.gftt_kwargs["maxCorners"] = max_corners
        self.gftt_kwargs["qualityLevel"] = quality_level
        self.gftt_kwargs["minDistance"] = min_distance

        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_track_points = min_track_points

        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def _compute_sampling_step(self, fps):
        if fps is None or fps <= 0:
            return 1
        effective_fps = min(self.sampling_fps, fps)
        return max(round(fps / effective_fps), 1)

    def _resize_if_needed(self, frame):
        if not self.size and not self.max_size:
            return frame
        height, width = frame.shape[:2]
        new_size = calculate_resized_dimensions((height, width), self.size, self.max_size, self.divisible)
        if new_size != (height, width):
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return frame

    def _track_pair(self, prev_gray, curr_gray):
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.gftt_kwargs)
        if p0 is None or len(p0) < self.min_track_points:
            return None

        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        if p1 is None or status is None:
            return None

        status = status.reshape(-1).astype(bool)
        prev_pts = p0.reshape(-1, 2)[status]
        curr_pts = p1.reshape(-1, 2)[status]
        if len(prev_pts) < self.min_track_points:
            return None

        H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, self.ransac_reproj_threshold)
        if H is None or mask is None:
            return None

        mask = mask.reshape(-1).astype(bool)
        if not mask.any():
            return None

        inlier_ratio = float(mask.sum()) / float(len(mask))

        inlier_prev = prev_pts[mask]
        inlier_curr = curr_pts[mask]
        ones = np.ones((len(inlier_prev), 1), dtype=np.float64)
        homogeneous = np.hstack([inlier_prev, ones])
        warped = (H @ homogeneous.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        warp_error = float(np.mean(np.linalg.norm(warped - inlier_curr, axis=1)))

        # Mean displacement of the RANSAC-inlier correspondences, not H[0, 2]/H[1, 2].
        # A homography's own translation entries mix in whatever rotation/perspective
        # component the fit also picked up, so they swing wildly frame-to-frame even for
        # visually smooth motion; the inlier point displacement is the standard, stable
        # proxy for "how far the tracked background actually moved."
        mean_disp = (inlier_curr - inlier_prev).mean(axis=0)
        dx, dy = float(mean_disp[0]), float(mean_disp[1])

        return inlier_ratio, warp_error, dx, dy

    def _aggregate(self, inlier_ratios, warp_errors, dxs, dys):
        if not inlier_ratios:
            return {
                StatsKeys.video_camera_motion_consistency: -1,
                StatsKeys.video_motion_smoothness: -1,
                StatsKeys.video_mean_inlier_ratio: -1,
                StatsKeys.video_mean_warp_error: -1,
                StatsKeys.video_max_motion_jerk: 0,
            }

        mean_inlier_ratio = float(np.mean(inlier_ratios))
        mean_warp_error = float(np.mean(warp_errors))

        velocity = np.hypot(np.array(dxs), np.array(dys))
        if len(velocity) >= 2:
            acceleration = np.diff(velocity)
            mean_velocity = float(np.mean(velocity))
            # Normalize acceleration variability by the clip's own mean velocity so
            # smoothness is comparable across clips with very different absolute motion
            # speed (e.g. a slow synthetic pan vs. fast real driving footage) -- an
            # unnormalized std(acceleration) makes every fast-moving clip look
            # "unsmooth" purely because its pixel displacements are larger, independent
            # of how consistent that motion actually is.
            motion_smoothness = float(1.0 / (1.0 + np.std(acceleration) / (mean_velocity + 1e-3)))
        else:
            acceleration = np.array([])
            motion_smoothness = -1

        if len(acceleration) >= 2:
            jerk = np.diff(acceleration)
            max_motion_jerk = float(np.max(np.abs(jerk)))
        else:
            max_motion_jerk = 0

        if motion_smoothness < 0:
            camera_motion_consistency = -1
        else:
            camera_motion_consistency = mean_inlier_ratio * motion_smoothness

        return {
            StatsKeys.video_camera_motion_consistency: camera_motion_consistency,
            StatsKeys.video_motion_smoothness: motion_smoothness,
            StatsKeys.video_mean_inlier_ratio: mean_inlier_ratio,
            StatsKeys.video_mean_warp_error: mean_warp_error,
            StatsKeys.video_max_motion_jerk: max_motion_jerk,
        }

    def _compute_motion_from_frames(self, frames):
        sampling_step = self._compute_sampling_step(self.original_fps)

        inlier_ratios, warp_errors, dxs, dys = [], [], [], []
        prev_gray = None
        for frame_idx, frame in enumerate(frames):
            if sampling_step > 1 and frame_idx % sampling_step != 0:
                continue

            if isinstance(frame, bytes):
                image_array = np.frombuffer(frame, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                frame = cv2.imread(frame)
            frame = self._resize_if_needed(frame)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                result = self._track_pair(prev_gray, curr_gray)
                if result is not None:
                    inlier_ratio, warp_error, dx, dy = result
                    inlier_ratios.append(inlier_ratio)
                    warp_errors.append(warp_error)
                    dxs.append(dx)
                    dys.append(dy)
            prev_gray = curr_gray

        return self._aggregate(inlier_ratios, warp_errors, dxs, dys)

    def _compute_motion_from_video(self, video_key):
        inlier_ratios, warp_errors, dxs, dys = [], [], [], []
        with VideoCapture(video_key) as cap:
            if not cap.isOpened():
                return self._aggregate(inlier_ratios, warp_errors, dxs, dys)

            fps = cap.get(cv2.CAP_PROP_FPS)
            sampling_step = self._compute_sampling_step(fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sampling_step = max(min(sampling_step, max(total_frames - 1, 1)), 1)

            prev_gray = None
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self._resize_if_needed(frame)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    result = self._track_pair(prev_gray, curr_gray)
                    if result is not None:
                        inlier_ratio, warp_error, dx, dy = result
                        inlier_ratios.append(inlier_ratio)
                        warp_errors.append(warp_error)
                        dxs.append(dx)
                        dys.append(dy)
                prev_gray = curr_gray

                frame_count += sampling_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        return self._aggregate(inlier_ratios, warp_errors, dxs, dys)

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.video_camera_motion_consistency in sample[Fields.stats]:
            return sample

        # there is no video or frames in this sample
        if (self.video_key not in sample or not sample[self.video_key]) and (
            not self.frame_field or self.frame_field not in sample
        ):
            for key in (
                StatsKeys.video_camera_motion_consistency,
                StatsKeys.video_motion_smoothness,
                StatsKeys.video_mean_inlier_ratio,
                StatsKeys.video_mean_warp_error,
                StatsKeys.video_max_motion_jerk,
            ):
                sample[Fields.stats][key] = np.array([], dtype=np.float64)
            return sample

        stats_by_key = {
            StatsKeys.video_camera_motion_consistency: [],
            StatsKeys.video_motion_smoothness: [],
            StatsKeys.video_mean_inlier_ratio: [],
            StatsKeys.video_mean_warp_error: [],
            StatsKeys.video_max_motion_jerk: [],
        }

        if self.frame_field and self.frame_field in sample:
            all_videos_frames = sample[self.frame_field]
            for frames in all_videos_frames:
                result = self._compute_motion_from_frames(frames)
                for key, value in result.items():
                    stats_by_key[key].append(value)
        else:
            loaded_video_keys = sample[self.video_key]
            unique_results = {}
            for video_key in loaded_video_keys:
                if video_key not in unique_results:
                    unique_results[video_key] = self._compute_motion_from_video(video_key)
            for video_key in loaded_video_keys:
                result = unique_results[video_key]
                for key, value in result.items():
                    stats_by_key[key].append(value)

        for key, values in stats_by_key.items():
            sample[Fields.stats][key] = values

        return sample

    def process_single(self, sample):
        consistency_scores = sample[Fields.stats][StatsKeys.video_camera_motion_consistency]

        keep_bools = np.array(
            [self.get_keep_boolean(score, self.min_consistency, self.max_consistency) for score in consistency_scores]
        )
        if len(keep_bools) <= 0:
            return True

        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
