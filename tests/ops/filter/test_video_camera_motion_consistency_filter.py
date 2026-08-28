import os
import shutil
import tempfile
import unittest

import numpy as np
from datasets import Dataset

from data_juicer.ops.filter.video_camera_motion_consistency_filter import (
    VideoCameraMotionConsistencyFilter,
)
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

cv2 = LazyLoader("cv2", "opencv-contrib-python")


def _make_background(h=240, w=320, seed=0):
    rng = np.random.RandomState(seed)
    bg = rng.randint(0, 255, size=(h + 100, w + 100, 3), dtype=np.uint8)
    # overlay a grid of lines so Shi-Tomasi has real corners to detect
    bg[::10, :, :] = 0
    bg[:, ::10, :] = 0
    return bg


def _write_video(path, frames, fps=10):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


class VideoCameraMotionConsistencyFilterTest(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix="dj_camera_motion_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def _video_path(self, name):
        return os.path.join(self.tmp_dir, name)

    def _static_frames(self, num_frames=15, h=240, w=320):
        bg = _make_background(h, w, seed=1)
        crop = bg[50 : 50 + h, 50 : 50 + w]
        return [crop.copy() for _ in range(num_frames)]

    def _pan_frames(self, num_frames=15, h=240, w=320, step=2):
        bg = _make_background(h, w, seed=2)
        frames = []
        for i in range(num_frames):
            y, x = 50 + i * step, 50 + i * step
            frames.append(bg[y : y + h, x : x + w].copy())
        return frames

    def _rotate_frames(self, num_frames=15, h=240, w=320, angle_step=1.5):
        bg = _make_background(h, w, seed=3)
        crop = bg[50 : 50 + h, 50 : 50 + w]
        center = (w // 2, h // 2)
        frames = []
        for i in range(num_frames):
            mat = cv2.getRotationMatrix2D(center, angle_step * i, 1.0)
            frames.append(cv2.warpAffine(crop, mat, (w, h)))
        return frames

    def _scene_cut_frames(self, num_frames=16, h=240, w=320):
        half = num_frames // 2
        first = self._pan_frames(num_frames=half, h=h, w=w, step=2)
        bg2 = _make_background(h, w, seed=99)
        crop2 = bg2[50 : 50 + h, 50 : 50 + w]
        second = [crop2.copy() for _ in range(num_frames - half)]
        return first + second

    def _run_stats(self, op, video_path):
        sample = {op.video_key: [video_path], Fields.stats: {}}
        sample = op.compute_stats_single(sample)
        return sample[Fields.stats][StatsKeys.video_camera_motion_consistency][0]

    def test_static(self):
        path = self._video_path("static.mp4")
        _write_video(path, self._static_frames())
        op = VideoCameraMotionConsistencyFilter()
        score = self._run_stats(op, path)
        self.assertGreater(score, 0.5)

    def test_smooth_pan(self):
        path = self._video_path("pan.mp4")
        _write_video(path, self._pan_frames())
        op = VideoCameraMotionConsistencyFilter(min_consistency=0.3)
        score = self._run_stats(op, path)
        self.assertGreaterEqual(score, 0.3)

    def test_smooth_rotation(self):
        path = self._video_path("rotate.mp4")
        _write_video(path, self._rotate_frames())
        op = VideoCameraMotionConsistencyFilter(min_consistency=0.3)
        score = self._run_stats(op, path)
        self.assertGreaterEqual(score, 0.3)

    def test_duplicate_frames(self):
        frames = self._pan_frames()
        frames = [frames[i // 2] for i in range(len(frames) * 2)]
        path = self._video_path("dup.mp4")
        _write_video(path, frames)
        op = VideoCameraMotionConsistencyFilter()
        score = self._run_stats(op, path)
        self.assertTrue(np.isfinite(score))

    def test_dropped_frames(self):
        frames = self._pan_frames(num_frames=20)
        frames = frames[::3]
        path = self._video_path("dropped.mp4")
        _write_video(path, frames)
        op = VideoCameraMotionConsistencyFilter()
        score = self._run_stats(op, path)
        self.assertTrue(np.isfinite(score))

    def test_reordered_frames(self):
        pan_frames = self._pan_frames()
        pan_path = self._video_path("pan_ref.mp4")
        _write_video(pan_path, pan_frames)

        rng = np.random.RandomState(42)
        shuffled = list(pan_frames)
        rng.shuffle(shuffled)
        shuffled_path = self._video_path("shuffled.mp4")
        _write_video(shuffled_path, shuffled)

        op = VideoCameraMotionConsistencyFilter()
        pan_score = self._run_stats(op, pan_path)
        shuffled_score = self._run_stats(op, shuffled_path)
        self.assertLess(shuffled_score, pan_score)

    def test_brightness_flicker(self):
        frames = self._pan_frames()
        flickered = []
        for i, frame in enumerate(frames):
            factor = 1.6 if i % 2 == 0 else 0.4
            flickered.append(np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8))
        path = self._video_path("flicker.mp4")
        _write_video(path, flickered)
        op = VideoCameraMotionConsistencyFilter()
        score = self._run_stats(op, path)
        self.assertTrue(np.isfinite(score))

    def test_scene_cut(self):
        pan_path = self._video_path("pan_ref2.mp4")
        _write_video(pan_path, self._pan_frames(num_frames=16))

        cut_path = self._video_path("scene_cut.mp4")
        _write_video(cut_path, self._scene_cut_frames())

        op = VideoCameraMotionConsistencyFilter()
        pan_score = self._run_stats(op, pan_path)
        cut_score = self._run_stats(op, cut_path)
        self.assertLess(cut_score, pan_score)

    def test_invalid_path(self):
        op = VideoCameraMotionConsistencyFilter()
        score = self._run_stats(op, self._video_path("does_not_exist.mp4"))
        self.assertEqual(score, -1)

    def test_corrupted_video(self):
        path = self._video_path("corrupted.mp4")
        with open(path, "wb"):
            pass
        op = VideoCameraMotionConsistencyFilter()
        score = self._run_stats(op, path)
        self.assertEqual(score, -1)

    def test_single_frame(self):
        path = self._video_path("single.mp4")
        _write_video(path, self._static_frames(num_frames=1))
        op = VideoCameraMotionConsistencyFilter()
        score = self._run_stats(op, path)
        self.assertEqual(score, -1)

    def test_any(self):
        static_path = self._video_path("static_any.mp4")
        _write_video(static_path, self._static_frames())
        corrupted_path = self._video_path("corrupted_any.mp4")
        with open(corrupted_path, "wb"):
            pass

        ds_list = [{"videos": [static_path, corrupted_path]}]
        op = VideoCameraMotionConsistencyFilter(min_consistency=0.3, any_or_all="any")
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        dataset = op.run(dataset)
        self.assertEqual(dataset.num_rows, 1)

    def test_all(self):
        static_path = self._video_path("static_all.mp4")
        _write_video(static_path, self._static_frames())
        corrupted_path = self._video_path("corrupted_all.mp4")
        with open(corrupted_path, "wb"):
            pass

        ds_list = [{"videos": [static_path, corrupted_path]}]
        op = VideoCameraMotionConsistencyFilter(min_consistency=0.3, any_or_all="all")
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        dataset = op.run(dataset)
        self.assertEqual(dataset.num_rows, 0)

    def test_stats_caching(self):
        path = self._video_path("cache.mp4")
        _write_video(path, self._pan_frames())
        op = VideoCameraMotionConsistencyFilter()
        sample = {op.video_key: [path], Fields.stats: {}}
        sample = op.compute_stats_single(sample)
        first = sample[Fields.stats][StatsKeys.video_camera_motion_consistency]

        # tamper with the cached value to prove the second call is a no-op
        sample[Fields.stats][StatsKeys.video_camera_motion_consistency] = ["cached"]
        sample = op.compute_stats_single(sample)
        self.assertEqual(sample[Fields.stats][StatsKeys.video_camera_motion_consistency], ["cached"])
        self.assertNotEqual(first, ["cached"])

    def test_frame_field(self):
        frames = self._pan_frames()
        frame_dir = self._video_path("frames")
        os.makedirs(frame_dir, exist_ok=True)
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frame_dir, f"{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

        op = VideoCameraMotionConsistencyFilter(min_consistency=0.3, frame_field="frames")
        sample = {"frames": [frame_paths], Fields.stats: {}}
        sample = op.compute_stats_single(sample)
        score = sample[Fields.stats][StatsKeys.video_camera_motion_consistency][0]
        self.assertGreaterEqual(score, 0.3)


if __name__ == "__main__":
    unittest.main()
