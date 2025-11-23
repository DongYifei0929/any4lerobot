from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from scipy.spatial.transform import Rotation as R

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_process_scripts.egodex.utils.skeleton_tfs import DEFAULT_TFS  # uwm

VIDEO_TOP_KEY = "observation.images.top_head"
STATE_FEATURE_KEY = "observation.state"
ACTION_FEATURE_KEY = "action"
EEF_STATE_KEY = "eef.state"
EEF_ACTION_KEY = "eef.action"
ANNOTATION_TEXT_KEY = "annotation.language.action_text"
TIMESTAMP_KEY = "timestamp"
CAMERA_INTRINSIC_KEY = "camera.intrinsic"
CAMERA_EXTRINSIC_KEY = "camera.extrinsic"

DEFAULT_VIDEO_SHAPE = (1080, 1920, 3)
DEFAULT_FPS = 30
DEFAULT_TARGET_FPS = 10
DEFAULT_ROBOT_TYPE = "dex"

TRANSFORM_KEYS = sorted(DEFAULT_TFS)
EEF_KEYS = ("leftHand", "rightHand")
GRIPPER_CLOSED_VALUE = 0.0

CAMERA_ALIGN_FIX = np.array(
    [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)
PIXEL_EPS = 1e-9
AXIS_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
TEXT_COLOR = (255, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
TEXT_LINE_STEP = 18
POINT_COLORS = {"leftHand": (0, 165, 255), "rightHand": (255, 0, 255)}
AXIS_LENGTH = 0.1
AXIS_THICKNESS = 2
POINT_RADIUS = 5

DATASET_PATH_PREFIX = "data"
VIEW_NAME = "front_view"
DEFAULT_DATASET_NAME = "EgoDex_Preprocessed"


@dataclass
class EpisodeData:
    source_path: Path
    data_id: str
    task_name: str
    instruction: str
    state_vectors: np.ndarray
    eef_states: np.ndarray
    eef_actions: np.ndarray
    video_path: Path | None
    camera_tf: np.ndarray
    camera_intrinsics: np.ndarray
    transforms_camera_frame: Dict[str, np.ndarray]

    @property
    def num_frames(self) -> int:
        return self.state_vectors.shape[0]


def get_lerobot_dataset_class():
    try:
        module = importlib.import_module("lerobot.common.datasets.lerobot_dataset")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "需要安装 'huggingface-lerobot' 才能生成 LeRobot 数据集：pip install huggingface-lerobot"
        ) from exc
    return getattr(module, "LeRobotDataset")


def rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2))
    singular = sy < 1e-6
    if not singular:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = 0.0
    return roll, pitch, yaw


def transform_to_pose7(mat: np.ndarray) -> np.ndarray:
    pose = np.empty(7, dtype=np.float32)
    pose[:3] = mat[:3, 3]
    roll, pitch, yaw = rotation_matrix_to_rpy(mat[:3, :3])
    pose[3] = roll
    pose[4] = pitch
    pose[5] = yaw
    pose[6] = GRIPPER_CLOSED_VALUE
    return pose


def compute_eef_states_actions(transforms_camera_frame: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    missing = [key for key in EEF_KEYS if key not in transforms_camera_frame]
    if missing:
        raise KeyError(f"缺少末端执行器变换：{', '.join(missing)}")

    num_frames = transforms_camera_frame[EEF_KEYS[0]].shape[0]
    eef_dim = len(EEF_KEYS) * 7
    eef_states = np.empty((num_frames, eef_dim), dtype=np.float32)
    eef_actions = np.zeros((num_frames, eef_dim), dtype=np.float32)

    for frame_idx in range(num_frames):
        offset = 0
        for key in EEF_KEYS:
            tf = transforms_camera_frame[key][frame_idx]
            eef_states[frame_idx, offset : offset + 7] = transform_to_pose7(tf)
            offset += 7

    if num_frames > 1:
        for frame_idx in range(num_frames - 1):
            offset = 0
            for key in EEF_KEYS:
                curr_tf = transforms_camera_frame[key][frame_idx]
                fut_tf = transforms_camera_frame[key][frame_idx + 1]
                rel_tf = np.linalg.inv(curr_tf) @ fut_tf
                delta_trans = rel_tf[:3, 3].astype(np.float32)
                delta_rpy = rotation_matrix_to_rpy(rel_tf[:3, :3])
                eef_actions[frame_idx, offset : offset + 7] = np.array(
                    [
                        delta_trans[0],
                        delta_trans[1],
                        delta_trans[2],
                        float(delta_rpy[0]),
                        float(delta_rpy[1]),
                        float(delta_rpy[2]),
                        0.0,
                    ],
                    dtype=np.float32,
                )
                offset += 7

    return eef_states, eef_actions


def align_eef_transforms(
    transforms_camera_frame: Dict[str, np.ndarray], align_matrix: np.ndarray | None = None
) -> Dict[str, np.ndarray]:
    if align_matrix is None:
        return {key: value.copy() for key, value in transforms_camera_frame.items()}

    matrix = np.asarray(align_matrix, dtype=np.float32).reshape(4, 4)
    aligned: Dict[str, np.ndarray] = {}
    for key, value in transforms_camera_frame.items():
        aligned[key] = np.matmul(matrix[np.newaxis, ...], value).astype(np.float32)
    return aligned






def extract_instruction(attrs: h5py.AttributeManager) -> str:
    def attr_text(key: str, default: str = "") -> str:
        value = attrs.get(key, default)
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    if attr_text("llm_type") == "reversible":
        direction = attr_text("which_llm_description", "1")
        key = "llm_description" if direction == "1" else "llm_description2"
        return attr_text(key).strip()
    return attr_text("llm_description").strip()


def load_episode(hdf5_path: Path, input_root: Path | None) -> EpisodeData:
    with h5py.File(hdf5_path, "r") as root:
        camera_tf = np.asarray(root["/transforms/camera"], dtype=np.float32)
        instruction = extract_instruction(root.attrs)
        transforms_world = {
            key: np.asarray(root[f"/transforms/{key}"], dtype=np.float32)
            for key in TRANSFORM_KEYS
        }
        intrinsics = np.asarray(root["/camera/intrinsic"], dtype=np.float32)

    if intrinsics is None:
        raise ValueError(f"未能在 {hdf5_path} 找到相机内参数据。")

    num_frames = camera_tf.shape[0]
    cam_inv = np.linalg.inv(camera_tf).astype(np.float32)
    transforms_camera_frame: Dict[str, np.ndarray] = {
        key: np.einsum("tij,tjk->tik", cam_inv, transforms_world[key], dtype=np.float32)
        for key in TRANSFORM_KEYS
    }
    if "rightHand" in transforms_camera_frame:
        transforms_camera_frame["rightHand"] = transforms_camera_frame["rightHand"].copy()
        # transforms_camera_frame["rightHand"][:, :3, 3] *= -1
        # transforms_camera_frame["rightHand"][:, :3, 1] *= -1

    state_vectors = np.concatenate(
        [transforms_camera_frame[key].reshape(num_frames, 16) for key in TRANSFORM_KEYS],
        axis=1,
    )

    eef_states, eef_actions = compute_eef_states_actions(transforms_camera_frame)

    try:
        if input_root is not None:
            relative_parts = hdf5_path.relative_to(input_root).parts
        else:
            raise ValueError
    except ValueError:
        relative_parts = hdf5_path.parts

    task_folder = relative_parts[-2] if len(relative_parts) >= 2 else hdf5_path.parent.name
    task_name = task_folder or "unknown_task"
    data_id = f"{task_name}/{hdf5_path.stem}" if task_folder else hdf5_path.stem

    video_path = hdf5_path.with_suffix(".mp4")

    return EpisodeData(
        source_path=hdf5_path,
        data_id=data_id,
        task_name=task_name,
        instruction=instruction or "unknown EgoDex instruction",
        state_vectors=state_vectors,
        eef_states=eef_states,
        eef_actions=eef_actions,
        video_path=video_path if video_path.exists() else None,
        camera_tf=camera_tf,
        camera_intrinsics=intrinsics,
        transforms_camera_frame=transforms_camera_frame,
    )




def build_features(eef_state_dim: int, eef_action_dim: int, target_fps: float, transform_names: List[str]) -> Dict[str, Dict[str, object]]:
    video_feature = {
        "dtype": "video",
        "shape": [int(x) for x in DEFAULT_VIDEO_SHAPE],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(target_fps),
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }

    def vector_feature(size: int, dtype: str = "float32") -> Dict[str, object]:
        return {"dtype": dtype, "shape": [int(size)]}

    features: Dict[str, Dict[str, object]] = {
        VIDEO_TOP_KEY: video_feature,
        STATE_FEATURE_KEY: vector_feature(1),
        ACTION_FEATURE_KEY: vector_feature(1),
        EEF_STATE_KEY: vector_feature(eef_state_dim),
        EEF_ACTION_KEY: vector_feature(eef_action_dim),
        ANNOTATION_TEXT_KEY: {"dtype": "int64", "shape": [1]},
        TIMESTAMP_KEY: vector_feature(1),
        CAMERA_INTRINSIC_KEY: vector_feature(9),
        CAMERA_EXTRINSIC_KEY: vector_feature(16),
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for name in transform_names:
        if name == "camera":
            continue
        features[name] = vector_feature(16)
    return features

def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")


def write_modality_file(dataset_root: Path, eef_state_dim: int, eef_action_dim: int) -> None:
    segments = len(EEF_KEYS)
    if eef_state_dim % segments or eef_action_dim % segments:
        raise ValueError("EEF 维度无法平均分配给左右末端执行器，请检查输入维度。")

    eef_state_stride = int(eef_state_dim / segments)
    eef_action_stride = int(eef_action_dim / segments)

    state_slices: Dict[str, Dict[str, object]] = {}
    action_slices: Dict[str, Dict[str, object]] = {}
    for idx, key in enumerate(EEF_KEYS):
        name = "left_end_effector" if "left" in key.lower() else "right_end_effector"
        state_slices[name] = {
            "start": int(idx * eef_state_stride),
            "end": int((idx + 1) * eef_state_stride),
            "original_key": EEF_STATE_KEY,
        }
        action_slices[name] = {
            "start": int(idx * eef_action_stride),
            "end": int((idx + 1) * eef_action_stride),
            "original_key": EEF_ACTION_KEY,
        }

    modality = {
        "state": state_slices,
        "action": action_slices,
        "video": {"top_head": {"original_key": VIDEO_TOP_KEY}},
        "annotation": {"language.action_text": {"original_key": ANNOTATION_TEXT_KEY}},
    }
    write_json(dataset_root / "meta" / "modality.json", modality)


def update_episode_annotation_files(dataset_root: Path, episode_rows: List[Dict[str, object]]) -> None:
    episodes_dir = dataset_root / "meta" / "episodes"
    if not episodes_dir.exists():
        return

    for row in episode_rows:
        episode_file = episodes_dir / f"episode_{row['episode_index']:06d}.json"
        if not episode_file.exists():
            continue
        with episode_file.open("r", encoding="utf-8") as fp:
            episode_info = json.load(fp)
        annotation = episode_info.setdefault("annotation", {})
        language_entry = annotation.setdefault("language.action_text", {})
        language_entry["task_index"] = int(row["task_index"])
        language_entry["text"] = row["instruction"]
        language_entry["source"] = "EgoDex attribute"

        episode_info["instruction"] = row["instruction"]
        episode_info.setdefault("source", {})["egodex_data_id"] = row["data_id"]
        if row.get("video_path"):
            episode_info.setdefault("video", {})["top_head"] = row["video_path"]

        with episode_file.open("w", encoding="utf-8") as fp:
            json.dump(episode_info, fp, indent=2, ensure_ascii=False)


def update_info_file(dataset_root: Path, total_episodes: int, unique_tasks: int) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return
    with info_path.open("r", encoding="utf-8") as fp:
        info = json.load(fp)
    info["description"] = "EgoDex → LeRobot 转换数据集，包含真实的手部 SE(3) 与摄像机参数，图像与动作仍为占位符。"
    info["source_dataset"] = "EgoDex"
    info["num_episodes_converted"] = int(total_episodes)
    info["num_unique_tasks"] = int(unique_tasks)
    write_json(info_path, info)


def write_tasks_file(dataset_root: Path, task_rows: List[Dict[str, object]]) -> None:
    write_jsonl(dataset_root / "meta" / "tasks.jsonl", task_rows)


def convert_egodex_to_lerobot(
    input_path: Path,
    output_path: Path,
    repo_id: str,
    max_episodes: int | None = None,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> Path:
    hdf5_files = sorted(path for path in input_path.rglob("*.hdf5") if path.is_file())
    if not hdf5_files:
        raise FileNotFoundError(f"在 {input_path} 未找到任何 .hdf5 文件。")
    if max_episodes is not None:
        hdf5_files = hdf5_files[: max(0, max_episodes)]

    dataset_root = (output_path / repo_id).resolve()
    if dataset_root.exists():
        raise ValueError
        shutil.rmtree(dataset_root)

    LeRobotDataset = get_lerobot_dataset_class()

    first_episode = load_episode(hdf5_files[0], input_path)
    eef_state_dim = first_episode.eef_states.shape[1]
    eef_action_dim = first_episode.eef_actions.shape[1]

    # 计算 stride（简单整除假设）
    if DEFAULT_FPS % target_fps != 0:
        raise ValueError(f"源FPS {DEFAULT_FPS} 不能被目标FPS {target_fps} 整除，暂不支持该比例。")
    stride = int(DEFAULT_FPS / target_fps)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(dataset_root),
        fps=float(target_fps),
        robot_type=DEFAULT_ROBOT_TYPE,
        features=build_features(eef_state_dim, eef_action_dim, target_fps, TRANSFORM_KEYS),
    )

    tasks: Dict[str, int] = {}
    task_rows: List[Dict[str, object]] = []
    episode_annotation_rows: List[Dict[str, object]] = []

    def ensure_task(task_name: str, instruction: str) -> int:
        if task_name not in tasks:
            idx = len(tasks)
            tasks[task_name] = idx
            task_rows.append({"task_index": idx, "task": task_name, "instruction": instruction})
        return tasks[task_name]

    episodes_saved = 0
    progress = tqdm(total=len(hdf5_files), desc="Converting EgoDex")

    for file_idx, hdf5_path in enumerate(hdf5_files):
        episode = first_episode if file_idx == 0 else load_episode(hdf5_path, input_path)
        task_index = ensure_task(episode.task_name, episode.instruction)
        episode_index = dataset.meta.total_episodes

        zero_state = np.zeros(1, dtype=np.float32)
        zero_action = np.zeros(1, dtype=np.float32)
        aligned_transforms = align_eef_transforms(episode.transforms_camera_frame, CAMERA_ALIGN_FIX)

        # ===== 下采样：每 stride 取一帧 =====
        num_frames_src = episode.num_frames
        indices = np.arange(0, num_frames_src, stride, dtype=int)
        # 防止最后一个索引越界或空
        if len(indices) == 0:
            continue

        subsampled_transforms = {
            k: v[indices] for k, v in aligned_transforms.items()
        }
        subsampled_camera_tf = episode.camera_tf[indices]
        # 重新计算 EEF state/action（delta 会对应较大的时间步）
        aligned_eef_states, aligned_eef_actions = compute_eef_states_actions(subsampled_transforms)
        num_frames_sub = aligned_eef_states.shape[0]
        intrinsic = episode.camera_intrinsics.astype(np.float32).reshape(9)
        intrinsic[0*3+2], intrinsic[1*3+2] = intrinsic[1*3+2], intrinsic[0*3+2]

        progress.write(f"[episode {episode_index:05d}] 写入 {episode.data_id}，原始帧数 {episode.num_frames} → 下采样后 {num_frames_sub}")
        for frame_idx in range(num_frames_sub):
            extrinsic = subsampled_camera_tf[frame_idx].astype(np.float32).reshape(16)
            transform_vectors = {
                name: subsampled_transforms[name][frame_idx].astype(np.float32).reshape(16)
                for name in TRANSFORM_KEYS
                if name != "camera" and name in subsampled_transforms
            }
            dataset.add_frame(
                {
                    STATE_FEATURE_KEY: zero_state.copy(),
                    ACTION_FEATURE_KEY: zero_action.copy(),
                    EEF_STATE_KEY: aligned_eef_states[frame_idx],
                    EEF_ACTION_KEY: aligned_eef_actions[frame_idx],
                    ANNOTATION_TEXT_KEY: np.array([task_index], dtype=np.int64),
                    TIMESTAMP_KEY: np.array([frame_idx / target_fps], dtype=np.float32),
                    CAMERA_INTRINSIC_KEY: intrinsic,
                    CAMERA_EXTRINSIC_KEY: extrinsic,
                    **transform_vectors,
                }
            )

        dataset.save_episode(task=episode.task_name, encode_videos=False)
        video_rel_path: str | None = None
        if episode.video_path and episode.video_path.exists():
            rel_path = dataset.meta.get_video_file_path(episode_index, VIDEO_TOP_KEY)
            video_dest = dataset.root / rel_path
            video_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(episode.video_path, video_dest)
            video_rel_path = str(rel_path)

        episodes_saved += 1
        episode_annotation_rows.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "instruction": episode.instruction,
                "data_id": episode.data_id,
                "video_path": video_rel_path,
            }
        )
        progress.update(1)

    progress.close()

    if episodes_saved == 0:
        raise RuntimeError("无有效 EgoDex episode 被转换。")

    dataset.consolidate(run_compute_stats=False, keep_image_files=False)
    task_rows_sorted = sorted(task_rows, key=lambda row: int(row["task_index"]))
    write_modality_file(dataset_root, eef_state_dim, eef_action_dim)
    write_tasks_file(dataset_root, task_rows_sorted)
    write_json(
        dataset_root / "meta" / "annotation_maps.json",
        {ANNOTATION_TEXT_KEY: {str(row["task_index"]): row["instruction"] for row in task_rows_sorted}},
    )
    update_episode_annotation_files(dataset_root, episode_annotation_rows)
    update_info_file(dataset_root, total_episodes=episodes_saved, unique_tasks=len(tasks))
    return dataset_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if not cli_args or cli_args[0] not in {"convert", "visualize"}:
        cli_args = ["convert", *cli_args]

    parser = argparse.ArgumentParser(description="EgoDex 数据处理工具：支持数据转换与末端执行器可视化。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="将 EgoDex 数据集转换为 LeRobot 数据集")
    convert_parser.add_argument("--input_path", type=Path, required=True)
    convert_parser.add_argument("--output_path", type=Path, required=True)
    convert_parser.add_argument("--repo_id", type=str, required=True)
    convert_parser.add_argument("--max_episodes", type=int, default=None)
    convert_parser.add_argument("--target_fps", type=float, default=DEFAULT_TARGET_FPS, help="目标下采样FPS (必须整除源FPS 30)")

    visualize_parser = subparsers.add_parser("visualize", help="在指定帧上绘制 EgoDex 末端执行器姿态")
    visualize_parser.add_argument("--hdf5_path", type=Path, required=True)
    visualize_parser.add_argument("--frame_idx", type=int, required=True)
    visualize_parser.add_argument("--output_path", type=Path, required=True)
    visualize_parser.add_argument("--video_path", type=Path, default=None)

    return parser.parse_args(cli_args)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_path.mkdir(parents=True, exist_ok=True)
    dataset_root = convert_egodex_to_lerobot(
        input_path=args.input_path,
        output_path=args.output_path,
        repo_id=args.repo_id,
        max_episodes=args.max_episodes,
        target_fps=args.target_fps,
    )
    print(f"已生成 LeRobot 数据集：{dataset_root}")
    return



if __name__ == "__main__":
    main()