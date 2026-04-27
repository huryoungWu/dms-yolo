#!/usr/bin/env python3
"""
Runnable YOLOv8 training/prediction script converted from a Kaggle notebook.

What this script fixes compared with the extracted notebook:
- Replaces Kaggle-only paths with command-line arguments.
- Replaces notebook/CLI-style training snippets with valid Python API calls.
- Adds a main entry point so it can run as a normal Python program.
- Optionally prepares a cleaned dataset while preserving YOLO labels.

Typical dataset layout:
    dataset_root/
      data.yaml
      train/
        images/
        labels/
      valid/
        images/
        labels/
      test/
        images/
        labels/   # optional

Example usage:
    python dms_yolov8_runnable.py \
        --mode all \
        --dataset-root ./dms-driver-monitoring-system \
        --model yolov8n.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 16

    python dms_yolov8_runnable.py \
        --mode train \
        --dataset-root ./dms-driver-monitoring-system \
        --data-yaml ./dms-driver-monitoring-system/data.yaml

    python dms_yolov8_runnable.py \
        --mode predict \
        --trained-model ./runs/detect/train_v1/weights/best.pt \
        --predict-source ./dms-driver-monitoring-system/test/images
"""

from __future__ import annotations

import argparse
import platform
import logging
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageOps
from skimage.io import imread
from ultralytics import YOLO

LOGGER = logging.getLogger("dms_yolov8")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_batch_arg(value: str):
    """Allow batch to be an int (e.g. 8, 16), auto mode -1, or a fraction like 0.7."""
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --batch value: {value}") from exc
    if numeric.is_integer() and numeric >= 1:
        return int(numeric)
    if numeric == -1:
        return -1
    if 0 < numeric < 1:
        return numeric
    raise argparse.ArgumentTypeError("--batch must be a positive integer, -1, or a fraction between 0 and 1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DMS YOLOv8 training/prediction locally.")
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "predict", "all"],
        default="all",
        help="prepare: build cleaned dataset/yaml; train: train only; predict: predict only; all: prepare+train+predict",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("./dataset"), help="Dataset root directory")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Path to original or existing YOLO data.yaml")
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional class names used when generating a data.yaml from scratch",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("./workdir"), help="Output working directory")
    parser.add_argument("--use-cleaned", action="store_true", help="Train on cleaned dataset prepared under work-dir")
    parser.add_argument("--target-size", type=int, default=640, help="Resize target size for cleaned images")
    parser.add_argument("--skip-visualization", action="store_true", help="Do not save sample visualization figures")

    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model name or checkpoint for training")
    parser.add_argument("--trained-model", type=Path, default=None, help="Existing trained .pt model for prediction")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=parse_batch_arg, default=8,
                        help="Batch size: positive int, -1 for auto, or 0<x<1 for memory fraction")
    parser.add_argument("--device", default=None, help="Training/prediction device, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=(0 if platform.system() == "Windows" else 8),
                        help="Dataloader workers. Windows建议先用0，内存更稳")
    parser.add_argument("--cache", action="store_true", help="Cache dataset in RAM/disk for speed; low-memory machines should leave this off")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="Use mixed precision training (default: on)")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable mixed precision if your environment is unstable")

    parser.add_argument(
        "--predict-source",
        type=Path,
        default=None,
        help="Prediction image directory or file. Defaults to dataset_root/test/images if available.",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Prediction confidence threshold")

    parser.add_argument("--project", type=Path, default=Path("./runs/detect"), help="Ultralytics project output dir")
    parser.add_argument("--train-name", default="train_v1", help="Ultralytics train run name")
    parser.add_argument("--predict-name", default="predict_v1", help="Ultralytics predict run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample plotting")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_files_in(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def find_split_dirs(dataset_root: Path) -> dict:
    mapping = {}
    for split in ("train", "valid", "val", "test"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if images_dir.exists():
            canonical = "val" if split == "valid" else split
            mapping[canonical] = {"images": images_dir, "labels": labels_dir if labels_dir.exists() else None}
    return mapping


def is_unrelated(image_path: Path) -> bool:
    """Simple heuristic kept from the notebook: filter extreme aspect ratios."""
    with Image.open(image_path) as img:
        width, height = img.size
    aspect_ratio = width / max(height, 1)
    return aspect_ratio < 0.5 or aspect_ratio > 2.5



def clean_dataset_images(input_dir: Path, output_dir: Path, target_size: int = 640) -> List[str]:
    """Clean, resize, and save images to a new directory."""
    unrelated_images: List[str] = []
    ensure_dir(output_dir)
    cleaned_count = 0

    for img_path in image_files_in(input_dir):
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if is_unrelated(img_path):
                    LOGGER.info("Skipping unrelated image: %s", img_path.name)
                    unrelated_images.append(img_path.name)
                    continue
                resized_img = ImageOps.pad(
                    img,
                    (target_size, target_size),
                    color=(114, 114, 114),
                    centering=(0.5, 0.5),
                )
                resized_img.save(output_dir / img_path.name)
                cleaned_count += 1
        except (OSError, Image.DecompressionBombError) as exc:
            LOGGER.warning("Skipping unreadable image %s: %s", img_path.name, exc)

    LOGGER.info("Cleaned dataset: %s valid images from %s", cleaned_count, input_dir)
    LOGGER.info("Found %s unrelated images in %s", len(unrelated_images), input_dir)
    return unrelated_images



def copy_matching_labels(src_labels_dir: Optional[Path], dst_labels_dir: Path, kept_image_names: Iterable[str]) -> None:
    if src_labels_dir is None or not src_labels_dir.exists():
        LOGGER.warning("No labels directory found, skipping label copy: %s", src_labels_dir)
        return
    ensure_dir(dst_labels_dir)
    count = 0
    for image_name in kept_image_names:
        label_name = f"{Path(image_name).stem}.txt"
        src_label = src_labels_dir / label_name
        if src_label.exists():
            shutil.copy2(src_label, dst_labels_dir / label_name)
            count += 1
    LOGGER.info("Copied %s label files to %s", count, dst_labels_dir)



def display_random(data_path: Path, save_path: Optional[Path] = None, num_images: int = 12, seed: int = 42) -> None:
    images = image_files_in(data_path)
    num_images = min(num_images, len(images))
    if num_images == 0:
        LOGGER.warning("No images found in %s", data_path)
        return

    random.seed(seed)
    cols = 3
    rows = (num_images // cols) + int(num_images % cols > 0)
    plt.figure(figsize=(cols * 3, rows * 3))

    selected = random.sample(images, k=num_images) if len(images) >= num_images else images
    for i, img_path in enumerate(selected, start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(imread(img_path))
        plt.axis("off")

    plt.suptitle("Random images", fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path is not None:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        LOGGER.info("Saved sample figure to %s", save_path)
    plt.close()



def plot_image_grid(image_paths: List[Path], save_path: Path, title: str = "Predictions", cols: int = 4, figsize=(20, 15)) -> None:
    if not image_paths:
        LOGGER.warning("No images found to plot in grid.")
        return

    rows = len(image_paths) // cols + (1 if len(image_paths) % cols else 0)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows, cols)

    for idx, img_path in enumerate(image_paths):
        row, col = divmod(idx, cols)
        with Image.open(img_path) as img:
            axes[row, col].imshow(img)
        axes[row, col].axis("off")

    for idx in range(len(image_paths), rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved prediction grid to %s", save_path)



def resolve_data_yaml(args: argparse.Namespace) -> Path:
    if args.data_yaml is not None:
        return args.data_yaml.resolve()
    default_yaml = args.dataset_root / "data.yaml"
    if default_yaml.exists():
        return default_yaml.resolve()
    raise FileNotFoundError(
        "No data.yaml found. Please provide --data-yaml, or place data.yaml under dataset-root, "
        "or use --class-names to generate one via --mode prepare."
    )



def write_updated_yaml(src_yaml: Optional[Path], dst_yaml: Path, train_dir: Path, val_dir: Path, test_dir: Optional[Path], class_names: Optional[List[str]]) -> Path:
    data = {}
    if src_yaml is not None and src_yaml.exists():
        with src_yaml.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                data.update(loaded)

    if class_names:
        data["names"] = class_names
        data["nc"] = len(class_names)

    if "names" in data and "nc" not in data and isinstance(data["names"], (list, tuple)):
        data["nc"] = len(data["names"])

    data["train"] = str(train_dir.resolve())
    data["val"] = str(val_dir.resolve())
    if test_dir is not None and test_dir.exists():
        data["test"] = str(test_dir.resolve())

    missing = [key for key in ("train", "val", "names", "nc") if key not in data]
    if missing:
        raise ValueError(
            f"Cannot create runnable data.yaml because required fields are missing: {missing}. "
            "Supply an existing data.yaml or pass --class-names."
        )

    ensure_dir(dst_yaml.parent)
    with dst_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    LOGGER.info("Wrote YOLO data.yaml to %s", dst_yaml)
    return dst_yaml



def prepare_dataset(args: argparse.Namespace) -> Path:
    splits = find_split_dirs(args.dataset_root)
    if "train" not in splits or "val" not in splits:
        raise FileNotFoundError(
            f"Expected train/images and valid/images or val/images under {args.dataset_root}."
        )

    src_yaml = None
    if args.data_yaml is not None:
        src_yaml = args.data_yaml
    else:
        default_yaml = args.dataset_root / "data.yaml"
        if default_yaml.exists():
            src_yaml = default_yaml

    if args.use_cleaned:
        cleaned_root = ensure_dir(args.work_dir / "cleaned_dataset")
        for split_name in ("train", "val", "test"):
            split = splits.get(split_name)
            if split is None:
                continue
            dst_images = cleaned_root / split_name / "images"
            dst_labels = cleaned_root / split_name / "labels"
            skipped = set(clean_dataset_images(split["images"], dst_images, target_size=args.target_size))
            kept = [p.name for p in image_files_in(dst_images) if p.name not in skipped]
            copy_matching_labels(split["labels"], dst_labels, kept)
            if not args.skip_visualization:
                display_random(
                    dst_images,
                    save_path=args.work_dir / "figures" / f"sample_{split_name}.png",
                    seed=args.seed,
                )
        train_dir = cleaned_root / "train" / "images"
        val_dir = cleaned_root / "val" / "images"
        test_dir = cleaned_root / "test" / "images"
    else:
        train_dir = splits["train"]["images"]
        val_dir = splits["val"]["images"]
        test_dir = splits.get("test", {}).get("images") if splits.get("test") else None
        if not args.skip_visualization:
            display_random(train_dir, save_path=args.work_dir / "figures" / "sample_train.png", seed=args.seed)
            if test_dir is not None:
                display_random(test_dir, save_path=args.work_dir / "figures" / "sample_test.png", seed=args.seed)
            display_random(val_dir, save_path=args.work_dir / "figures" / "sample_val.png", seed=args.seed)

    output_yaml = args.work_dir / "data.yaml"
    return write_updated_yaml(
        src_yaml=src_yaml,
        dst_yaml=output_yaml,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        class_names=args.class_names,
    )



def train_model(data_yaml: Path, args: argparse.Namespace) -> Path:
    LOGGER.info(
        "Starting training with model=%s, data=%s, imgsz=%s, batch=%s, workers=%s, amp=%s",
        args.model, data_yaml, args.imgsz, args.batch, args.workers, args.amp
    )
    model = YOLO(args.model)
    try:
        train_results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            plots=True,
            project=str(args.project),
            name=args.train_name,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            amp=args.amp,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "bad allocation" in msg or "out of memory" in msg or "cuda" in msg:
            raise SystemExit(
                "训练在反向传播阶段内存分配失败。请优先尝试：\n"
                "1) --batch 4 或 --batch 2\n"
                "2) --imgsz 512 或 416\n"
                "3) Windows 下加 --workers 0\n"
                "4) 用 --batch -1 让 Ultralytics 自动估算 batch\n"
                "5) 确认没有别的程序占用显卡；仍不行就先 --device cpu 验证流程\n"
                f"原始错误: {exc}"
            ) from exc
        raise
    run_dir = Path(train_results.save_dir)
    LOGGER.info("Training finished. Run directory: %s", run_dir)

    final_model_path = args.work_dir / "final_model.pt"
    shutil.copy2(run_dir / "weights" / "best.pt", final_model_path)
    LOGGER.info("Saved final model copy to %s", final_model_path)
    return final_model_path



def show_training_results(run_dir: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    for name in ("confusion_matrix.png", "results.png", "PR_curve.png"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            LOGGER.info("Copied training artifact %s", src.name)



def predict_model(model_path: Path, source: Path, args: argparse.Namespace) -> Path:
    LOGGER.info("Running prediction with model=%s on source=%s", model_path, source)
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source),
        conf=args.conf,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(args.project),
        name=args.predict_name,
        device=args.device,
    )
    del results  # results object not needed further
    predict_dir = args.project / args.predict_name
    LOGGER.info("Prediction outputs saved to %s", predict_dir)
    return predict_dir



def resolve_predict_source(args: argparse.Namespace) -> Path:
    if args.predict_source is not None:
        return args.predict_source
    candidate = args.dataset_root / "test" / "images"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Prediction source not found. Provide --predict-source.")



def main() -> None:
    setup_logging()
    args = parse_args()
    ensure_dir(args.work_dir)
    ensure_dir(args.project)

    prepared_yaml: Optional[Path] = None
    trained_model: Optional[Path] = None

    if args.mode in {"prepare", "all"}:
        prepared_yaml = prepare_dataset(args)
        LOGGER.info("Prepared dataset yaml: %s", prepared_yaml)

    if args.mode == "train":
        prepared_yaml = resolve_data_yaml(args)

    if args.mode in {"train", "all"}:
        if prepared_yaml is None:
            prepared_yaml = resolve_data_yaml(args)
        trained_model = train_model(prepared_yaml, args)
        show_training_results(args.project / args.train_name, args.work_dir / "training_artifacts")

    if args.mode == "predict":
        if args.trained_model is None:
            raise FileNotFoundError("For --mode predict, please provide --trained-model path.")
        trained_model = args.trained_model

    if args.mode in {"predict", "all"}:
        if trained_model is None:
            if args.trained_model is not None:
                trained_model = args.trained_model
            else:
                fallback_best = args.project / args.train_name / "weights" / "best.pt"
                if fallback_best.exists():
                    trained_model = fallback_best
                else:
                    raise FileNotFoundError(
                        "No trained model available for prediction. Provide --trained-model or run training first."
                    )
        predict_source = resolve_predict_source(args)
        predict_dir = predict_model(trained_model, predict_source, args)
        predicted_images = sorted(predict_dir.glob("*.jpg")) + sorted(predict_dir.glob("*.png"))
        if not args.skip_visualization and predicted_images:
            plot_image_grid(
                predicted_images[:12],
                save_path=args.work_dir / "figures" / "prediction_grid.png",
                title="Sample Predictions",
            )

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
