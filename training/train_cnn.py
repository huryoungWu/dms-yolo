"""
CNN 训练脚本：基于 MobileNetV2 迁移学习构建眼部和嘴部分类模型。

用法:
    python -m training.train_cnn --dataset_path data/eyes --dataset_type eye --output_dir models/trained
    python -m training.train_cnn --dataset_path data/mouth --dataset_type mouth --output_dir models/trained
"""

import argparse
import os
import json

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def _load_tf():
    """延迟加载 TensorFlow"""
    import tensorflow as tf
    return tf


def split_dataset(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将数据集按指定比例划分为训练集、验证集和测试集。

    Args:
        X: 特征数组
        y: 标签数组
        train_ratio: 训练集比例，默认 0.8
        val_ratio: 验证集比例，默认 0.1
        test_ratio: 测试集比例，默认 0.1

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)

    Raises:
        ValueError: 比例之和不为 1 或数据集太小
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"比例之和必须为 1.0，当前为 {total}"
        )

    if len(X) < 3:
        raise ValueError("数据集至少需要 3 个样本")

    # 第一次划分：分出训练集和临时集（验证+测试）
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_ratio, random_state=42, stratify=y
    )

    # 第二次划分：从临时集中分出验证集和测试集
    if len(X_temp) < 2:
        # 数据太少时，验证集和测试集各取一半
        mid = max(1, len(X_temp) // 2)
        X_val, X_test = X_temp[:mid], X_temp[mid:]
        y_val, y_test = y_temp[:mid], y_temp[mid:]
    else:
        test_fraction = test_ratio / temp_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_fraction, random_state=42,
            stratify=y_temp
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_images(dataset_path, dataset_type, target_size=(64, 64)):
    """
    从目录加载图像数据集。

    Args:
        dataset_path: 数据集根目录
        dataset_type: "eye" 或 "mouth"
        target_size: 目标图像尺寸

    Returns:
        (images, labels) - numpy 数组
    """
    if dataset_type == "eye":
        class_dirs = {"open": 0, "closed": 1}
    elif dataset_type == "mouth":
        # 支持 normal/yawn 或 no_yawn/yawn 两种目录命名
        if os.path.isdir(os.path.join(dataset_path, "normal")):
            class_dirs = {"normal": 0, "yawn": 1}
        else:
            class_dirs = {"no_yawn": 0, "yawn": 1}
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}，请使用 'eye' 或 'mouth'")

    images = []
    labels = []

    for class_name, label in class_dirs.items():
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"警告: 目录不存在，跳过 {class_path}")
            continue

        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue

            # resize 到目标尺寸
            img = cv2.resize(img, target_size)

            # 灰度图转 3 通道（MobileNetV2 需要 3 通道输入）
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 归一化到 [0, 1]
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(label)

    if len(images) == 0:
        raise ValueError(f"未能从 {dataset_path} 加载任何图像")

    return np.array(images), np.array(labels)


def build_model(input_shape=(64, 64, 3)):
    """
    构建基于 MobileNetV2 迁移学习的二分类模型。

    架构:
        1. MobileNetV2 base (冻结)
        2. GlobalAveragePooling2D
        3. Dense(128, relu)
        4. Dropout(0.5)
        5. Dense(1, sigmoid)

    Args:
        input_shape: 输入图像形状

    Returns:
        编译好的 Keras 模型
    """
    tf = _load_tf()

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    # 冻结 base 层，只训练顶层
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    评估模型并生成报告。

    Args:
        model: 训练好的 Keras 模型
        X_test: 测试集特征
        y_test: 测试集标签

    Returns:
        dict 包含 accuracy, precision, recall, f1, confusion_matrix, report
    """
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "accuracy": report_dict["accuracy"],
        "precision": report_dict["weighted avg"]["precision"],
        "recall": report_dict["weighted avg"]["recall"],
        "f1": report_dict["weighted avg"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "report": report_text,
    }

    return results


def train(dataset_path, dataset_type, output_dir, epochs=20, batch_size=32):
    """
    完整训练流程：加载数据 → 划分 → 构建模型 → 训练 → 评估 → 保存。

    Args:
        dataset_path: 数据集路径
        dataset_type: "eye" 或 "mouth"
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批大小

    Returns:
        评估结果 dict
    """
    print(f"加载 {dataset_type} 数据集: {dataset_path}")
    X, y = load_images(dataset_path, dataset_type)
    print(f"加载完成: {len(X)} 张图像")

    # 80/10/10 划分
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 构建模型
    print("构建 MobileNetV2 迁移学习模型...")
    model = build_model(input_shape=(64, 64, 3))
    model.summary()

    # 训练
    print(f"开始训练 ({epochs} epochs, batch_size={batch_size})...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # 评估
    print("评估模型...")
    results = evaluate_model(model, X_test, y_test)

    print("\n===== 评估报告 =====")
    print(results["report"])
    print(f"混淆矩阵:\n{np.array(results['confusion_matrix'])}")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_name = f"{dataset_type}_model.h5"
    model_path = os.path.join(output_dir, model_name)
    model.save(model_path)
    print(f"模型已保存: {model_path}")

    # 保存评估报告
    report_path = os.path.join(output_dir, f"{dataset_type}_eval_report.json")
    report_data = {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "confusion_matrix": results["confusion_matrix"],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"评估报告已保存: {report_path}")

    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="基于 MobileNetV2 迁移学习训练眼部/嘴部分类模型"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--dataset_type", type=str, required=True,
        choices=["eye", "mouth"],
        help="数据集类型: eye（眼部）或 mouth（嘴部）"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/trained",
        help="模型输出目录（默认: models/trained）"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="训练轮数（默认: 20）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="批大小（默认: 32）"
    )

    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
