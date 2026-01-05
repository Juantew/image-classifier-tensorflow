"""
Inference script for the Image Classifier (TensorFlow).

Usage examples:
--------------
# Run inference on 9 CIFAR-10 test images
python src/inference.py --model models/keras/<your_model>.keras --num 9

# Save a grid image of predictions
python src/inference.py --model models/keras/<your_model>.keras --num 9 --save-grid reports/figures/pred_grid.png
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using an exported Keras model.")
    parser.add_argument("--model", type=str, required=True, help="Path to exported .keras model file.")
    parser.add_argument("--class-names", type=str, default="models/class_names.json",
                        help="Path to class_names.json (default: models/class_names.json).")
    parser.add_argument("--dataset", type=str, default="cifar10", help="TFDS dataset name (default: cifar10).")
    parser.add_argument("--split", type=str, default="test", help="TFDS split (default: test).")
    parser.add_argument("--img-size", type=int, nargs=2, default=[160, 160],
                        help="Image size as two ints: H W (default: 160 160). Must match training.")
    parser.add_argument("--num", type=int, default=9, help="Number of images to run inference on (default: 9).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42).")
    parser.add_argument("--save-grid", type=str, default=None,
                        help="Optional path to save prediction grid image (e.g., reports/figures/pred_grid.png).")
    return parser.parse_args()


def load_class_names(path: str) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)


def preprocess(image: tf.Tensor, label: tf.Tensor, img_size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Match training-time preprocessing:
    - resize
    - float32
    - scale to [0, 1]
    """
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def sample_batch(ds: tf.data.Dataset, num: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Take a reasonably-sized batch and sample `num` items from it for display.
    """
    # Grab one batch of 128 for sampling
    x_batch, y_batch = next(iter(ds.batch(128).take(1)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(x_batch.shape[0], size=min(num, x_batch.shape[0]), replace=False)
    return x_batch.numpy()[idx], y_batch.numpy()[idx]


def make_grid(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray,
              class_names: List[str], title: str = "Predictions") -> None:
    n = images.shape[0]
    grid = int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(10, 10))
    for i in range(n):
        ax = plt.subplot(grid, grid, i + 1)
        plt.imshow(images[i], interpolation="nearest")
        t = class_names[int(y_true[i])]
        p = class_names[int(y_pred[i])]
        c = float(conf[i])
        ax.set_title(f"T:{t}\nP:{p} ({c:.2f})", fontsize=9)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    class_names = load_class_names(args.class_names)
    img_size = (args.img_size[0], args.img_size[1])

    # Load model
    model = tf.keras.models.load_model(args.model)

    # Load TFDS data (raw), then preprocess
    ds = tfds.load(args.dataset, split=args.split, as_supervised=True)
    ds = ds.map(lambda x, y: preprocess(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # Sample images
    images, labels = sample_batch(ds, num=args.num, seed=args.seed)

    # Predict
    proba = model.predict(images, verbose=0)
    pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)

    # Print a compact text summary
    correct = (pred == labels).sum()
    print(f"Ran inference on {len(labels)} images | correct: {correct}/{len(labels)}")
    for i in range(len(labels)):
        t = class_names[int(labels[i])]
        p = class_names[int(pred[i])]
        print(f"[{i}] true={t:>10s} | pred={p:>10s} | conf={conf[i]:.3f}")

    # Visualize
    make_grid(images, labels, pred, conf, class_names, title="Inference Demo")

    # Optionally save the grid
    if args.save_grid:
        os.makedirs(os.path.dirname(args.save_grid), exist_ok=True)
        plt.figure(figsize=(10, 10))
        n = images.shape[0]
        grid = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            ax = plt.subplot(grid, grid, i + 1)
            plt.imshow(images[i], interpolation="nearest")
            t = class_names[int(labels[i])]
            p = class_names[int(pred[i])]
            c = float(conf[i])
            ax.set_title(f"T:{t}\nP:{p} ({c:.2f})", fontsize=9)
            ax.axis("off")
        plt.suptitle("Inference Demo")
        plt.tight_layout()
        plt.savefig(args.save_grid, dpi=200)
        plt.close()
        print(f"Saved prediction grid to: {args.save_grid}")


if __name__ == "__main__":
    main()
