from __future__ import annotations

# dataset_generator.py

from pathlib import Path
import math
import random

import torch
from PIL import Image, ImageDraw


IMAGE_SIZE = 32
OUTPUT_FILE = Path.cwd() / "shapes_dataset.pt"


def _new_canvas() -> Image.Image:
    """Create a white grayscale PIL image."""
    return Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a uint8 tensor with shape [64, 64]."""
    return torch.frombuffer(
        bytearray(image.tobytes()),
        dtype=torch.uint8,
    ).reshape(IMAGE_SIZE, IMAGE_SIZE).clone()


def _random_width() -> int:
    """Return a random line thickness."""
    return random.randint(1, 1)


def _random_point(margin: float) -> tuple[float, float]:
    return (
        random.uniform(margin, IMAGE_SIZE - 1 - margin),
        random.uniform(margin, IMAGE_SIZE - 1 - margin),
    )


def _rotate_point(
    x: float,
    y: float,
    angle: float,
) -> tuple[float, float]:
    """Rotate a point around the origin."""
    cosine = math.cos(angle)
    sine = math.sin(angle)

    return (
        x * cosine - y * sine,
        x * sine + y * cosine,
    )


def _place_points_inside(
    points: list[tuple[float, float]],
    line_width: int,
) -> list[tuple[int, int]]:
    """Shift points so the entire shape remains inside the image."""
    half_width = line_width / 2

    min_x = min(x for x, _ in points) - half_width
    max_x = max(x for x, _ in points) + half_width
    min_y = min(y for _, y in points) - half_width
    max_y = max(y for _, y in points) + half_width

    min_cx = -min_x
    max_cx = IMAGE_SIZE - 1 - max_x
    min_cy = -min_y
    max_cy = IMAGE_SIZE - 1 - max_y

    if min_cx > max_cx or min_cy > max_cy:
        raise ValueError("Shape is too large to fit inside the image.")

    cx = random.uniform(min_cx, max_cx)
    cy = random.uniform(min_cy, max_cy)

    return [
        (round(x + cx), round(y + cy))
        for x, y in points
    ]


def _generate_rotated_polygon(
    points: list[tuple[float, float]],
    width: int,
) -> torch.Tensor:
    """Draw a rotated polygon fully inside the image."""
    image = _new_canvas()
    draw = ImageDraw.Draw(image)

    angle = random.uniform(0, 2 * math.pi)

    rotated_points = [
        _rotate_point(x, y, angle)
        for x, y in points
    ]

    placed_points = _place_points_inside(rotated_points, width)

    draw.line(
        placed_points + [placed_points[0]],
        fill=0,
        width=width,
        joint="curve",
    )

    return _to_tensor(image)


def generate_circle() -> torch.Tensor:
    """Generate a randomly sized black circle."""
    image = _new_canvas()
    draw = ImageDraw.Draw(image)

    width = _random_width()
    margin = width + 1

    radius = random.uniform(
        3.0,
        (IMAGE_SIZE - 2 * margin) / 2,
    )

    cx = random.uniform(
        margin + radius,
        IMAGE_SIZE - 1 - margin - radius,
    )
    cy = random.uniform(
        margin + radius,
        IMAGE_SIZE - 1 - margin - radius,
    )

    bbox = [
        round(cx - radius),
        round(cy - radius),
        round(cx + radius),
        round(cy + radius),
    ]

    draw.ellipse(
        bbox,
        outline=0,
        width=width,
    )

    return _to_tensor(image)


def generate_oval() -> torch.Tensor:
    """Generate a randomly sized and rotated oval."""
    width = _random_width()

    rx = random.uniform(4.0, 11.0)
    ry = random.uniform(2.5, 8.0)

    points = []

    for index in range(96):
        theta = 2 * math.pi * index / 96

        points.append(
            (
                rx * math.cos(theta),
                ry * math.sin(theta),
            )
        )

    return _generate_rotated_polygon(points, width)


def generate_line() -> torch.Tensor:
    """Generate a random line fully inside the image."""
    image = _new_canvas()
    draw = ImageDraw.Draw(image)

    width = _random_width()
    margin = width + 1

    x1, y1 = _random_point(margin)
    x2, y2 = _random_point(margin)

    draw.line(
        [
            (round(x1), round(y1)),
            (round(x2), round(y2)),
        ],
        fill=0,
        width=width,
    )

    return _to_tensor(image)


def generate_rectangle() -> torch.Tensor:
    """Generate a randomly sized and rotated rectangle with guaranteed aspect ratio contrast."""
    width = _random_width()

    while True:
        rectangle_width = random.uniform(8.0, 20.0)
        rectangle_height = random.uniform(6.0, 18.0)
        # Guarantee a clear difference so it cannot look like a square
        if abs(rectangle_width - rectangle_height) >= 4.0:
            break

    half_width = rectangle_width / 2
    half_height = rectangle_height / 2

    points = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ]

    return _generate_rotated_polygon(points, width)


def generate_square() -> torch.Tensor:
    """Generate a randomly sized and rotated square."""
    width = _random_width()

    side = random.uniform(6.0, 18.0)
    half_side = side / 2

    points = [
        (-half_side, -half_side),
        (half_side, -half_side),
        (half_side, half_side),
        (-half_side, half_side),
    ]

    return _generate_rotated_polygon(points, width)


def generate_triangle() -> torch.Tensor:
    """Generate a randomly sized and rotated triangle."""
    width = _random_width()

    triangle_width = random.uniform(8.0, 20.0)
    triangle_height = random.uniform(7.0, 20.0)

    points = [
        (0.0, -triangle_height / 2),
        (-triangle_width / 2, triangle_height / 2),
        (triangle_width / 2, triangle_height / 2),
    ]

    return _generate_rotated_polygon(points, width)


def save_dataset(samples_per_shape: int = 10_000) -> None:
    """Generate all samples and save them into one PyTorch file."""

    generators = {
        "circle": generate_circle,
        "oval": generate_oval,
        "line": generate_line,
        "rectangle": generate_rectangle,
        "square": generate_square,
        "triangle": generate_triangle,
    }

    class_names = list(generators.keys())
    images: list[torch.Tensor] = []
    labels: list[int] = []

    for label, (shape_name, generator) in enumerate(generators.items()):
        print(f"Generating {samples_per_shape} {shape_name} samples...")

        for _ in range(samples_per_shape):
            images.append(generator())
            labels.append(label)

    dataset = {
        "images": torch.stack(images),

        # Shape: [number_of_samples]
        "labels": torch.tensor(labels, dtype=torch.long),

        # Integer-to-name mapping
        "class_names": class_names,
    }

    torch.save(dataset, OUTPUT_FILE)

    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Images shape: {dataset['images'].shape}")
    print(f"Labels shape: {dataset['labels'].shape}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    save_dataset(samples_per_shape=10_000)
