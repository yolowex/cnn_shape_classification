import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle


DATASET_PATH = Path.cwd() / "shapes_dataset.pt"


if __name__ == "__main__":
    # Load the dataset directly from the .pt file
    data = torch.load(DATASET_PATH, weights_only=False)

    images = data["images"]          # shape: [N, 32, 32]
    labels = data["labels"]          # shape: [N]
    class_names = data["class_names"]

    num_rows = 5
    num_columns = 8
    samples_per_shape = 7

    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(8, 6),
    )
    figure.subplots_adjust(
        bottom=0.15,
        hspace=0.35,
        wspace=0.2,
    )

    axes = axes.ravel()

    def show_random_batch() -> None:
        selected_indices = []

        # Select exactly two random images from each shape class
        for class_index in range(len(class_names)):
            class_indices = [
                index
                for index, label in enumerate(labels)
                if label.item() == class_index
            ]

            selected_indices.extend(
                random.sample(class_indices, samples_per_shape)
            )

        # Shuffle the batch so shapes are not always in the same positions
        random.shuffle(selected_indices)

        for axis, random_index in zip(axes, selected_indices):
            image = images[random_index]
            label = labels[random_index]
            shape_name = class_names[label.item()]

            axis.clear()
            axis.imshow(
                image.numpy(),
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            axis.set_title(shape_name)
            axis.axis("off")

            # Dark red border around the image
            border = Rectangle(
                (-0.5, -0.5),
                image.shape[1],
                image.shape[0],
                linewidth=2,
                edgecolor="#8B0000",
                facecolor="none",
                transform=axis.transData,
                clip_on=False,
            )
            axis.add_patch(border)

        figure.canvas.draw_idle()

    # Create the button below the images
    button_axis = figure.add_axes([0.4, 0.03, 0.2, 0.07])
    next_button = Button(button_axis, "Next batch")
    next_button.on_clicked(lambda event: show_random_batch())

    # Show the first random batch
    show_random_batch()

    # Keep the window open
    plt.show()
