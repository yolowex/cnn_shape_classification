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

    figure, axis = plt.subplots()
    figure.subplots_adjust(bottom=0.2)

    def show_random_sample() -> None:
        random_index = random.randrange(len(images))

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
        axis.set_title(f"Label: {shape_name}")
        axis.axis("off")

        # Dark red border around the image
        border = Rectangle(
            (-0.5, -0.5),
            image.shape[1],
            image.shape[0],
            linewidth=3,
            edgecolor="#8B0000",   # dark red
            facecolor="none",
            transform=axis.transData,
            clip_on=False,
        )
        axis.add_patch(border)

        figure.canvas.draw_idle()

    # Create the button below the image
    button_axis = figure.add_axes([0.4, 0.05, 0.2, 0.08])
    next_button = Button(button_axis, "Next")
    next_button.on_clicked(lambda event: show_random_sample())

    # Show the first random sample
    show_random_sample()

    # Keep the window open
    plt.show()