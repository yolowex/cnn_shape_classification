import random

import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from load_dataset import *

if __name__ == "__main__":


    # Display one randomly selected sample.
    figure, axis = plt.subplots()
    figure.subplots_adjust(bottom=0.2)

    def show_random_sample() -> None:
        random_index = random.randrange(len(dataset))

        image, label = dataset[random_index]
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

        figure.canvas.draw_idle()

    # Create the button below the image.
    button_axis = figure.add_axes(
        [0.4, 0.05, 0.2, 0.08]
    )

    next_button = Button(
        button_axis,
        "Next",
    )

    next_button.on_clicked(
        lambda event: show_random_sample()
    )

    # Show the first random sample.
    show_random_sample()

    # Keep the window open.
    plt.show()
