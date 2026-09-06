import torch
from torch.utils.data import TensorDataset, DataLoader


data = torch.load("shapes_dataset.pt", weights_only=False)

images = data["images"]
labels = data["labels"]
class_names = data["class_names"]

dataset = TensorDataset(images, labels)
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)

images_batch, labels_batch = next(iter(loader))

if __name__ == '__main__':


    print(class_names)
    # ['circle', 'oval', 'line', 'rectangle', 'square', 'triangle']

    print(images_batch.shape)
    print(labels_batch.shape)

    # Convert a numeric label back to its shape name.
    label = labels_batch[0].item()
    print(class_names[label])
