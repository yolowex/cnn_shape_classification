import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

data = torch.load("shapes_dataset.pt", weights_only=False)

images = data["images"].float() / 255.0
labels = data["labels"].long()
class_names = data["class_names"]

dataset = TensorDataset(images, labels)

train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
)

print(images.shape)
print(images.dtype)
print(images.min().item(), images.max().item())

unique_labels, counts = torch.unique(labels, return_counts=True)
print(unique_labels)
print(counts)
