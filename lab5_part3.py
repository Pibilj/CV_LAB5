import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Check if we can use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# loading and normalizing the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "fruit_images")

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = ImageFolder(root=data_path, transform=transform)
print(f"\nDataset loaded: {len(dataset)} images")
print(f"Classes: {dataset.classes}")
print(f"Number of classes: {len(dataset.classes)}")

# Stacking into tensors
labels = []
images = []
for i in range(len(dataset)):
    img, label = dataset[i]
    labels.append(label)
    images.append(img)

labels = torch.tensor(labels)
images = torch.stack(images)


# Making the stratisized split
x_train, x_test, y_train, y_test = train_test_split(
    images.numpy(),
    labels.numpy(),
    test_size=0.2,
    stratify=labels.numpy(),
    random_state=42,
)


# Converting x & y test and train back into tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# creating the test and train data sets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

print(f"\nTraining set: {len(train_dataset)} images")
print(f"Test set: {len(test_dataset)} images")

# Get number of classes dynamically
num_classes = len(dataset.classes)


# Defining the CNN architecture
class CNN(nn.Module):
    def __init__(self, N, M, num_classes=5, input_shape=(3, 64, 64), kernal_size=3):
        super(CNN, self).__init__()

        # making a module list for the convolution layers so we can have a different amount of them
        self.convs = nn.ModuleList()
        in_channels = input_shape[0]

        # Here we dynamically create the ammount of convolution layers we want for the NN
        for i in N:
            conv_layer = nn.Conv2d(
                in_channels, i, kernel_size=kernal_size, padding=kernal_size // 2
            )
            self.convs.append(conv_layer)
            in_channels = i

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for conv in self.convs:
                x = self.pool(F.relu(conv(x)))
            self.flattened_size = x.view(1, -1).size(1)

        self.fcs = nn.ModuleList()
        in_features = self.flattened_size
        for out_features in M:
            self.fcs.append(nn.Linear(in_features, out_features))
            in_features = out_features

        self.output = nn.Linear(in_features, num_classes)

    def forward(self, x):
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))

        x = x.view(x.size(0), -1)

        for fc in self.fcs:
            x = self.dropout(F.relu(fc(x)))

        x = self.dropout(x)
        x = self.output(x)
        return x


# Training function
def train_model(
    model, train_loader, test_loader, criterion, optimizer, epochs=50, device=device
):
    """
    Train the CNN model and track metrics.
    """
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        # TRAINING
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total

        # TESTING
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100.0 * correct / total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
            )

    return train_losses, test_losses, train_accs, test_accs


# Define 6 different CNN configurations to test
configurations = [
    {
        "name": "Config 1: 1 Conv Layer (16)",
        "N": [16],
        "M": [128],
        "kernel_size": 3,
        "lr": 0.001,
        "batch_size": 32,
        "optimizer": "Adam",
        "epochs": 50,
    },
    {
        "name": "Config 2: 2 Conv Layers (16, 32)",
        "N": [16, 32],
        "M": [256],
        "kernel_size": 3,
        "lr": 0.001,
        "batch_size": 32,
        "optimizer": "Adam",
        "epochs": 50,
    },
    {
        "name": "Config 3: 3 Conv Layers (16, 32, 64)",
        "N": [16, 32, 64],
        "M": [500],
        "kernel_size": 3,
        "lr": 0.01,
        "batch_size": 64,
        "optimizer": "SGD",
        "epochs": 50,
    },
    {
        "name": "Config 4: 2 Conv Layers (32, 64) - 5x5 Kernel",
        "N": [32, 64],
        "M": [256, 128],
        "kernel_size": 5,
        "lr": 0.001,
        "batch_size": 32,
        "optimizer": "Adam",
        "epochs": 50,
    },
    {
        "name": "Config 5: 3 Conv Layers (32, 64, 128)",
        "N": [32, 64, 128],
        "M": [256],
        "kernel_size": 3,
        "lr": 0.001,
        "batch_size": 16,
        "optimizer": "Adam",
        "epochs": 50,
    },
    {
        "name": "Config 6: 2 Conv Layers (16, 32) - SGD",
        "N": [16, 32],
        "M": [256, 128],
        "kernel_size": 3,
        "lr": 0.01,
        "batch_size": 32,
        "optimizer": "SGD",
        "epochs": 50,
    },
]

# Store results
results = []

print("\n" + "=" * 80)
print("RUNNING CNN EXPERIMENTS")
print("=" * 80)

for idx, config in enumerate(configurations):
    print(f"\n{'='*80}")
    print(f"Running {config['name']}")
    print(f"{'='*80}")

    # Create data loaders with specified batch size
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Create model
    model = CNN(
        N=config["N"],
        M=config["M"],
        num_classes=num_classes,
        input_shape=(3, 64, 64),
        kernal_size=config["kernel_size"],
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    # Train model
    train_losses, test_losses, train_accs, test_accs = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        epochs=config["epochs"],
        device=device,
    )

    # Store results
    results.append(
        {
            "config": config,
            "model": model,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "final_train_acc": train_accs[-1],
            "final_test_acc": test_accs[-1],
        }
    )

    print(f"\n{config['name']} - Final Results:")
    print(f"  Train Accuracy: {train_accs[-1]:.2f}%")
    print(f"  Test Accuracy: {test_accs[-1]:.2f}%")

# RESULTS SUMMARY
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Config':<45} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 70)
for idx, result in enumerate(results):
    config_name = result["config"]["name"]
    train_acc = result["final_train_acc"]
    test_acc = result["final_test_acc"]
    print(f"{config_name:<45} {train_acc:>10.2f}% {test_acc:>10.2f}%")

# Find best configuration
best_idx = np.argmax([r["final_test_acc"] for r in results])
best_config = results[best_idx]["config"]["name"]
best_test_acc = results[best_idx]["final_test_acc"]

print(f"\n🏆 Best Configuration: {best_config}")
print(f"   Test Accuracy: {best_test_acc:.2f}%")

# VISUALIZATION 1: Test Accuracy vs. Configuration
plt.figure(figsize=(12, 6))
config_names = [r["config"]["name"].split(":")[1].strip() for r in results]
test_accs_final = [r["final_test_acc"] for r in results]

plt.bar(range(len(results)), test_accs_final, color="skyblue", edgecolor="navy")
plt.xlabel("Configuration Index", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.title(
    "CNN Test Accuracy Comparison Across Configurations", fontsize=14, fontweight="bold"
)
plt.xticks(
    range(len(results)), [f"Config {i+1}" for i in range(len(results))], rotation=45
)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# VISUALIZATION 2: Loss vs. Epoch for top 2 configurations
top_2_indices = np.argsort([r["final_test_acc"] for r in results])[-2:]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, result_idx in enumerate(top_2_indices):
    result = results[result_idx]
    config_name = result["config"]["name"]

    axes[ax_idx].plot(result["train_losses"], label="Train Loss", linewidth=2)
    axes[ax_idx].plot(result["test_losses"], label="Test Loss", linewidth=2)
    axes[ax_idx].set_xlabel("Epoch", fontsize=11)
    axes[ax_idx].set_ylabel("Loss", fontsize=11)
    axes[ax_idx].set_title(
        f'{config_name}\nTest Acc: {result["final_test_acc"]:.2f}%', fontsize=10
    )
    axes[ax_idx].legend()
    axes[ax_idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# VISUALIZATION 3: Accuracy vs. Epoch for best configuration
best_result = results[best_idx]

plt.figure(figsize=(10, 6))
plt.plot(best_result["train_accs"], label="Train Accuracy", linewidth=2)
plt.plot(best_result["test_accs"], label="Test Accuracy", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title(f"Accuracy vs. Epoch - {best_config}", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# DETAILED EVALUATION OF BEST MODEL
print("\n" + "=" * 80)
print("DETAILED EVALUATION OF BEST MODEL")
print("=" * 80)

best_model = results[best_idx]["model"]
best_model.eval()

# Get predictions on test set
test_loader_eval = DataLoader(test_dataset, batch_size=64, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader_eval:
        batch_x = batch_x.to(device)
        outputs = best_model(batch_x)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Print classification metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("\nClassification Report:")
class_names = dataset.classes
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title(f"Confusion Matrix - {best_config}", fontsize=14, fontweight="bold")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# EXTRA CREDIT: Visualize feature maps
print("\n" + "=" * 80)
print("EXTRA CREDIT: FEATURE MAP VISUALIZATION")
print("=" * 80)

# Get a sample image from test set
sample_idx = 0
sample_img, sample_label = test_dataset[sample_idx]
sample_img = sample_img.unsqueeze(0).to(device)

# Extract feature maps from each convolutional layer
feature_maps = []
x = sample_img

best_model.eval()
with torch.no_grad():
    for i, conv in enumerate(best_model.convs):
        x = conv(x)
        x = F.relu(x)
        feature_maps.append(x.cpu().squeeze(0))
        x = best_model.pool(x)

# Visualize feature maps from first conv layer
num_filters = min(16, feature_maps[0].shape[0])
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for i in range(num_filters):
    axes[i].imshow(feature_maps[0][i], cmap="viridis")
    axes[i].set_title(f"Filter {i+1}", fontsize=9)
    axes[i].axis("off")

plt.suptitle(
    f"Feature Maps - Layer 1 (Sample: {class_names[sample_label]})",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("CNN EXPERIMENT COMPLETE!")
print("=" * 80)
