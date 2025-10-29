"""
Part 2: Image Classification with Multilayer Perceptron (MLP)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if we can use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# STEP 1: DATA LOADING & PREPROCESSING

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "fruit_images")

# Transform: Resize to 64x64 and convert to tensor (automatically normalizes to [0,1])
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # Converts to [0, 1] range and changes to (C, H, W)
    ]
)

dataset = ImageFolder(root=data_path, transform=transform)
print(f"\nDataset loaded: {len(dataset)} images")
print(f"Classes: {dataset.classes}")
print(f"Number of classes: {len(dataset.classes)}")

# fetch all images and labels
images = []
labels = []

for i in range(len(dataset)):
    img, label = dataset[i]
    images.append(img)
    labels.append(label)

# Stack into tensors
images = torch.stack(images)  # Shape: (N, 3, 64, 64)
labels = torch.tensor(labels)

print(f"Images shape: {images.shape}")  # (N, 3, 64, 64)
print(f"Labels shape: {labels.shape}")  # (N,)

# Flatten images: (N, 3, 64, 64) → (N, 3*64*64) = (N, 12288)
# This is for MLP input (it expects 1D vectors, not 2D images)
images_flat = images.view(images.size(0), -1)
print(f"Flattened images shape: {images_flat.shape}")  # (N, 12288)

# Split into train/test (80/20) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    images_flat.numpy(),
    labels.numpy(),
    test_size=0.2,
    stratify=labels.numpy(),
    random_state=42,
)

print(f"\nTraining set: {len(X_train)} images")
print(f"Test set: {len(X_test)} images")

# Convert back to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# STEP 2: DEFINE MLP MODEL


class MLP(nn.Module):
    """
    Multilayer Perceptron for image classification.

    Architecture:
    Input → Hidden Layer 1 → ReLU → Hidden Layer 2 → ReLU → ... → Output → Softmax

    Parameters:
    - input_size: Flattened image dimension (e.g., 3*64*64 = 12288)
    - hidden_sizes: List of hidden layer sizes (e.g., [128, 64])
    - num_classes: Number of output classes (e.g., 5 for 5 fruits)
    """

    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()

        # Build layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # Non-linear activation
            prev_size = hidden_size

        # Output layer (no activation here, CrossEntropyLoss handles softmax)
        layers.append(nn.Linear(prev_size, num_classes))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


# STEP 3: TRAINING FUNCTION


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50):
    """
    Train the MLP model and track metrics.

    Returns:
    - train_losses: List of average training loss per epoch
    - test_losses: List of average test loss per epoch
    - train_accs: List of training accuracy per epoch
    - test_accs: List of test accuracy per epoch
    """
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        # TRAINING
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass & optimization
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total

        # TESTING
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # No gradient computation for testing
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

        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
            )

    return train_losses, test_losses, train_accs, test_accs


# STEP 4: HYPERPARAMETER EXPERIMENTS

# Define 6 different configurations to test
# Format: (hidden_sizes, learning_rate, batch_size, epochs)
configurations = [
    {
        "name": "Config 1: 1 layer (128)",
        "hidden_sizes": [128],
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 50,
    },
    {
        "name": "Config 2: 2 layers (128, 64)",
        "hidden_sizes": [128, 64],
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 50,
    },
    {
        "name": "Config 3: 2 layers (256, 128)",
        "hidden_sizes": [256, 128],
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 50,
    },
    {
        "name": "Config 4: 2 layers (128, 64) - High LR",
        "hidden_sizes": [128, 64],
        "lr": 0.01,
        "batch_size": 32,
        "epochs": 50,
    },
    {
        "name": "Config 5: 3 layers (256, 128, 64)",
        "hidden_sizes": [256, 128, 64],
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 50,
    },
    {
        "name": "Config 6: 2 layers (128, 64) - Large Batch",
        "hidden_sizes": [128, 64],
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 50,
    },
]

# Store results
results = []

input_size = X_train.shape[1]  # 12288
num_classes = len(dataset.classes)  # 5

print("\n" + "=" * 80)
print("RUNNING EXPERIMENTS")
print("=" * 80)

for idx, config in enumerate(configurations):
    print(f"\n{'='*80}")
    print(f"Running {config['name']}")
    print(f"{'='*80}")

    # Create data loaders with specified batch size
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Create model
    model = MLP(input_size, config["hidden_sizes"], num_classes).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Train model
    train_losses, test_losses, train_accs, test_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, epochs=config["epochs"]
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

# STEP 5: VISUALIZATION & ANALYSIS

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Print summary table
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

# Plot 1: Test Accuracy vs. Configuration
plt.figure(figsize=(12, 6))
config_names = [r["config"]["name"].split(":")[1].strip() for r in results]
test_accs_final = [r["final_test_acc"] for r in results]

plt.bar(range(len(results)), test_accs_final, color="skyblue", edgecolor="navy")
plt.xlabel("Configuration Index", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.title(
    "Test Accuracy Comparison Across Configurations", fontsize=14, fontweight="bold"
)
plt.xticks(
    range(len(results)), [f"Config {i+1}" for i in range(len(results))], rotation=45
)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Loss vs. Epoch for best 2 configurations
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

# Plot 3: Accuracy vs. Epoch for best configuration
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

# STEP 6: DETAILED EVALUATION OF BEST MODEL

print("\n" + "=" * 80)
print("DETAILED EVALUATION OF BEST MODEL")
print("=" * 80)

best_model = results[best_idx]["model"]
best_model.eval()

# Get predictions on test set
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = best_model(batch_x)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=dataset.classes,
    yticklabels=dataset.classes,
)
plt.title(f"Confusion Matrix - {best_config}", fontsize=14, fontweight="bold")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# STEP 7: SHOW MISCLASSIFICATIONS

print("\n" + "=" * 80)
print("MISCLASSIFIED EXAMPLES")
print("=" * 80)

# Find misclassified samples
misclassified_indices = np.where(all_preds != all_labels)[0]

# Show at least one misclassified example per class
shown_classes = set()
fig, axes = plt.subplots(1, min(5, len(misclassified_indices)), figsize=(15, 3))
if len(misclassified_indices) == 1:
    axes = [axes]

plot_count = 0
for idx in misclassified_indices:
    true_label = all_labels[idx]
    pred_label = all_preds[idx]

    if true_label not in shown_classes and plot_count < 5:
        # Get original image (unflatten)
        img = X_test[idx].view(3, 64, 64).permute(1, 2, 0).numpy()

        axes[plot_count].imshow(img)
        axes[plot_count].set_title(
            f"True: {dataset.classes[true_label]}\nPred: {dataset.classes[pred_label]}",
            fontsize=10,
            color="red",
        )
        axes[plot_count].axis("off")

        shown_classes.add(true_label)
        plot_count += 1

    if plot_count >= 5:
        break

plt.suptitle("Misclassified Examples", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nTotal misclassifications: {len(misclassified_indices)} / {len(all_labels)}")
print(f"Error rate: {100 * len(misclassified_indices) / len(all_labels):.2f}%")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE!")
print("=" * 80)
