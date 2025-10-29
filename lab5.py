import numpy as np
import os
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision import transforms

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# fetching and grayscaling the images from the dataset

data_path = "./fruit_images"

download_flag = not os.path.exists(data_path)

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

dataset = ImageFolder(root=data_path, transform=transform)

N = len(dataset)
imgs_gray = []
labels = []

for i in range(N):
    x, y = dataset[i]
    img_rgb = x.permute(1, 2, 0).numpy()
    img_gray = rgb2gray(img_rgb)
    imgs_gray.append(img_gray)
    labels.append(dataset.classes[y])

imgs_gray = np.stack(imgs_gray, axis=0)
print("Loaded greyscale images shape:", imgs_gray.shape)


# This method is for showing 10 of the images and their label
def show_images_grid(
    imgs,
    labels=None,
    rows=2,
    cols=5,
    title="Our Dataset Samples",
    display_size=(32, 32),
):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < len(imgs):
            img_resized = resize(imgs[i], display_size, anti_aliasing=True)
            if img_resized.ndim == 2:
                ax.imshow(img_resized, cmap="gray")
            if labels is not None:
                ax.set_title(labels[i], fontsize=9)
    plt.tight_layout()
    plt.show()


show_images_grid(imgs_gray[:10], labels[:10], rows=2, cols=5, display_size=(64, 64))

# computing the hog

features_list = []
hog_imgs = []

for i in range(len(imgs_gray)):
    feat, hog_img = hog(
        imgs_gray[i],
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
    )
    print(feat.shape)
    features_list.append(feat)
    hog_imgs.append(exposure.rescale_intensity(hog_img, in_range=(0, hog_img.max())))

print("number of images processed:", len(features_list))
print("HOG feature length (first image):", len(features_list[0]))


# displaying the HOGed images
def show_hog_grid(hog_images, rows=2, cols=5, title="HOG Renderings"):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < len(hog_images):
            ax.imshow(hog_images[i], cmap="gray")
    plt.tight_layout()
    plt.show()


show_hog_grid(hog_imgs[:10], rows=2, cols=5)

# here is stratisfying the data into a 80 train 20 test split
labels = []
for i in range(N):
    _, lab = dataset[i]
    labels.append(lab)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(
    features_list, labels, test_size=0.2, stratify=labels, random_state=42
)

# here is the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Define K values to test
k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

print("\n" + "=" * 60)
print("K-Sweep Evaluation")
print("=" * 60)

# Test each K value on the same train/test split
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(x_train, y_train)
    y_pred_k = knn_k.predict(x_test)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)
    print(f"K={k}: Test Accuracy = {acc:.4f} ({acc*100:.2f}%)")

# Find the best K
best_k_idx = np.argmax(accuracies)
best_k = k_values[best_k_idx]
best_accuracy = accuracies[best_k_idx]

print("\n" + "-" * 60)
print(f"Best K: {best_k} with accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("-" * 60)

# Plot Accuracy vs. K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker="o", linewidth=2, markersize=8)
plt.xlabel("K Value", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title("KNN Accuracy vs. K Value", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])

# Highlight the best K
plt.axvline(x=best_k, color="r", linestyle="--", alpha=0.5, label=f"Best K={best_k}")
plt.scatter(
    [best_k],
    [best_accuracy],
    color="red",
    s=150,
    zorder=5,
    label=f"Best Accuracy={best_accuracy:.4f}",
)
plt.legend()
plt.tight_layout()
plt.show()

# Re-train with the best K and report final test accuracy
print("\n" + "=" * 60)
print(f"Final Model: Re-training with Best K={best_k}")
print("=" * 60)

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(x_train, y_train)
final_y_pred = final_knn.predict(x_test)
final_accuracy = accuracy_score(y_test, final_y_pred)

print(
    f"\nFinal Test Accuracy with K={best_k}: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)"
)
print("\nDetailed Classification Report:")
print(classification_report(y_test, final_y_pred, target_names=dataset.classes))
