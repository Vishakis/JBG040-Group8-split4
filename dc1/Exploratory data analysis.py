import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

import matplotlib.pyplot as plt

# Data for plotting
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
train_loss = [1.7565, 1.6767, 1.6151, 1.5644, 1.5186, 1.4915, 1.4494, 1.4407, 1.4307, 1.4286, 1.4320, 1.4237]
test_loss = [1.6930, 1.6284, 1.5813, 1.5607, 1.5225, 1.5099, 1.4833, 1.4866, 1.4798, 1.4837, 1.4760, 1.4759]

# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Training Loss", marker='o')
plt.plot(epochs, test_loss, label="Testing Loss", marker='o')

# Adding titles and labels
plt.title('Training and Testing Loss over 12 Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()


# ------------------------------------------------------------------------------
# 1. Inspect .npy files in a specified folder
# ------------------------------------------------------------------------------
data_folder = r"C:\Users\20232369\OneDrive\Y2\Q3\DC1\dc1\dc1\data"  # Change as needed
file_list = glob.glob(os.path.join(data_folder, "*.npy"))

for file_path in file_list:
    file_name = os.path.basename(file_path)
    data = np.load(file_path, allow_pickle=True)
    print(f"File: {file_name}")

    if isinstance(data, dict):
        print("Keys in dictionary:", list(data.keys()))
        if 'label' in data:
            print("Label:", data['label'])
    else:
        print("Shape:", data.shape)
        print("Data type:", data.dtype)
        print("First few entries:", list(data.flat)[:5])
    print("-" * 40)

# ------------------------------------------------------------------------------
# 2. Count how many .npy files each label has in the training directory
# ------------------------------------------------------------------------------
data_dir = r"path_to_train_directory"  # Replace with the actual path
labels = ["Pneumothorax", "Nodule", "Infiltration", "Effusion", "Atelectasis", "NoFinding"]

for label in labels:
    folder_path = os.path.join(data_dir, label)
    if os.path.isdir(folder_path):
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        count = len(npy_files)
        print(f"{label}: {count} .npy files in the training set.")
    else:
        print(f"Folder for label '{label}' does not exist in {data_dir}.")

# ------------------------------------------------------------------------------
# 3. Load X_train, X_test, Y_train, and Y_test, then inspect shapes
# ------------------------------------------------------------------------------
X_train = np.load(r"C:\Users\20232369\OneDrive\Y2\Q3\DC1\dc1\dc1\data\X_train.npy", allow_pickle=True)
X_test = np.load(r"C:\Users\20232369\OneDrive\Y2\Q3\DC1\dc1\dc1\data\X_test.npy", allow_pickle=True)
Y_train = np.load(r"C:\Users\20232369\OneDrive\Y2\Q3\DC1\dc1\dc1\data\Y_train.npy", allow_pickle=True)
Y_test  = np.load(r"C:\Users\20232369\OneDrive\Y2\Q3\DC1\dc1\dc1\data\Y_test.npy", allow_pickle=True)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

# ------------------------------------------------------------------------------
# 4. Plot a few random samples from the training set
# ------------------------------------------------------------------------------
num_samples = 5
plt.figure(figsize=(15, 3))
for i in range(num_samples):
    idx = random.randint(0, X_train.shape[0] - 1)
    image = X_train[idx, 0, :, :]  # Remove channel dimension for visualization
    label = Y_train[idx]

    plt.subplot(1, num_samples, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
plt.show()

# ------------------------------------------------------------------------------
# 5. Print training label distribution and show a bar chart
# ------------------------------------------------------------------------------
unique_labels, counts = np.unique(Y_train, return_counts=True)
print("Training Label Distribution:")
for lbl, count in zip(unique_labels, counts):
    print(f"Label {lbl}: {count} samples")

plt.figure(figsize=(8, 4))
plt.bar(unique_labels, counts, tick_label=unique_labels)
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Training Label Distribution")
plt.show()

# ------------------------------------------------------------------------------
# 6. Pixel intensity statistics for X_train
# ------------------------------------------------------------------------------
print("Pixel Intensity Statistics for X_train:")
print("Min pixel value:", X_train.min())
print("Max pixel value:", X_train.max())
print("Mean pixel value:", X_train.mean())
print("Standard Deviation:", X_train.std())

# ------------------------------------------------------------------------------
# 7. Print label distributions (training vs. testing) with percentages
# ------------------------------------------------------------------------------
label_map = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "NoFinding",
    4: "Nodule",
    5: "Pneumothorax",
}

def print_label_distribution(y, dataset_name):
    unique, counts = np.unique(y, return_counts=True)
    total = np.sum(counts)
    print(f"{dataset_name} Label Distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / total) * 100
        class_name = label_map.get(label, "Unknown")
        print(f"Label {label} ({class_name}): {count} samples ({percentage:.2f}%)")
    print("-" * 40)

print_label_distribution(Y_train, "Training")
print_label_distribution(Y_test,  "Testing")

# ------------------------------------------------------------------------------
# 8. Visualize the label distributions side by side
# ------------------------------------------------------------------------------
unique_train, counts_train = np.unique(Y_train, return_counts=True)
unique_test, counts_test   = np.unique(Y_test, return_counts=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(unique_train, counts_train, tick_label=unique_train)
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Count")
axes[0].set_title("Training Label Distribution")

axes[1].bar(unique_test, counts_test, tick_label=unique_test)
axes[1].set_xlabel("Label")
axes[1].set_ylabel("Count")
axes[1].set_title("Testing Label Distribution")

plt.tight_layout()
plt.show()
