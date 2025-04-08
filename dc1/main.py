import plotext

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.train_test import train_model, test_model
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Needed for grad-cam interpolation and softmax
import torchvision.models as models  # Using ResNet-50 from torchvision
from torchsummary import summary  # type: ignore

# Other imports
import os
import itertools
import argparse
from datetime import datetime
from pathlib import Path
from typing import List

# Additional imports for evaluation metrics
import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, confusion_matrix
import random

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
print(f"Current CUDA Device: {torch.cuda.current_device()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")


# ----------------------------
# Grad-CAM Implementation
# ----------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate_cam(self, input_image: torch.Tensor, target_class: int = None):
        """
        Generates the Grad-CAM for the given input image.
        Args:
            input_image (torch.Tensor): Input image tensor of shape (1,3,224,224)
            target_class (int, optional): Class index to compute CAM for. If None, uses predicted class.
        Returns:
            cam (np.ndarray): The computed CAM of shape (H, W)
            target_class (int): The class index used.
        """
        self.model.zero_grad()
        output = self.model(input_image)  # Forward pass
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        loss = output[0, target_class]
        loss.backward()  # Backward pass

        # Retrieve gradients and activations from target layer
        gradients = self.gradients[0]  # shape: [C, H, W]
        activations = self.activations[0]  # shape: [C, H, W]

        # Global Average Pooling on gradients: compute channel-wise weights
        weights = torch.mean(gradients, dim=(1, 2))
        # Compute weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = F.relu(cam)
        # Normalize CAM to [0,1]
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        return cam.cpu().detach().numpy(), target_class


def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load train and test datasets
    train_dataset = ImageDataset(
        Path("data/X_train.npy"),
        Path("data/Y_train.npy"),
        preprocess=args.preprocess,
        augment=True  # TRUE/FALSE for data augmentation - adjusted in image_dataset.py
    )
    test_dataset = ImageDataset(
        Path("data/X_test.npy"),
        Path("data/Y_test.npy"),
        preprocess=args.preprocess,
        augment=False  # NO AUGMENTATION FOR TESTING
    )

    # ----------------------------------------------------------------
    # Replace given model with a ResNet-50 (pretrained on ImageNet)
    # ----------------------------------------------------------------
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features  # typically 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),  # Dropout layer (20% probability)
        nn.Linear(num_ftrs, out_features=6)  # Fully connected layer
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.15)
    loss_function = nn.CrossEntropyLoss()
    print("Using loss function:", loss_function.__class__.__name__)

    n_epochs = args.nb_epochs  # default 12 epochs
    batch_size = args.batch_size  # default 15

    # Device setup
    if torch.cuda.is_available():
        print("CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        summary(model, (3, 224, 224), device=device)
    elif torch.backends.mps.is_available():
        print("Apple Silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        device = "cpu"
        print("No GPU found, training on CPU...")
        summary(model, (3, 224, 224), device=device)

    # Verify a sample from the dataset
    sample_img, _ = train_dataset[0]
    print("Sample image shape:", sample_img.shape)  # Expect torch.Size([3, 224, 224])

    # Set up batch samplers
    train_sampler = BatchSampler(batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches)
    test_sampler = BatchSampler(batch_size=100, dataset=test_dataset, balanced=args.balanced_batches)

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []

    if not args.evaluate_only:
        for epoch in range(n_epochs):
            # Train one epoch
            train_losses = train_model(model, train_sampler, optimizer, loss_function, device)
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_loss = float(avg_train_loss.cpu())  # Convert to CPU and float
            mean_losses_train.append(avg_train_loss)

            # Test one epoch
            test_losses = test_model(model, test_sampler, loss_function, device)
            avg_test_loss = sum(test_losses) / len(test_losses)
            avg_test_loss = float(avg_test_loss.cpu())  # Convert to CPU and float
            mean_losses_test.append(avg_test_loss)

            # Step the scheduler after each epoch
            scheduler.step()

            # Calculate F2 score after each epoch
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for x, y in test_sampler:
                    x = x.to(device)
                    outputs = model(x)
                    _, preds = torch.max(outputs, 1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(y.cpu().numpy())
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            f2 = fbeta_score(all_labels, all_preds, beta=2, average='weighted')

            # Plot training and testing losses
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title(f"Train and test loss (Epoch {epoch + 1}, F2: {f2:.4f})")
            plotext.xticks([i for i in range(len(mean_losses_train))])
            plotext.show()

            print(
                f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, F2 Score = {f2:.4f}")

        final_train_loss = sum(mean_losses_train) / len(mean_losses_train)
        final_test_loss = sum(mean_losses_test) / len(mean_losses_test)
        print("\nFinal Average Training Loss:", final_train_loss)
        print("Final Average Testing Loss:", final_test_loss)

        # Save the model
        now = datetime.now()
        os.makedirs("model_weights", exist_ok=True)
        model_path = f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

        # Save loss plot
        plt.figure(figsize=(9, 10), dpi=80)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(range(1, 1 + n_epochs), mean_losses_train, label="Train", color="blue")
        ax2.plot(range(1, 1 + n_epochs), mean_losses_test, label="Test", color="red")
        fig.legend()


    elif args.model_path:
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)

    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    loss_path = f"artifacts/session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png"
    fig.savefig(loss_path)
    print(f"Loss plot saved to: {loss_path}")

    # ----------------------------
    # Evaluation Metrics Calculation FOR LAST EPOCH
    # ----------------------------
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_sampler:
            x = x.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    f2 = fbeta_score(all_labels, all_preds, beta=2, average='weighted')
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    f1_score = fbeta_score(all_labels, all_preds, beta=1, average='weighted')

    print("\nEvaluation Metrics on Test Set:")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"F2 Score: {f2:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

    # ----------------------------
    # New: Compute Softmax Confidence Intervals and Entropy-Based Uncertainty
    # ----------------------------
    model.eval()
    all_softmax = []
    all_sample_entropies = []
    with torch.no_grad():
        for x, y in test_sampler:
            x = x.to(device)
            outputs = model(x)
            # Compute softmax probabilities
            softmax_outputs = F.softmax(outputs, dim=1)
            all_softmax.append(softmax_outputs.cpu().numpy())
            # Compute entropy for each sample: -sum(p*log(p))
            entropy = -torch.sum(softmax_outputs * torch.log(softmax_outputs + 1e-8), dim=1)
            all_sample_entropies.append(entropy.cpu().numpy())
    all_softmax = np.concatenate(all_softmax, axis=0)
    all_sample_entropies = np.concatenate(all_sample_entropies, axis=0)

    # Bootstrap confidence intervals for the mean probability per class
    num_bootstrap = 1000
    bootstrap_means = np.zeros((num_bootstrap, all_softmax.shape[1]))
    for i in range(num_bootstrap):
        indices = np.random.choice(len(all_softmax), len(all_softmax), replace=True)
        bootstrap_sample = all_softmax[indices]
        bootstrap_means[i] = bootstrap_sample.mean(axis=0)
    ci_lower = np.percentile(bootstrap_means, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_means, 97.5, axis=0)
    mean_softmax = all_softmax.mean(axis=0)

    print("\nSoftmax Probabilities (Mean over Test Set) with 95% Confidence Intervals:")
    for i in range(all_softmax.shape[1]):
        print(f"Class {i}: {ci_lower[i]:.3f} - {ci_upper[i]:.3f}")

    avg_entropy = np.mean(all_sample_entropies)
    print(f"\nAverage Entropy over Test Samples: {avg_entropy:.3f}")

    # ----------------------------
    # Visualization 1: Bar Chart for Overall Accuracy and F2 Score
    # ----------------------------
    plt.figure(figsize=(6, 4))
    overall_metrics = [acc, f2]
    overall_labels = ['Overall Accuracy', 'Overall F2 Score']
    plt.bar(overall_labels, overall_metrics, color=['blue', 'green'], alpha=0.7)
    plt.ylim(0, 1)
    plt.title("Overall Model Performance (Final Epoch)")
    plt.ylabel("Score")
    overall_bar_path = "artifacts/overall_performance.png"
    plt.savefig(overall_bar_path)
    plt.close()
    print(f"Overall performance bar chart saved to: {overall_bar_path}")

    # ----------------------------
    # Visualization 2: Radar Chart for Per-Class Accuracy and F2 Score
    # ----------------------------
    # Compute per-class accuracy (as recall per class) from the confusion matrix
    per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    # Compute per-class F2 score
    per_class_f2 = fbeta_score(all_labels, all_preds, beta=2, average=None)

    def plot_radar(categories, values1, values2, title, file_path):
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # complete the loop
        values1 = list(values1)
        values1 += values1[:1]
        values2 = list(values2)
        values2 += values2[:1]
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories)
        ax.plot(angles, values1, linewidth=2, linestyle='solid', label="Accuracy")
        ax.fill(angles, values1, alpha=0.25)
        ax.plot(angles, values2, linewidth=2, linestyle='solid', label="F2 Score")
        ax.fill(angles, values2, alpha=0.25)
        plt.title(title)
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.savefig(file_path)
        plt.close()

    categories = [f"Class {i}" for i in range(len(per_class_accuracy))]
    radar_chart_path = "artifacts/per_class_radar.png"
    plot_radar(categories, per_class_accuracy, per_class_f2, "Per-Class Performance: Accuracy and F2 Score",
               radar_chart_path)
    print(f"Per-class performance radar chart saved to: {radar_chart_path}")

    # ----------------------------
    # Generate and save a confusion matrix plot
    # ----------------------------
    print("\n=== Generating and saving confusion matrix plot... ===")

    def generate_confusion_matrix(model, dataset, device):
        """
        Generate a confusion matrix to be saved with the function below.
        """
        y_pred = []
        y_true = []
        model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                data, label = dataset[i]
                data = data.unsqueeze(0).to(device)
                out = model(data)
                pred = torch.argmax(out, dim=1)
                y_pred.append(pred.cpu().item())
                y_true.append(label)
        class_names = list(range(max(y_true) + 1))
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        return cm, class_names

    def plot_save_confusion_matrix(cm, class_names, file_path="confusion_matrix.png"):
        """
        Plot and save confusion matrix using matplotlib.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="black")
        plt.xticks(ticks=range(len(class_names)), labels=class_names)
        plt.yticks(ticks=range(len(class_names)), labels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(file_path)
        plt.close(fig)
        return file_path

    try:
        print("Generating confusion matrix...")
        cm, class_names = generate_confusion_matrix(model, test_dataset, device)
        output_dir = "output_confusion_matrix"
        os.makedirs(output_dir, exist_ok=True)
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_save_confusion_matrix(cm, class_names, file_path=cm_path)
        print(f"Confusion matrix saved successfully at: {cm_path}")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        import traceback
        traceback.print_exc()

    # ----------------------------
    # Grad-CAM Visualization for Multiple Sample Test Images
    # ----------------------------
    # Create a GradCAM instance using the target layer (last conv layer in model.layer4)
    grad_cam_instance = GradCAM(model, target_layer=model.layer4[-1])

    # Define the number of samples you want to visualize, e.g., 5 random samples
    num_samples = 5
    sample_indices = random.sample(range(len(test_dataset)), num_samples)

    for idx in sample_indices:
        sample_img, sample_label = test_dataset[idx]
        sample_img_batch = sample_img.unsqueeze(0).to(device)
        cam, predicted_class = grad_cam_instance.generate_cam(sample_img_batch)

        # Upsample CAM from its native resolution (e.g., 7x7) to 224x224
        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,H,W)
        cam_resized = F.interpolate(cam_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        cam_resized = cam_resized.squeeze().numpy()

        # Convert sample image to numpy for visualization.
        sample_img_np = sample_img.cpu().permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(sample_img_np, cmap='gray')
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)  # Overlay the heatmap
        plt.title(f"Grad-CAM Overlay (Predicted Class: {predicted_class})")
        gradcam_path = f"artifacts/gradcam_{idx}.png"
        plt.savefig(gradcam_path)
        plt.close()
        print(f"Grad-CAM visualization for sample {idx} saved to: {gradcam_path}")

    # Remove hooks after processing all samples
    grad_cam_instance.remove_hooks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", help="number of training iterations", default=12, type=int)
    parser.add_argument("--batch_size", help="batch size", default=15, type=int)
    parser.add_argument("--balanced_batches", help="balance batches for class labels", default=True, type=bool)
    parser.add_argument("--evaluate_only", help="skip training and only evaluate an existing model",
                        action="store_true")
    parser.add_argument("--model_path", help="path to a saved model for evaluation", type=str, default=None)
    parser.add_argument(
        "--preprocess",
        help="Preprocessing method to apply (hist_eq, clahe, hist_eq_gaussian, none)",
        type=str,
        choices=["hist_eq", "clahe", "hist_eq_gaussian", "none"],
        default="none",
    )
    parser.add_argument("--augment", help="Enable data augmentation (flip & rotate)", action="store_true")
    args = parser.parse_args()

    args.preprocess = "hist_eq"  # adjust as needed: "hist_eq", "clahe", "hist_eq_gaussian", "none"
    main(args)
