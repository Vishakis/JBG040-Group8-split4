import plotext

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.train_test import train_model, test_model
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
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

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
print(f"Current CUDA Device: {torch.cuda.current_device()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")


def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load train and test datasets
    train_dataset = ImageDataset(
        Path("data/X_train.npy"),
        Path("data/Y_train.npy"),
        preprocess=args.preprocess,
        augment=True       #TRUE/FALSE for data augmentation - 50% to flip and rotate with -+20 degrees
    )
    test_dataset = ImageDataset(
        Path("data/X_test.npy"),
        Path("data/Y_test.npy"),
        preprocess=args.preprocess,
        augment=False           #NO AUGMENTATION FOR TESTING
    )

    # ----------------------------------------------------------------
    # Replace given model with a ResNet-50 (pretrained on ImageNet)
    # ----------------------------------------------------------------
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features  # typically 2048 for ResNet-50
    model.fc = nn.Linear(num_ftrs, 6)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()
    print("Using loss function:", loss_function.__class__.__name__)

    n_epochs = args.nb_epochs  # default 10 epochs
    batch_size = args.batch_size

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
            plotext.xticks([i for i in range(len(mean_losses_train))])  # Adjust x-ticks range appropriately
            plotext.show()

            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, F2 Score = {f2:.4f}")

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

    # Compute confusion matrix and other metrics
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

    # Generate and save a confusion matrix plot using your plot_confusion_matrix function
    print("\n=== Generating and saving confusion matrix plot... ===")

    def generate_confusion_matrix(model, dataset, device):
        """
        Generate a confusion matrix to be saved with the function below.
        """
        y_pred = []
        y_true = []

        # Ensure the model is in evaluation mode
        model.eval()
        with torch.no_grad():
            # Iterate through the dataset
            for i in range(len(dataset)):
                data, label = dataset[i]
                data = data.unsqueeze(0).to(device)  # Add batch dimension
                out = model(data)
                pred = torch.argmax(out, dim=1)
                y_pred.append(pred.cpu().item())
                y_true.append(label)

        # Generate the confusion matrix
        class_names = list(range(max(y_true) + 1))  # Assuming class labels are contiguous
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        return cm, class_names

    def plot_save_confusion_matrix(cm, class_names, file_path="confusion_matrix.png"):
        """
        Plot and save confusion matrix using matplotlib.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        # Create a heatmap
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="black")

        # Add labels, title, and axes
        plt.xticks(ticks=range(len(class_names)), labels=class_names)
        plt.yticks(ticks=range(len(class_names)), labels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        # Save the plot
        plt.savefig(file_path)
        plt.close(fig)
        return file_path

    try:
        print("Generating confusion matrix...")
        cm, class_names = generate_confusion_matrix(model, test_dataset, device)

        # Specify save path
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        cm_path = os.path.join(output_dir, "confusion_matrix.png")

        # Plot and save confusion matrix
        plot_save_confusion_matrix(cm, class_names, file_path=cm_path)
        print(f"Confusion matrix saved successfully at: {cm_path}")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", help="number of training iterations", default=30, type=int)
    parser.add_argument("--batch_size", help="batch size", default=25, type=int)
    parser.add_argument("--balanced_batches", help="balance batches for class labels", default=True, type=bool)
    parser.add_argument("--evaluate_only", help="skip training and only evaluate an existing model", action="store_true")
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

    args.preprocess = "hist_eq"  # adjust as needed "hist_eq","clahe","hist_eq_gaussian","none"
    main(args)