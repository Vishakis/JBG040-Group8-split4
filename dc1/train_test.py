from tqdm import tqdm
import torch
from dc1.net import Net
from dc1.batch_sampler import BatchSampler
from typing import Callable, List
import torchvision.models as models

def train_model(
        model: models.resnet50(pretrained=True),
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,     # Added scheduler here
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Let's keep track of all the losses:
    losses = []
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)
        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()

    # Update learning rate after all batches are processed (at the end of each epoch) - currently step is 10
    scheduler.step()
    return losses


def test_model(
        model: models.resnet50(pretrained=True),
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []
    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)
    return losses
