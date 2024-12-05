import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transform, split=None):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            root (str): Root directory for images.
            transform (callable): Transformations to apply to the images.
            split (str, optional): Dataset split to filter (e.g., "Train", "Validation", "Test").
        """
        df = pd.read_csv(csv_file)

        # Filter by split if provided
        if split is not None:
            df = df[df['split'] == split]

        # Ensure we only keep valid emotion classes
        df = df[df['emotion'].notna()]  # Filter out rows with NaN emotion
        self.paths = df['image_path'].tolist()  # List of image paths
        self.targets = df['emotion'].to_numpy(dtype=np.int64)  # Emotion labels
        self.valence_arousal = df[['valence', 'arousal']].to_numpy(dtype=np.float32)  # Valence and arousal
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
    # Normalize and use image path directly without appending root
        img_path = os.path.normpath(self.paths[idx])  # Normalize path to avoid redundant slashes
        if img_path.startswith('./') or img_path.startswith('../'):  # If relative, append root
            img_path = os.path.join(self.root, img_path.lstrip('./'))

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Get labels
        emotion_label = self.targets[idx]
        valence = torch.tensor(self.valence_arousal[idx, 0], dtype=torch.float32)
        arousal = torch.tensor(self.valence_arousal[idx, 1], dtype=torch.float32)

        return img, (emotion_label, valence, arousal)

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, class_weights, concordance_loss_fn, num_classes, device):
        """
        Args:
            class_weights (dict): Class weights for emotion labels.
            concordance_loss_fn (callable): Loss function for valence and arousal (e.g., Concordance Correlation Coefficient Loss).
            num_classes (int): Number of emotion classes.
            device (torch.device): Device to move weights to (e.g., "cuda" or "cpu").
        """
        super(MultiTaskLossWrapper, self).__init__()
        self.num_classes = num_classes
        self.device = device

        # Initialize losses
        weights = torch.FloatTensor(list(class_weights.values())).to(device)
        self.loss_emotions = nn.CrossEntropyLoss(weight=weights)
        self.loss_valence = concordance_loss_fn
        self.loss_arousal = concordance_loss_fn

    def forward(self, preds, target):
        """
        Args:
            preds (torch.Tensor): Model predictions.
            target (tuple): Ground truth labels (emotion, valence, arousal).
        Returns:
            torch.Tensor: Combined multi-task loss.
        """
        # Compute individual losses
        loss_emotions = self.loss_emotions(preds[:, :self.num_classes], target[0])
        loss_valence = self.loss_valence(preds[:, self.num_classes], target[1])
        loss_arousal = self.loss_arousal(preds[:, self.num_classes + 1], target[2])

        # Combine losses
        return loss_emotions + (loss_valence + loss_arousal) * 1  # Adjust weight if needed
