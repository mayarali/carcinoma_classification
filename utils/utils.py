import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
import torch.functional as F
from PIL import Image
from torchvision.io import read_image
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import os
import torch.nn as nn
import umap


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, one_hot=True):
        self.image_paths = image_paths
        if labels is None:
            self.labels = [np.nan] * len(image_paths)
        elif one_hot:
            self.labels = torch.LongTensor(np.eye(len(set(labels)))[labels])
        else:
            self.labels = labels
        self.transform = transform
        # self.images = [read_image(path) for path in image_paths]
        # self.images = [cv2.imread(path) for path in image_paths]
        self.images = [Image.open(path) for path in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.images[index]
        if self.labels is not None:
            label = self.labels[index]
        else:
            label = np.nan

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_image_and_label_paths(image_dir, image_format="png"):
    image_paths = [
        f"{image_dir}/{fname}"
        for fname in os.listdir(image_dir)
        if fname.split(".")[-1] == image_format
    ]
    label_path = f"{image_dir}/labels.csv"
    return image_paths, label_path


def get_train_test_paths(image_paths, train_ids):
    train_set_paths = [
        path
        for path in image_paths
        for train_id in train_ids
        if path.split("_")[-1].split(".")[0] == str(train_id)
    ]
    test_set_paths = list(set(image_paths) - set(train_set_paths))
    return train_set_paths, test_set_paths


def prepare_train_test_data(
    image_paths,
    train_ids,
    labels,
    train_transforms=None,
    test_transforms=None,
    train_batch_size=64,
    test_batch_size=1024,
    num_workers=8,
    **kwargs,
):
    train_set_paths, test_set_paths = get_train_test_paths(image_paths, train_ids)
    train_set, train_loader = prepare_data(
        train_set_paths,
        labels=labels,
        transforms=train_transforms,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        **kwargs,
    )
    test_set, test_loader = prepare_data(
        test_set_paths,
        transforms=test_transforms,
        batch_size=test_batch_size,
        num_workers=num_workers,
        shuffle=False,
        **kwargs,
    )
    return train_set, train_loader, test_set, test_loader


def prepare_data(
    image_paths,
    labels=None,
    transforms=None,
    batch_size=64,
    num_workers=8,
    shuffle=False,
    **kwargs,
):
    dataset = CustomDataset(image_paths, labels=labels, transform=transforms, **kwargs)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataset, data_loader


@torch.no_grad()
def extract_feature_sets(
    feature_extractor, data_loader, num_samples_target=None, train=True, device="cuda:0"
):
    feature_extractor.eval()
    if not train or num_samples_target is None:
        num_samples_target = len(data_loader.dataset)

    X = []
    y = []
    num_samples_current = 0
    for batch in itertools.cycle(data_loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        X.append(feature_extractor(images).detach().cpu().numpy())
        y.append(labels.detach().cpu().numpy())

        num_samples_current += X[-1].shape[0]
        if num_samples_current >= num_samples_target:
            break
    X = np.concatenate(X)
    y = np.concatenate(y)
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")
    return X, y


def prepare_data_sklearn(image_paths, labels=None, transforms=None):
    gray_scale = T.Grayscale()
    dataset = CustomDataset(image_paths, labels=labels, transform=transforms)
    X = np.concatenate(
        [gray_scale(img).numpy().flatten()[None, :] for img, label in dataset]
    )
    y = np.concatenate([np.array(label)[None, :] for img, label in dataset])
    print(f"X.shape: {X.shape}, y.shape: {y.shape}")
    return X, y


def run_sklearn_classifier(classifier, X_train, y_train, X_test, cv=5):
    scorer = make_scorer(f1_score, average="micro")
    # Evaluate the pipeline using cross-validation and mean F1 score
    scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring=scorer)
    mean_f1 = np.mean(scores)
    print(f"F1 scores: {scores}")
    print(f"Mean F1 score: {mean_f1}")
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    train_f1 = f1_score(y_train, y_train_pred, average="micro")
    print(f"Train F1 score: {train_f1}")
    return y_train_pred, y_test_pred


def create_submission_file(
    test_set_paths, y_test_pred, submission_file_path="submission.csv"
):
    ids = [int(path.split("_")[-1].split(".")[0]) for path in test_set_paths]
    submission = pd.DataFrame({"id": ids, "malignant": y_test_pred})
    submission.to_csv(submission_file_path, index=False)

    # show label distribution
    analyze_class_distribution(submission)
    return submission


def analyze_class_distribution(labels_df):
    # bring into different form
    # expanded_df = pd.get_dummies(labels_df["malignant"])
    # # rename columns
    # print(expanded_df.columns)
    # expanded_df.columns = ["negative", "benign", "malignant"]
    # print(expanded_df.columns)
    # class_counts = expanded_df.iloc[:, 0:].sum()

    class_counts = labels_df["malignant"].value_counts()
    print(class_counts.index)
    # class_counts.index = ["negative", "benign", "malignant"]
    # define index mapping
    index_mapping = {-1: "negative", 0: "benign", 1: "malignant"}
    # map index
    class_counts.index = class_counts.index.map(index_mapping)
    print(class_counts.index)
    # Plot the bar plot
    class_counts.plot(kind="bar")
    # rotate x-labels
    plt.xticks(rotation=45)

    # Set labels and title
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.title("Class ID Counts")

    # Show the plot
    plt.show()


class FeatureExtractor:
    def __init__(self, model):
        self.model = model

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)

    def __call__(self, x):
        result = self.model(x)
        return result["out"]


def visualize_umap(features, labels):
    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(features)

    # Create a scatter plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="viridis")
    plt.colorbar()

    # Set labels and title
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("UMAP Visualization")

    # Show the plot
    plt.show()


def check_data(data_dir="./data"):
    transform = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True)])

    image_dir = f"{data_dir}/oxml-carinoma-classification"
    image_paths, label_path = get_image_and_label_paths(image_dir)
    image_paths[0:1], len(image_paths)

    labels_df = pd.read_csv(label_path)
    labels = labels_df["malignant"].tolist()
    train_ids = labels_df["id"].tolist()
    train_set_paths, test_set_paths = get_train_test_paths(
        image_paths, train_ids=train_ids
    )
    len(train_set_paths), len(test_set_paths)

    train_set, train_loader, test_set, test_loader = prepare_train_test_data(
        image_paths,
        train_ids,
        labels,
        train_transforms=transform,
        test_transforms=transform,
        train_batch_size=64,
        test_batch_size=1024,
        num_workers=8,
        one_hot=False,
    )
    return train_set, train_loader, test_set, test_loader


class ClassificationHead(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=3, dropout_prob=0.0):
        super().__init__()
        self.fc_1 = nn.Linear(in_features, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, num_classes)
        self.dropout_1 = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.dropout_1(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.fc_2(x)
        return x