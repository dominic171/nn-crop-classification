import csv
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler


from models.temp_cnn import TempCNN
from models.rnn import RNN
from models.transformer_encoder import TransformerEncoder
from eurocrops import EuroCrops, EuroRegion
from utils.scheduled_optimizer import ScheduledOptim


def train_on_region(device, training_args):
    generator = torch.Generator().manual_seed(2024)
    train_dataset = EuroRegion(
        "train",
        training_args["train_region"],
        training_args["test_region"],
        training_args["path_train_data_h5"],
        training_args["path_test_data_h5"],
        training_args["path_train_labels_dir"],
        training_args["path_test_labels_dir"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args["batch_size"],
        sampler=RandomSampler(train_dataset, generator=generator),
        generator=generator,
    )

    test_dataset = EuroRegion(
        "test",
        training_args["train_region"],
        training_args["test_region"],
        training_args["path_train_data_h5"],
        training_args["path_test_data_h5"],
        training_args["path_train_labels_dir"],
        training_args["path_test_labels_dir"],
        label_encoder=train_dataset.label_encoder,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_args["batch_size"],
        sampler=SequentialSampler(test_dataset),
        generator=generator,
    )

    train(device, training_args, train_dataset, train_loader, test_dataset, test_loader)


def train(
    device, training_args, train_dataset, train_loader, test_dataset, test_loader
):
    if training_args["model_name"] == "TempCNN":
        model = TempCNN(
            train_dataset.input_dim,
            train_dataset.nclasses,
            train_dataset.sequence_length,
        )
        optimizer = torch.optim.Adam(model.parameters())
        loss_function = torch.nn.NLLLoss()
    elif training_args["model_name"] == "RNN":
        model = RNN(
            train_dataset.input_dim,
            train_dataset.nclasses,
            train_dataset.sequence_length,
        )
        optimizer = torch.optim.Adam(model.parameters())
        loss_function = torch.nn.NLLLoss()
    elif training_args["model_name"] == "Transformer":
        # transformer does not work on MPS (Mac GPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = TransformerEncoder(
            in_channels=train_dataset.input_dim,
            nclasses=train_dataset.nclasses,
            len_max_seq=train_dataset.sequence_length,
        )
        optimizer = ScheduledOptim(
            torch.optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98),
                eps=1e-09,
                weight_decay=0.0001,
            ),
            model.d_model,
            5,
        )
        loss_function = torch.nn.NLLLoss()
    else:
        raise ValueError("Invalid model name.")

    model.to(device)
    loss_function.to(device)

    num_epochs = training_args["num_epochs"]

    def optim_step():
        # ScheduledOptim has custom step function
        if isinstance(optimizer, ScheduledOptim):
            optimizer.step_and_update_lr()
        else:
            optimizer.step()

    # Training loop
    train_start_time = time.time()
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, (_, inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optim_step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100}")
                running_loss = 0.0

    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    print(f"Training took {train_time} seconds")

    # Save model
    model_path = os.path.join("results", f"{training_args['model_name']}.pt")
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Testing loop
    model.to(device)
    model.eval()

    all_recnos = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        test_start_time = time.time()
        for recnos, inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass to get outputs (log probabilities)
            outputs = model(inputs)

            # Convert log probabilities to actual class predictions
            _, predicted = torch.max(outputs, 1)

            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            recnos = [recno.item() for recno in recnos]
            all_recnos.extend(recnos)
        test_end_time = time.time()
        test_time = test_end_time - test_start_time
        print(f"Testing took {test_time} seconds")

    # Compute and save metrics
    metrics = compute_metrics(all_labels, all_predictions)
    metrics_path = os.path.join("results", f"metrics_{training_args['model_name']}.csv")
    os.makedirs("results", exist_ok=True)
    with open(metrics_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerows(metrics)

    # Save results
    all_labels_names = train_dataset.map_labels(all_labels)
    all_predictions_names = train_dataset.map_labels(all_predictions)
    all_results = zip(all_recnos, all_labels_names, all_predictions_names)
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", f"results_{training_args['model_name']}.csv")
    with open(results_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["recno", "Actual", "Predicted"])
        writer.writerows(all_results)


def compute_metrics(labels, predictions):
    acc = accuracy_score(labels, predictions)
    f1_class_mean = f1_score(labels, predictions, average="macro")
    f1_class_weighted = f1_score(labels, predictions, average="weighted")

    metrics = [
        ("Accuracy", acc),
        ("F1 (class mean)", f1_class_mean),
        ("F1 (class weighted)", f1_class_weighted),
    ]

    print(f"Accuracy: {acc}")
    print(f"F1 score (macro): {f1_class_mean}")
    print(f"F1 score (weighted): {f1_class_weighted}")

    return metrics
