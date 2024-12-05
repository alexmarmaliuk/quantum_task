import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.plotting import *

def training_loop(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epochs=25,
    batch_size=8,
    output=True,
    plot=True,
    
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_accuracy_stat = []
    test_accuracy_stat = []
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for sentences, tags in train_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            optimizer.zero_grad()
            loss = model.forward_crf(sentences, lengths=torch.tensor([len(s) for s in sentences]).to(device), tags=tags)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            predicted = model.forward_crf(sentences, lengths=torch.tensor([len(s) for s in sentences]).to(device))
            padded_predicted = [sublist + [0] * (tags.shape[1] - len(sublist)) for sublist in predicted]
            padded_predicted = torch.tensor(padded_predicted)
            tags_flat = tags.view(-1).to(device)
            predicted = padded_predicted.view(-1).to(device)
            mask = (tags_flat != 0)
            predicted = predicted[mask]
            tags_flat = tags_flat[mask]

            correct += (predicted == tags_flat).sum().item()
            total += tags_flat.size(0)

        train_accuracy_stat.append(correct / total * 100)
        if (output):
            print(f"Epoch {epoch + 1}/{epochs}, Train Accuracy: {correct / total * 100:.2f}%")

        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sentences, tags in test_loader:
                sentences = sentences.to(device)
                tags = tags.to(device)
                predicted = model.forward_crf(sentences, lengths=torch.tensor([len(s) for s in sentences]).to(device))
                padded_predicted = [sublist + [0] * (tags.shape[1] - len(sublist)) for sublist in predicted]
                padded_predicted = torch.tensor(padded_predicted)
                tags_flat = tags.view(-1).to(device)
                predicted = padded_predicted.view(-1).to(device)
                mask = (tags_flat != 0)
                predicted = predicted[mask]
                tags_flat = tags_flat[mask]

                correct += (predicted == tags_flat).sum().item()
                total += tags_flat.size(0)

        test_accuracy = correct / total * 100
        test_accuracy_stat.append(test_accuracy)
        if (output):
            print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {test_accuracy:.2f}%\n")
    if (plot):
        plt.plot(np.arange(len(train_accuracy_stat)), train_accuracy_stat, label='Training')
        plt.plot(np.arange(len(test_accuracy_stat)), test_accuracy_stat, label='Testing')
        plot_setup(xlabel='Epoch', ylabel='Accuracy, %')