"""
Step 4b: Improved LSTM Model (2 layers, more epochs)
=======================================================
Same as 04_train_lstm.py but with:
- 2-layer LSTM (instead of 1) to capture more complex patterns
- Dropout between LSTM layers to reduce overfitting
- 40 training epochs (instead of 20) to let the model converge further

How to use:
1. Make sure this file is in the same folder as the .npy files from step 3
2. Run: python 04b_train_lstm_v2.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
DATA_PREFIX = "soc_dataset"
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
NUM_EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "soc_lstm_model_v2.pt"
# ------------------------------------------------------------------


class SOCLSTM(nn.Module):
    """A 2-layer LSTM model: sequence in, single SOC value out."""

    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,  # only applies between layers, so needs num_layers > 1
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        soc_pred = self.fc(last_step_out)
        return soc_pred.squeeze(-1)


def load_data():
    X_train = np.load(f"{DATA_PREFIX}_X_train.npy")
    y_train = np.load(f"{DATA_PREFIX}_y_train.npy")
    X_test = np.load(f"{DATA_PREFIX}_X_test.npy")
    y_test = np.load(f"{DATA_PREFIX}_y_test.npy")
    return X_train, y_train, X_test, y_test


def train_model(model, train_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss (MSE): {avg_loss:.6f}")

    return model


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()

    y_true = y_test.numpy()
    predictions_pct = predictions * 100
    y_true_pct = y_true * 100

    rmse = np.sqrt(np.mean((predictions_pct - y_true_pct) ** 2))
    mae = np.mean(np.abs(predictions_pct - y_true_pct))

    print(f"\nTest RMSE: {rmse:.3f}%")
    print(f"Test MAE:  {mae:.3f}%")

    return predictions_pct, y_true_pct, rmse, mae


def plot_predictions(predictions_pct, y_true_pct, n_points=1000):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true_pct[:n_points], label="Actual SOC", color="tab:blue")
    plt.plot(predictions_pct[:n_points], label="Predicted SOC", color="tab:orange", alpha=0.7)
    plt.xlabel("Sample index (test set)")
    plt.ylabel("SOC (%)")
    plt.title("Predicted vs Actual SOC - v2, 2-layer LSTM (first {} test samples)".format(n_points))
    plt.legend()
    plt.tight_layout()
    plt.savefig("soc_prediction_plot_v2.png", dpi=150)
    print("Saved prediction plot as soc_prediction_plot_v2.png")
    plt.show()


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SOCLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    print(f"\nModel architecture:\n{model}\n")

    print("Starting training...\n")
    model = train_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE)

    predictions_pct, y_true_pct, rmse, mae = evaluate_model(model, X_test_t, y_test_t)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    plot_predictions(predictions_pct, y_true_pct)
