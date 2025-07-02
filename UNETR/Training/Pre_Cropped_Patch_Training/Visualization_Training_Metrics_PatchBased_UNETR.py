import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# List of your metric CSV files (order doesn't matter)
csv_files = [
    '/Users/mattiapanza/Downloads/metrics_f0.csv',
    '/Users/mattiapanza/Downloads/metrics_f1.csv',
    '/Users/mattiapanza/Downloads/metrics_f2.csv',
    '/Users/mattiapanza/Downloads/metrics_f3.csv',
    '/Users/mattiapanza/Downloads/metrics_f4.csv'
]

# Store each fold's metrics in a list
fold_metrics = []

# Load each file and store DataFrame
for f in csv_files:
    df = pd.read_csv(f)
    fold_metrics.append(df)

# Stack DataFrames by epoch
metrics_concat = pd.concat(fold_metrics, axis=0, ignore_index=True)

# Group by epoch to average over folds
grouped = metrics_concat.groupby('epoch')

# Compute means and stds
train_loss_mean = grouped['train_loss'].mean()
train_loss_std = grouped['train_loss'].std()

val_loss_mean = grouped['val_loss'].mean()
val_loss_std = grouped['val_loss'].std()

val_dice_mean = grouped['val_dice'].mean()
val_dice_std = grouped['val_dice'].std()

epochs = train_loss_mean.index

plt.figure(figsize=(10, 6))

# Training loss
plt.plot(epochs, train_loss_mean, label='Train Loss', color='blue')
plt.fill_between(epochs, train_loss_mean-train_loss_std, train_loss_mean+train_loss_std, color='blue', alpha=0.2)

# Validation loss
plt.plot(epochs, val_loss_mean, label='Val Loss', color='red')
plt.fill_between(epochs, val_loss_mean-val_loss_std, val_loss_mean+val_loss_std, color='red', alpha=0.2)

# Validation Dice
plt.plot(epochs, val_dice_mean, label='Val Dice', color='green')
plt.fill_between(epochs, val_dice_mean-val_dice_std, val_dice_mean+val_dice_std, color='green', alpha=0.2)

plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Metric Value', fontsize = 15)
plt.title('Training & Validation Loss and Validation Dice (Averaged over 5 folds)', fontsize = 17)
plt.xticks(fontsize = 13)
plt.yticks(fontsize=13)
plt.legend(fontsize = 12)
plt.grid(True)
plt.tight_layout()
plt.show()

