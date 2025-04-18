import matplotlib.pyplot as plt
import numpy as np  # Added for smoothing

# 把loss.txt里的数值画出来，每行是一个loss

def smooth_losses(losses, smoothing_factor=0.9):
    smoothed = []
    last = losses[0]
    for loss in losses:
        smoothed_value = last * smoothing_factor + loss * (1 - smoothing_factor)
        smoothed.append(smoothed_value)
        last = smoothed_value
    return smoothed

def plot_losses(file_path):
    with open(file_path, 'r') as f:
        losses = [float(line.strip()) for line in f if line.strip()]
    smoothed_losses = smooth_losses(losses)  # Smooth the losses
    plt.plot(smoothed_losses, marker='o', label='Smoothed Loss')  # Plot smoothed losses
    plt.plot(losses, alpha=0.5, label='Original Loss')  # Plot original losses with transparency
    plt.title('Loss Values')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("loss.png")

if __name__ == "__main__":
    plot_losses('loss.txt')
