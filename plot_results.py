import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

def plot_signal(csv_file):
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found. Run the C++ program first.")
        return

    # 1. Read the CSV Data
    print(f"Reading {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Setup the Plot
    plt.figure(figsize=(12, 6))

    # --- Subplot 1: Signal Comparison ---
    plt.subplot(1, 2, 1)
    plt.title(f"Signal Denoising (First {len(df)} points)")
    
    # FIXED: Convert pandas Series to numpy arrays using .to_numpy()
    # This prevents the "Multi-dimensional indexing" ValueError
    index_np = df['Index'].to_numpy()
    noisy_np = df['NoisyInput'].to_numpy()
    smoothed_np = df['SmoothedOutput'].to_numpy()

    # Plot Noisy Input (Light Blue)
    plt.plot(index_np, noisy_np, 
             label='Noisy Input', color='lightgray', linestyle='-', alpha=0.7)
    
    # Plot Smoothed Output (Red)
    plt.plot(index_np, smoothed_np, 
             label='Smoothed Output', color='red', linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Subplot 2: Noise Removed ---
    # Visualize the difference (approximates the noise extracted)
    plt.subplot(1, 2, 2)
    plt.title("Estimated Noise (Input - Output)")
    
    noise_extracted = noisy_np - smoothed_np
    plt.plot(index_np, noise_extracted, color='blue', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude Difference')
    plt.grid(True, alpha=0.3)

    # 3. Save the Plot
    output_image = csv_file.replace('.csv', '.png')
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    print(f"Plot saved to: {output_image}")

if __name__ == "__main__":
    # Default to output.csv if no argument provided
    file_path = sys.argv[1] if len(sys.argv) > 1 else "output.csv"
    plot_signal(file_path)