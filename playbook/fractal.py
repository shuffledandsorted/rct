"""
fractal.py

This script generates a fractal image based on the characteristics of a repository, such as the number of files,
directory depth, and file sizes. The visualization reflects the structure and organization of the repository.

Nothing profound, even a little bit nonsensical. But there's something neat we're exploring here.

Future Idea:
- Consider using RCT (Recursive Contextual Tagging) to label folders based on their characteristics or contents.
  This could enhance the visualization and organization of the repository structure by providing meaningful labels
  that reflect the context and purpose of each folder.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def analyze_repository(base_path):
    num_files = 0
    max_depth = 0
    total_size = 0
    max_files_in_dir = 0

    for root, dirs, files in os.walk(base_path):
        # Gonna need to only include the files we are considering.
        # Text files.
        depth = root.count(os.sep) - base_path.count(os.sep)
        max_depth = max(max_depth, depth)
        num_files += len(files)
        for f in files:
            file_size = os.path.getsize(os.path.join(root, f))
            total_size += file_size

    return num_files, max_depth, total_size, max_files_in_dir


def generate_fractal_image(num_files, max_depth, total_size, max_files_in_dir):
    # Map repository data to fractal parameters
    width, height = 800, 800
    max_iter = max_depth
    zoom = 1 + max_depth * max_depth / 10
    x_offset = -0.5
    y_offset = 0

    # Adjust color mapping based on max_files_in_dir
    # Seems like this needs normalization?
    color_map = plt.cm.get_cmap("inferno", num_files)

    # Create a grid of complex numbers
    x = np.linspace(-2.0, 1.0, width) * zoom + x_offset
    y = np.linspace(-1.5, 1.5, height) * zoom + y_offset
    X, Y = np.meshgrid(x, y)
    # Gotta look into generating fractals.
    C = X + 1j * Y

    # Initialize the fractal image
    Z = np.zeros_like(C)
    img = np.zeros(C.shape, dtype=int)

    # Generate the Mandelbrot set
    for i in range(max_iter):
        mask = np.abs(Z) < 10
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        img += mask

    # Plot the fractal image with adjusted color mapping
    plt.imshow(img, cmap=color_map, extent=(-2, 1, -1.5, 1.5))
    plt.title("Fractal Vibes of the Repository")
    plt.axis("off")
    plt.show()


# Example usage
base_directory = ".."  # Path to your repository
num_files, max_depth, total_size, max_files_in_dir = analyze_repository(base_directory)
generate_fractal_image(num_files, max_depth, total_size, max_files_in_dir)
