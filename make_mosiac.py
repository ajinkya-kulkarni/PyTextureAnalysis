import numpy as np
import matplotlib.pyplot as plt

def show_mosaic(chunks, cmap = 'viridis'):
    """
    Shows a mosaic of the original image, constructed from its chunks, as n x n subplots with reduced borders.

    Parameters:
        img (numpy.ndarray): The input 2D grayscale image.
        chunks (list): A list of chunks, each of size chunk_size x chunk_size.
        overlap_pixels (int): The overlap between chunks, in pixels.
    """

    # Calculate the number of rows and columns
    n = int(np.ceil(np.sqrt(len(chunks))))

    fig, axs = plt.subplots(n, n, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i, chunk in enumerate(chunks):
        row = i // n
        col = i % n
        axs[row, col].imshow(chunk, cmap = cmap)
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

    plt.show()
