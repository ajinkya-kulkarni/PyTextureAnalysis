import numpy as np

def split_into_chunks(img, chunk_size):
    """
    Splits a 2D grayscale image into chunks of a given size.

    Parameters:
        img (numpy.ndarray): The input 2D grayscale image.
        chunk_size (int): The size of the chunks to split
        the image into.
        overlap_pixels (int): The overlap between chunks,
        in pixels.

    Returns:
        list: A list of chunks, each of
        size chunk_size x chunk_size.
        numpy.ndarray: The padded image, with
        size padded_size x padded_size, where padded_size is a multiple of chunk_size.
    """
    # Divide the image into chunks
    chunks = []
    for i in range(0, img.shape[0] - chunk_size + 1, chunk_size):
        for j in range(0, img.shape[1] - chunk_size + 1, chunk_size ):
            chunk = img[i:i + chunk_size, j:j + chunk_size]
            chunks.append(chunk)

    return chunks
