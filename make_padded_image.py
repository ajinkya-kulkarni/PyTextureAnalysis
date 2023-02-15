import numpy as np

def generate_padded_image(img, chunk_size):
    # Pad the image to make it square and a multiple of chunk_size
    max_size = max(img.shape)
    padded_size = max_size + (chunk_size - max_size % chunk_size) % chunk_size

    padded_img = np.full((padded_size, padded_size), np.nan)
#     padded_img = np.zeros((padded_size, padded_size))

    padded_img[:img.shape[0], :img.shape[1]] = img

    return padded_img
