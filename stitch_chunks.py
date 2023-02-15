import numpy as np

def stitch_back_chunks(analyzed_chunk_list, padded_img, img, chunk_size):
    # Calculate the number of chunks in each dimension
    num_chunks = padded_img.shape[0] // chunk_size

    # Initialize a new NumPy array for the reconstructed image

    reconstructed_img = np.full(padded_img.shape, np.inf)
#     reconstructed_img = np.zeros((padded_img.shape))

    # Iterate over each chunk and copy it back to the correct location in the reconstructed image
    for i in range(len(analyzed_chunk_list)):
        row = i // num_chunks
        col = i % num_chunks

        chunk = analyzed_chunk_list[i]
        start_row = row * chunk_size
        end_row = start_row + chunk_size

        start_col = col * chunk_size
        end_col = start_col + chunk_size
        reconstructed_img[start_row:end_row, start_col:end_col] = chunk

    # Crop the reconstructed image to the size of the original input image
    reconstructed_img = reconstructed_img[:img.shape[0], :img.shape[1]]

    return reconstructed_img
