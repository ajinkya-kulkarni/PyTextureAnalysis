# Texture Analysis using PyTextureAnalysis

PyTextureAnalysis is a Python package that contains tools to analyze the texture of images. This code contains functions to calculate the local orientation of fibers in an image, as well as the degree of coherence. A web application is also available for demonstrating the PyTextureAnalysis package, which allows users to analyze 2D grayscale images for texture analysis.

## Features
- Upload a 2D grayscale image for analysis
- Adjust image filter sigma, Gaussian local window, and window size for evaluating local density
- Adjust threshold value for pixel evaluation, spacing between orientation vectors, and scaling for orientation vectors
- Calculates local density, coherence, and orientation of the image
- Provides a progress bar for each stage of the analysis

## Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://textureinformation-package.streamlit.app/)

A web application developed using `Streamlit` is available at https://textureinformation-package.streamlit.app/. Check out the `Example.ipynb` file to learn how to use the package to extract and visualize local fiber orientation and organization.

## App Overview

![Streamlit App Screenshot](https://github.com/ajinkya-kulkarni/PyTextureAnalysis/blob/main/StreamlitApp.jpg)

## Requirements
- Python 3.8 or higher
- Streamlit
- OpenCV
- NumPy
- scikit-image
- Matplotlib

## Installation
1. Clone the repository.
2. Install the required packages via `pip install -r requirements.txt`.
3. Run the web application via `streamlit run PyTextureAnalysis_StreamlitApp.py`.

## Usage
1. Open the web application via `streamlit run PyTextureAnalysis_StreamlitApp.py`.
2. Upload a 2D grayscale image for analysis.
3. Adjust the various parameters using the sliders provided.
4. Click the "Analyze" button to begin the analysis.
5. View the progress of the analysis via the progress bar.
6. View the results of the analysis.

## Credits
This web application was developed, tested, and maintained by Ajinkya Kulkarni at the Max Planck Institute for Dynamics and Self-Organization, Göttingen.

## Contact
For more information or to provide feedback, please visit the project repository or contact the developer directly.