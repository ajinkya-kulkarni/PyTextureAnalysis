## Overview

PyTextureAnalysis is a Python package (inspired from http://bigwww.epfl.ch/demo/orientation/) that contains tools to analyze the texture of images. This code contains functions to calculate the local orientation of fibres in an image, as well as the degree of coherence, which is a measure of how bundled and organized the fibres are.

## Demo

A webapp develped using `Streamlit`(https://streamlit.io/) is available at https://textureinformation-package.streamlit.app/ or have a look at the `Example.ipynb` file to get an idea how to use the package to extract and visualize local fibre orientation and organization.

![alt text](https://github.com/ajinkya-kulkarni/PyTextureAnalysis/blob/main/StreamlitApp.png)

## Dependencies

The following dependencies must be installed:

- numpy<1.22
- numba==0.54.0
- opencv-python-headless
- scikit-image>=0.19.3
- matplotlib>=3.6.3
- Pillow>=9.4.0
- scipy>=1.4.1
- tqdm>=4.64.1

You can install these dependencies by first installing `Python` and then running the command: `pip install -r requirements.txt` in the terminal.
