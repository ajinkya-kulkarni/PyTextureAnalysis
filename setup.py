import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

# Configurations
with open("README.md", "r") as fh:
	long_description = fh.read()
	setuptools.setup(
	install_requires=['scipy','numpy','opencv-python-headless', 'scikit-image', 'matplotlib', 'Pillow'],
	python_requires='>=3',
	name='PyTextureAnalysis',
	version="0.1.0",
	author="Ajinkya Kulkarni",
	author_email="kulkajinkya@gmail.com",
	description="PyTextureAnalysis is a Python package for analyzing the texture of images, specifically the local orientation, degree of coherence, and structure tensor of an image. This package is built using NumPy, SciPy and OpenCV.",
	url="https://github.com/ajinkya-kulkarni/PyTextureAnalysis",
	packages=setuptools.find_packages(),
	include_package_data=True,
	license_files=["LICENSE"],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
		"Operating System :: OS Independent",
		"Natural Language :: English",
	],
)