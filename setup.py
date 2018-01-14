from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras', 'h5py', 'opencv-python', 'Pillow', 'pathlib']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Faceswap trainer application'
)
