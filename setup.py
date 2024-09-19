import os
from setuptools import find_packages, setup
import setuptools.command.install
import setuptools.command.develop
import subprocess

# Package meta-data.
NAME = 'nvsf'
DESCRIPTION = 'Novel View Synthesis Framework'
URL = 'https://github.com/gaurav00700/Selfsupervised-NVSF'
EMAIL = 'gauravsharma0509@gmail.com'
AUTHOR = 'Gaurav Sharma'
REQUIRES_PYTHON = '>=3.9.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # "torch@https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp39-cp39-linux_x86_64.whl",
    # "torchvision@https://download.pytorch.org/whl/cu118/torchvision-0.16.2%2Bcu118-cp39-cp39-linux_x86_64.whl",
    "torch-ema",
    "torchmetrics",
    "ninja",
    "trimesh",
    "opencv-python",
    "tensorboardX",
    "numpy==1.24",
    "pandas",
    "tqdm",
    "matplotlib==3.5.2",
    "PyMCubes",
    "rich",
    "pysdf",
    "dearpygui",
    "packaging",
    "scipy",
    "lpips",
    "imageio==2.13.0",
    "torchmetrics",
    "imageio-ffmpeg==0.4.8",
    "open3d",
    "configargparse",
    "scikit-image",
    "nksr",
    "black",
    "pyquaternion",
    "camtools==0.1.3",
    "natsort",
    "ujson",
    "pyglet==1.5.29",
    ]

# Additional URLs for packages
DEPENDENCY_LINKS = [
    # "https://download.pytorch.org/whl/cu118",
]

EXTRAS = {
    # 'feature': ['django'],
}
 
here = os.path.abspath(os.path.dirname(__file__))

with open("readme.md") as f:
    README = f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# find packages
# PACKAGES = find_packages(where='.', include=["nvsf", "nvsf.*"])
PACKAGES = find_packages()
PACKAGES_DIRS={
    'nerf': 'nvsf',
    'scripts': 'nvsf',
    'preprocess': 'nvsf',
}

def install_packages():
    # Install pip dependencies with extra index url 
    subprocess.check_call([
        'pip', 'install', 'torch==2.1.2+cu118', 'torchvision==0.16.2+cu118',
        '--extra-index-url', 'https://download.pytorch.org/whl/cu118', #'--upgrade', '--force-reinstall', '--no-cache-dir'
    ])

    # Install cuda tookit locally (use CUDA_HOME=CONDA_PREFIX)
    subprocess.check_call([
        'conda', 'install', '-c', 'nvidia/label/cuda-11.8.0', 'cuda-toolkit', '-y'
    ])

    # Install git dependencies
    subprocess.check_call([
        'pip', 'install', 'ninja', 'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch',
    ])

    # Install other requirement packages
    subprocess.check_call(['pip', 'install',] + REQUIRED)
    
    # Install cuda extensions
    subprocess.check_call([
        'pip', 'install',  
        'nvsf/nerf/raymarching', 
        'nvsf/nerf/chamfer3D'
    ])


class CustomInstallCommand(setuptools.command.install.install):
    """Standard installation"""
    def run(self):

        # Run the standard install first
        setuptools.command.install.install.run(self)
        
        install_packages()

class CustomDevelopCommand(setuptools.command.develop.develop):
    """Development installation"""
    def run(self):

        # Run the standard develop command
        setuptools.command.develop.develop.run(self)

        install_packages()


# custom install
CUSTOM_INSTALL = {
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    }

# CLI scripts
ENTRY_POINTS = {
    "console_scripts": [
        "train_nvsf = nvsf.scripts.main_nvsf:main", 
    ]
}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=PACKAGES,
    # package_dir=PACKAGES_DIRS,
    # install_requires=REQUIRED,
    # dependency_links = DEPENDENCY_LINKS,
    extras_require=EXTRAS,
    cmdclass=CUSTOM_INSTALL,
    entry_points=ENTRY_POINTS,
    include_package_data=True,
    license='None',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
