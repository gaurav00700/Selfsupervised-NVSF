__version__ = "0.1.0"

import os, sys, numpy

# entry point for kitti360 dev kit
nvsf_dir_path = os.path.abspath(os.path.join(__file__, ".." ))
sys.path.append(os.path.join(nvsf_dir_path, 'kitti360Scripts')) 
sys.path.append(os.path.join(nvsf_dir_path, 'deepen_data_suite')) 

# import packages and modules
from . import lib, nerf, preprocess, scripts

# For wildcard import of nvsf modules
__all__ = [
    "lib", 
    "nerf", 
    "preprocess", 
    "scripts",
]

#handler for np version
if numpy.__version__ > '1.20.0':
    numpy.int = numpy.int32 
    numpy.float = numpy.float64