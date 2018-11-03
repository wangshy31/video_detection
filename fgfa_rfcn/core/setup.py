from distutils.core import setup, Extension
import numpy as np

coviar_utils_module = Extension('coviar',
		sources = ['coviar_data_loader.c'],
		include_dirs=[np.get_include(), './ffmpeg/include/', '/usr/include/opencv2'],
		extra_compile_args=['-DNDEBUG', '-O3'],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale', '-lopencv_core', '-lopencv_highgui', '-L./ffmpeg/lib/']
)

setup ( name = 'coviar',
	version = '0.1',
	description = 'Utils for coviar training.',
	ext_modules = [ coviar_utils_module ]
)
