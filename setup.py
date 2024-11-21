from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='utils',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='utils.ext',
            sources=[
                'utils/extensions/extra/cloud/cloud.cpp',
                'utils/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'utils/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'utils/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'utils/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'utils/extensions/cpu/radius_filter/radius_filter.cpp',
                'utils/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
