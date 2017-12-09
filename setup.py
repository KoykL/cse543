from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
  Extension(
    'core/cards_cython',
    sources=['core/cards_cython.pyx', "core/cfunc.cpp"],
    extra_compile_args=['-std=c++11'],
    language="c++",
  ),
]

setup(
    ext_modules = cythonize(extensions, include_path = ["core"]),
)
