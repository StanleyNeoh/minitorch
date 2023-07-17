import os
from setuptools import setup, Extension
from Cython.Build import cythonize # type: ignore

curr_dir = os.path.dirname(os.path.abspath(__file__))

extensions = [
    Extension("hello", [os.path.join("pyx", "hello.pyx")], include_dirs=[curr_dir]),
    Extension("primes", [os.path.join("pyx", "primes.pyx")], include_dirs=[curr_dir]),
]
setup(
    ext_modules=cythonize(extensions, language_level=3),
)