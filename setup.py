import re
from setuptools import setup


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix
"""


setup(
    name="timem",
    description="A module for monitoring memory usage of a python program",
    version="1.0",
    author="Tanay Karve",
    url="https://github.com/tanaykarve/timem",
    py_modules=["timem"],
    install_requires=["psutil"],
    python_requires=">=3.4",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    license="BSD",
)