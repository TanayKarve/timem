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
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    version="0.0.1",
    description="A module for monitoring memory usage of a python program",
    author="Tanay Karve",
    url="https://github.com/tanaykarve/timem",
    py_modules=["timem"],
    install_requires=["psutil"],
    python_requires=">=3.4",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    license="BSD",
)
