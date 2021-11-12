# !/usr/bin/env python3

"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

from setuptools import setup
import os

version = {}
with open(os.path.join('directsearch', 'version.py')) as fp:
    exec(fp.read(), version)
__version__ = version['__version__']

setup(
    name='directsearch',
    version=__version__,
    description='A derivative-free solver for unconstrained minimization',
    long_description=open('README.rst').read(),
    author='Lindon Roberts',
    author_email='lindon.roberts@anu.edu.au',
    url='https://github.com/lindonroberts/directsearch',
    download_url='https://github.com/lindonroberts/directsearch/archive/v%s.tar.gz' % version,
    packages=['directsearch'],
    license='GNU GPL',
    keywords='mathematics derivative free optimization direct search',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Framework :: IPython',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    install_requires=['numpy >= 1.11', 'scipy >= 1.0'],
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=True,
)
