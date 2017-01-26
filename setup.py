#!/usr/bin/env python
######################################################################
# Copyright (C) 2017 Samuele Fiorini, Chiara Martini, Annalisa Barla
#
# GPL-3.0 License
######################################################################
"""cgm-tools setup script."""

from distutils.core import setup

# Package Version
from cgmtools import __version__ as version

setup(
    name='cgmtools',
    version=version,

    description=('A collection of Python tools to perform CGM (Continuous '
                 'Glucose Monitoring) analysis and forecast.'),
    long_description=open('README.md').read(),
    author='Samuele Fiorini, Chiara Martini, Annalisa Barla',
    author_email='{samuele.fiorini, chiara.martini}@dibris.unige.it, '
                 'annalisa.barla@unige.it',
    maintainer='Samuele Fiorini',
    maintainer_email='samuele.fiorini@dibris.unige.it',
    url='https://github.com/samuelefiorini/cgm-tools',
    download_url='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    license='GPL-3.0',

    packages=['cgmtools', 'cgmtools.forecast'],
    requires=['numpy (>=1.10.1)',
              'scipy (>=0.16.1)',
              'sklearn (>=0.18)',
              'keras (>=1.2.0)',
              'tensorflow (>=0.12.1)',
              'matplotlib (>=1.5.1)',
              'seaborn (>=0.7.0)'],
    scripts=[''],
)
