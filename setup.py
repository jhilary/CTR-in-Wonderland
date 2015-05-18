#!/usr/bin/env python

from setuptools import setup

packages = {
    'ciw': 'src',
}

setup(
    name='ciw',
    version='1.0',
    author='Ilariia Belova',
    author_email='ilariyabelova@gmail.com',
    description='Web crawler',
    packages=packages,
    package_dir=packages,
    entry_points={
        'console_scripts': [
            'ciw = ciw.main:ciw'
        ]
    },
)
