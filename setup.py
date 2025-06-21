from setuptools import setup, find_packages

setup(
    name='evdw_nc',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'mdtraj',
        'cupy-cuda12x',
    ],
    entry_points={
        'console_scripts': [
            'evdw_nc=evdw_nc.cli:main',
        ],
    },
    author='POMALAB',
    description='GPU-accelerated native contact calculation using enlarged VDW radii',
    include_package_data=True,
)
