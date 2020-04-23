from setuptools import setup

setup(
    name='SC-FC',
    version='0.0.1',
    description='Analysis scripts for the relationship between connectome structural connectivity and functional connectivity',
    url='https://github.com/mhturner/SC-FC',
    author='Max Turner',
    author_email='mhturner@stanford.edu',
    packages=['region_connectivity'],
    install_requires=['numpy',
        'scipy',
        'nibabel',
        'pandas',
        'neuprint-python',
        'scipy'],
    include_package_data=True,
    zip_safe=False,
)
