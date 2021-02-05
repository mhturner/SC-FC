from setuptools import setup

setup(
    name='SC-FC',
    version='1.0.0',
    description='Analysis scripts for the relationship between connectome structural connectivity and functional connectivity',
    url='https://github.com/mhturner/SC-FC',
    author='Max Turner',
    author_email='mhturner@stanford.edu',
    packages=['scfc'],
    install_requires=['numpy',
                      'scipy',
                      'nibabel',
                      'matplotlib',
                      'pandas',
                      'neuprint-python',
                      'scipy',
                      'seaborn',
                      'networkx',
                      'PyYAML',
                      'seriate',
                      'scikit-image'],
    include_package_data=True,
    zip_safe=False,
)
