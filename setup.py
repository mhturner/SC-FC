from setuptools import setup

setup(
    name='SC-FC',
    version='0.1.0',
    description='Analysis scripts for the relationship between connectome structural connectivity and functional connectivity',
    url='https://github.com/mhturner/SC-FC',
    author='Max Turner',
    author_email='mhturner@stanford.edu',
    packages=['scfc'],
    install_requires=['numpy',
                      'scipy',
                      'nibabel',
                      'pandas',
                      'neuprint-python',
                      'scipy',
                      'seaborn',
                      'scikit-learn',
                      'networkx',
                      'munkres',
                      'PyYAML'
                      ],
    include_package_data=True,
    zip_safe=False,
)
