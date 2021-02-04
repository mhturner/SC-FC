# SC-FC
Analysis and figure generation code for Turner, Mann, Clandinin: The connectome predicts resting state functional connectivity across the Drosophila brain

------------
To install, cd to the top level of the repository, where setup.py lives and:

pip install -e .

-or-

pip3 install -e .


Before you can run the Python code, you need to define a config.yaml file. See config_example.yaml and enter the path to the (functional) data directory, the path to where saved analysis should go, and your neuprint token to access the hemibrain dataset.

------------
To run the R code, you should install the natverse (http://natverse.org/) and define the following environment variables in your .Renviron file:

-GITHUB_PAT

-neuprint_server (e.g. "https://neuprint.janelia.org")

-neuprint_token

-neuprint_dataset (e.g. "hemibrain:v1.2")
