import yaml
import scfc
import os
import inspect
from scfc import bridge, anatomical_connectivity, functional_connectivity

path_to_config_file = os.path.join(inspect.getfile(bridge).split('scfc')[0], 'config.yaml')

path_to_config_file
with open(path_to_config_file, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
