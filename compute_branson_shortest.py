"""
Turner, Mann, Clandinin:

https://github.com/mhturner/SC-FC
mhturner@stanford.edu
"""
from scfc import bridge, anatomical_connectivity
import os

data_dir = bridge.getUserConfiguration()['data_dir']

include_inds_branson, name_list_branson = bridge.getBransonNames()
Branson_JRC2018 = anatomical_connectivity.getAtlasConnectivity(include_inds_branson, name_list_branson, 'branson')

# Shortest path distance:
shortest_path_dist = bridge.getShortestPathStats(Branson_JRC2018)

# save
shortest_path_dist.to_pickle(os.path.join(data_dir, 'Branson_ShortestPathDistance.pkl'))
