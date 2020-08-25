import networkx as nx
import numpy as np
import pandas as pd

def getNeuprintToken():
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1heHdlbGxob2x0ZXR1cm5lckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpMHJRX0M4akliX0ZrS2h2OU5DSElsWlpnRDY5YUMtVGdNLWVWM3lRP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzY2MTk1MzcwfQ.Q-57D4tX2sXMjWym2LFhHaUGHgHiUsIM_JI9xekxw_0'
    return token

    
def getRoiMapping():
    """
    Include ROIs that are at least 50% in the EM dataset
    Toss stuff in the optic lobes since those aren't in the functional dataset
    Mapping smooshes some anatomical ROI groups into single functional ROIs. E.g. MB lobes
    """
    # keys of mapping are roi names to use in analysis, based on functional data atlas
    #   values are lists of corresponding roi names in neuprint data
    mapping =  {'AL(R)':['AL(R)'], # 83% in EM volume
                'AOTU(R)':['AOTU(R)'],
                'ATL(R)': ['ATL(R)'],
                'ATL(L)': ['ATL(L)'],
                'AVLP(R)': ['AVLP(R)'],
                'BU(R)': ['BU(R)'],
                'BU(L)': ['BU(L)'], # 52% in EM volume
                'CAN(R)': ['CAN(R)'], # 68% in volume
                'CRE(R)': ['CRE(R)'],
                'CRE(L)': ['CRE(L)'], #% 90% in vol
                'EB': ['EB'],
                'EPA(R)': ['EPA(R)'],
                'FB': ['AB(R)', 'AB(L)', 'FB'],
                'GOR(R)': ['GOR(R)'],
                'GOR(L)': ['GOR(L)'], # ~60% in volume
                'IB': ['IB'], # This is lateralized in functional data but not EM. Smoosh IB_R and IB_L together in fxnal to match, see loadAtlasData
                'ICL(R)': ['ICL(R)'],
                'LAL(R)': ['LAL(R)'],
                'LH(R)': ['LH(R)'],
                'MBCA(R)': ['CA(R)'],
                'MBML(R)': ["b'L(R)", 'bL(R)', 'gL(R)'],
                'MBML(L)': ["b'L(L)", 'bL(L)', 'gL(L)'], # ~50-80% in volume
                'MBPED(R)': ['PED(R)'],
                'MBVL(R)': ["a'L(R)", 'aL(R)'],
                'NO': ['NO'],
                'PB': ['PB'],
                'PLP(R)': ['PLP(R)'],
                'PVLP(R)': ['PVLP(R)'],
                'SCL(R)': ['SCL(R)'],
                'SIP(R)': ['SIP(R)'],
                'SLP(R)': ['SLP(R)'],
                'SMP(R)': ['SMP(R)'],
                'SMP(L)': ['SMP(L)'],
                'SPS(R)': ['SPS(R)'],
                'VES(R)': ['VES(R)'], # 84% in vol
                'WED(R)': ['WED(R)']}

    return mapping


def getShortestPathStats(adjacency, alg='dijkstra'):
    """
    adjacency is pandas dataframe adjacency matrix
    """
    roi_names = adjacency.index

    graph = nx.from_numpy_matrix(adjacency.to_numpy(), create_using=nx.DiGraph)

    for e in graph.edges: #distance := 1 / edge weight
        graph.edges[e]['distance'] = 1/graph.edges[e]['weight']

    shortest_path_distance = pd.DataFrame(data=np.zeros_like(adjacency), index=roi_names, columns=roi_names)
    shortest_path_steps = pd.DataFrame(data=np.zeros_like(adjacency), index=roi_names, columns=roi_names)
    shortest_path_weight = pd.DataFrame(data=np.zeros_like(adjacency), index=roi_names, columns=roi_names)
    hub_count = pd.DataFrame(data=np.zeros(len(roi_names)), index=roi_names, columns=['count'])

    for r_ind, row in enumerate(roi_names):
        for c_ind, col in enumerate(roi_names):
            if alg == 'dijkstra':
                path = nx.algorithms.dijkstra_path(graph, source=r_ind, target=c_ind, weight='distance')
                path_length = nx.algorithms.dijkstra_path_length(graph, source=r_ind, target=c_ind, weight='distance')
            elif alg == 'bellman_ford':
                path = nx.algorithms.bellman_ford_path(graph, source=r_ind, target=c_ind, weight='distance')
                path_length = nx.algorithms.bellman_ford_path_length(graph, source=r_ind, target=c_ind, weight='distance')

            shortest_path_distance.loc[row, col] = path_length
            shortest_path_steps.loc[row, col] = len(path)

            new_path_weight = 0
            for p_ind in range(len(path)-1):
                a = path[p_ind]
                b = path[p_ind+1]
                new_path_weight += adjacency.to_numpy()[a, b]

            shortest_path_weight.loc[row, col] = new_path_weight

            if len(path) > 2:
                intermediate_nodes = path[1:-1]
                hub_count.iloc[intermediate_nodes] += 1

    return shortest_path_distance, shortest_path_steps, shortest_path_weight, hub_count
