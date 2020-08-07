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
