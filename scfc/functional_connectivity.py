"""
Turner, Mann, Clandinin: functional connectivity utils and functions.

https://github.com/mhturner/SC-FC
mhturner@stanford.edu
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import pdist


def filterRegionResponse(region_response, cutoff=None, fs=None):
    """
    Low pass filter region response trace.

    :region_response: np array
    :cutoff: Hz
    :fs: Hz
    """
    if fs is not None:
        sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')
        region_response_filtered = signal.sosfilt(sos, region_response)
    else:
        region_response_filtered = region_response

    return region_response_filtered


def trimRegionResponse(file_id, region_response, start_include=100, end_include=None):
    """
    Trim artifacts from selected brain data.

    Dropouts, weird baseline shifts etc.

    :file_id: string
    :region_response: np array
        either:
            nrois x frames (region responses)
            1 x frames (binary behavior response)
    :start_include: beginning timepoint of trace
    :end_include: end timepoint of trace
    """
    # Key: brain file id
    # Val: time inds to include
    brains_to_trim = {'2018-10-19_1': np.array(list(range(100, 900)) + list(range(1100, 2000))), # transient dropout spikes
                      '2017-11-08_1': np.array(list(range(100, 1900)) + list(range(2000, 4000))), # baseline shift
                      '2018-10-20_1': np.array(list(range(100, 1000)))} # dropout halfway through

    if file_id in brains_to_trim.keys():
        include_inds = brains_to_trim[file_id]
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, include_inds]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[include_inds]
    else: # use default start / end
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, start_include:end_include]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[start_include:end_include]

    return region_response_trimmed


def getProcessedRegionResponse(resp_fp, cutoff=None, fs=None):
    """
    Filter + trim region response.

    :resp_fp: filepath to response traces
    :cutoff: highpass cutoff (Hz)
    :fs: sampling frequency (Hz)
    """
    file_id = resp_fp.split('/')[-1].replace('.pkl', '')
    region_responses = pd.read_pickle(resp_fp)

    resp = filterRegionResponse(region_responses.to_numpy(), cutoff=cutoff, fs=fs)
    resp = trimRegionResponse(file_id, resp)

    region_responses_processed = pd.DataFrame(data=resp, index=region_responses.index)
    return region_responses_processed


def computeRegionResponses(brain, region_masks):
    """
    Get region responses from brain and list of region masks.

    :brain: xyzt array
    :region_masks: list of xyz mask arrays
    """
    region_responses = []
    for r_ind, mask in enumerate(region_masks):
        region_responses.append(np.mean(brain[mask, :], axis=0))

    return np.vstack(region_responses)


def getCmat(response_filepaths, include_inds, name_list):
    """Compute fxnal corrmat from response files.

    :response_filepaths: list of filepaths where responses live as .pkl files
    :include_inds: list of ROI number IDs to select
    :name_list: associated list of ROI names
    """
    cmats_z = []
    for resp_fp in response_filepaths:
        tmp = getProcessedRegionResponse(resp_fp, cutoff=0.01, fs=1.2)
        resp_included = tmp.reindex(include_inds).to_numpy()

        correlation_matrix = np.corrcoef(resp_included)

        np.fill_diagonal(correlation_matrix, np.nan)
        # fischer z transform (arctanh) and append
        new_cmat_z = np.arctanh(correlation_matrix)
        cmats_z.append(new_cmat_z)

    # Make mean pd Dataframe
    mean_cmat = np.nanmean(np.stack(cmats_z, axis=2), axis=2)
    np.fill_diagonal(mean_cmat, np.nan)
    CorrelationMatrix = pd.DataFrame(data=mean_cmat, index=name_list, columns=name_list)

    return CorrelationMatrix, cmats_z


def getMeanBrain(filepath):
    """Return time-average brain as np array."""
    meanbrain = np.asanyarray(nib.load(filepath).dataobj).astype('uint16')
    return meanbrain


def loadAtlasData(atlas_path, include_inds, name_list):
    """
    Load region atlas data.

    :atlas_path: fp to atlas brain
    :include_inds: list of ROI number IDs to select
    :name_list: associated list of ROI names
    """
    mask_brain = np.asarray(np.squeeze(nib.load(atlas_path).get_fdata()), 'uint16')

    roi_mask = []
    for r_ind, r_name in enumerate(name_list):
        new_roi_mask = np.zeros_like(mask_brain)
        new_roi_mask = mask_brain == include_inds[r_ind] # bool
        roi_mask.append(new_roi_mask)

    return roi_mask


def getRegionGeometry(atlas_path, include_inds, name_list):
    """
    Return atlas region geometry, size etc.

    :atlas_path: fp to atlas brain
    :include_inds: list of ROI number IDs to select
    :name_list: associated list of ROI names
    """
    roi_mask = loadAtlasData(atlas_path, include_inds, name_list)

    roi_size = [x.sum() for x in roi_mask]

    coms = np.vstack([center_of_mass(x) for x in roi_mask])

    # calulcate euclidean distance matrix between roi centers of mass
    dist_mat = np.zeros((len(roi_mask), len(roi_mask)))
    dist_mat[np.triu_indices(len(roi_mask), k=1)] = pdist(coms)
    dist_mat += dist_mat.T # symmetrize to fill in below diagonal

    DistanceMatrix = pd.DataFrame(data=dist_mat, index=name_list, columns=name_list)

    # geometric mean of the sizes for each pair of ROIs
    sz_mat = np.sqrt(np.outer(np.array(roi_size), np.array(roi_size)))
    SizeMatrix = pd.DataFrame(data=sz_mat, index=name_list, columns=name_list)

    return coms, roi_size, DistanceMatrix, SizeMatrix
