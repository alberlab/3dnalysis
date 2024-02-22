import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


DEFAULT_DIST_CUTOFF = 500  # nm
DEFAULT_METHOD = 'memory_efficient'


def run(struct_id, hss, params):
    
    # Read file name
    try:
        filename = params['filename']
    except KeyError:
        raise KeyError('AB-compartment filename must be specified')
    # Load the A/B compartment file with pandas
    try:
        ab = pd.read_csv(filename, sep='\s+', header=None)[3].values.astype(str)
    except KeyError:
        raise KeyError('AB-compartment file not found')
    # Assert that AB track is correct
    assert isinstance(ab, np.ndarray), 'AB-compartment track must be a numpy array'
    assert len(ab) == np.sum(hss.index.copy == 0),\
        'AB-compartment track must have the same length as the number of haploid beads in the structure'
    assert len(np.unique(ab)) == 2 or len(np.unique(ab)) == 3,\
        'AB-compartment track must contain only A and B and optionally NA (unspecified format)'
    assert 'A' in np.unique(ab) and 'B' in np.unique(ab), 'AB-compartment track must contain A and B'
    
    # Choose method to run
    try:
        method = params['method']
    except KeyError:
        method = DEFAULT_METHOD
    
    # Run the transAB analysis with the chosen method
    if method == 'time_efficient':
        return run_time_efficient(struct_id, hss, ab, params)
    elif method == 'memory_efficient':
        return run_memory_efficient(struct_id, hss, ab, params)


def run_time_efficient(struct_id, hss, ab, params):
    """ Computations are done in a time-efficient way, but memory consumption is higher.
    
    Calculations are done using numpy matrices, avoiding loops,
    which might be memory consuming depending on the number of structures and resolution."""
    
    # get coordinates of struct_id
    coord = hss.coordinates[:, struct_id, :]
    
    # Find the inter-chromosomal matrix (inter_ij = True if i and j are on different chromosomal alleles)
    inter_chrom = hss.index.chrom[:, None] != hss.index.chrom[None, :]
    inter_copy = hss.index.copy[:, None] != hss.index.copy[None, :]
    inter = np.logical_or(inter_chrom, inter_copy)
    
    # Find the proximity matrix (prox_ij = True if i and j are within a certain distance)
    try:  # get surface-to-surface distance threshold
        dist_sts_thresh = params['dist_cutoff']
    except KeyError:
        dist_sts_thresh = DEFAULT_DIST_CUTOFF
    # Get bead radii sum matrix (dcap_ij = ri + rj)
    dcap = cdist(hss.radii[:, None], -hss.radii[:, None])
    # Get distance threshold matrix
    dist_thresh = dcap + dist_sts_thresh
    # Get proximity matrix
    prox = cdist(coord, coord, 'euclidean') < dist_thresh
    
    # Combine inter and prox matrices
    inter_prox = np.logical_and(inter, prox)
    
    # Get the Trans AB matrices
    ab = adapt_to_index(ab, hss.index)  # adapt AB track to multi-ploid index
    ab_vstack = np.vstack([ab] * len(ab))  # convert AB track to a vertical stack (attention: not symmetric!)
    transA = np.logical_and(inter_prox, ab_vstack == 'A')
    transB = np.logical_and(inter_prox, ab_vstack == 'B')
    
    # Get TransAB ratio
    transAB_ratio = np.sum(transA, axis=1) / (np.sum(transA, axis=1) + np.sum(transB, axis=1))
    
    return transAB_ratio


def run_memory_efficient(struct_id, hss, ab, params):
    """ Computations are done in a memory-efficient way, but time consumption is higher.
    
    Instead of using numpy matrices, calculations are done using loops, with one bead at a time,
    and the most complex structure is a numpy array. """
    
    # get coordinates of struct_id
    coord = hss.coordinates[:, struct_id, :]
    
    # Initialize the transAB_ratio
    transAB_ratio = np.zeros(len(hss.index))
    
    # Loop over all beads
    for i in range(len(hss.index)):
        
        # Find the inter-chromosomal array (inter_ij = True if i and j are on different chromosomal alleles. fixed i)
        inter_chrom = hss.index.chrom[i] != hss.index.chrom
        inter_copy = hss.index.copy[i] != hss.index.copy
        inter = np.logical_or(inter_chrom, inter_copy)
        
        # Find the proximity matrix (prox_ij = True if i and j are within a certain distance)
        try:  # get surface-to-surface distance threshold
            dist_sts_thresh = params['dist_cutoff']
        except KeyError:
            dist_sts_thresh = DEFAULT_DIST_CUTOFF
        # Get bead radii sum array (dcap_ij = ri + rj, fixed i)
        dcap = hss.radii[i] + hss.radii
        # Get distance threshold array
        dist_thresh = dcap + dist_sts_thresh
        # Get proximity array
        prox = np.linalg.norm(coord - coord[i], axis=1) < dist_thresh
        
        # Combine inter and prox arrays
        inter_prox = np.logical_and(inter, prox)
        
        # Get the Trans AB arrays
        ab = adapt_to_index(ab, hss.index)
        transA = np.logical_and(inter_prox, ab == 'A')
        transB = np.logical_and(inter_prox, ab == 'B')
        
        # Get TransAB ratio
        transAB_ratio[i] = np.sum(transA) / (np.sum(transA) + np.sum(transB))
        
        del inter_chrom, inter_copy, inter, dcap, dist_thresh, prox, inter_prox, transA, transB
    
    transAB_ratio


def adapt_to_index(hap_track, index):
    """Given a haploid track, adapts it to the multi-ploid index.

    Args:
        hap_track (np.ndarray(nbead_hap)): haplotype track
        index (alabtools.utils.Index): index object
    Return:
        np.ndarray(nbeads): adapted track
    """
    multi_track = np.zeros(len(index), dtype=hap_track.dtype)
    for i in index.copy_index:
        # copy_index is a Dictionary, where:
        # - keys are haploid indices (0, 1, 2, ..., nbead_hap - 1)
        # - values are the multiploid indices for the corresponding haploid index
        # for examples {0: [0, 1000], 1: [1, 1001], ...}
        multi_track[index.copy_index[i]] = hap_track[i]
    return multi_track
