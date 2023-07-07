import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def run(struct_id, hss, params):
    
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
        dist_sts_thresh = 500  # nm
    # Get bead radii sum matrix (dcap_ij = ri + rj)
    dcap = cdist(hss.radii[:, None], -hss.radii[:, None])
    # Get distance threshold matrix
    dist_thresh = dcap + dist_sts_thresh
    # Get proximity matrix
    prox = cdist(coord, coord, 'euclidean') < dist_thresh
    # Set proximity matrix diagonal to False (bead is not in proximity with itself)
    np.fill_diagonal(prox, False)
    
    # Combine inter and prox matrices
    inter_prox = np.logical_and(inter, prox)

    # Get Inter-chromosomal contact ratio
    inter_ratio = np.sum(inter_prox, axis=1) / np.sum(prox, axis=1)
    
    return inter_ratio
