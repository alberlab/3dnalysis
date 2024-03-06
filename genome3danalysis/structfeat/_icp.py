import numpy as np
from alabtools.utils import Index
import h5py

DEFAULT_DIST_CUTOFF = 500  # nm

def run(struct_id: int, hss_opt: h5py.File, params: dict) -> np.ndarray:
    """ Calculate the inter-chromosomal contact probability (ICP) of a structure.
    
    ICP is defined as the ratio of the number of inter-chromosomal contacts to the total number of contacts:
        ICP[i] = N_inter[i] / N_total[i],
        where N_inter[i] is the number of inter-chromosomal beads within a distance threshold of bead i,
        and N_total[i] is the total number of beads within the distance threshold of bead i.

    Args:
        struct_id (int): The index of the structure in the HSS file.
        hss_opt (h5py.File): The optimized HSS file, with coordinates of different structures in separate datasets.
        params (dict): A dictionary containing the parameters for the analysis.

    Returns:
        (np.ndarray): The inter-chromosomal contact probability of each bead in the structure.
    """
    
    # get coordinates of struct_id
    coord = hss_opt['coordinates'][str(struct_id)][:]
    
    # get the radii of the beads
    radii = hss_opt['radii'][:]
    
    # get the index
    index = Index(hss_opt)
    
    # get the surface-to-surface distance threshold
    try:
        dist_sts_thresh = params['dist_cutoff']
    except KeyError:
        dist_sts_thresh = DEFAULT_DIST_CUTOFF
    
    # initialize the inter-chromosomal contact ratio
    inter_ratio = np.zeros(len(index)).astype(float)
    
    # loop over the beads to calculate the inter-chromosomal contact ratio
    for i in range(len(index)):
        
        # FIND PROXIMAL BEADS
        # First, we get the center-to-center distances between the bead i and all other beads
        dists = np.linalg.norm(coord - coord[i], axis=1)
        # Then, we get the center-to-center distance trhesholds between bead i and all other beads,
        # which is the sum of the radii and the surface-to-surface distance threshold:
        #       dcap_ij = ri + rj + d_sts_thresh, fixed i
        dcap = radii[i] + radii + dist_sts_thresh
        # Finally, we get the indices of the beads that are within the distance threshold
        prox_beads = np.where(dists < dcap)[0]
        # Remove the bead i from the proximal beads (no self-interaction)
        prox_beads = prox_beads[prox_beads != i]
        
        # FILTER INTER-CHROMOSOMAL FROM PROXIMAL BEADS
        # Get the chromosome and copy of the proximal beads
        chrom_prox_beads = index.chrom[prox_beads]
        copy_prox_beads = index.copy[prox_beads]
        # Get a mask that filters only the proximal beads that are inter-chromosomal (different chromosomes or different copies)
        inter_mask = np.logical_or(chrom_prox_beads != index.chrom[i], copy_prox_beads != index.copy[i])
        # Get the proximal beads that are inter-chromosomal
        prox_inter_beads = prox_beads[inter_mask]
        
        # GET INTER-CHROMOSOMAL CONTACT RATIO
        inter_ratio[i] = len(prox_inter_beads) / len(prox_beads)
        
        del dists, dcap, prox_beads, chrom_prox_beads, copy_prox_beads, inter_mask, prox_inter_beads
    
    return inter_ratio
