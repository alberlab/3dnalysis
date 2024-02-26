import numpy as np
import pandas as pd
from alabtools.utils import Index


def read_bed(filename: str, val_type: type = float) -> tuple:
    """ Read a bed file and return the chroms, starts, ends, and vals.
    
    The type of the fourth column can be specified with val_type.
    
    If the bed file does not contain a fourth column, chrom/starts/ends are returned.

    Args:
        filename (str): name (and path) of the bed file to read.
        val_type (type, optional): type of the fourth column. Defaults to float.

    Returns:
        (np.ndarray): chroms, of type str.
        (np.ndarray): starts, of type int.
        (np.ndarray): ends, of type int.
        (np.ndarray, optional): vals, of type val_type.
    """
    # Read the bed file using pandas
    # The separation key '\s+' means that the file is separated by any number of spaces
    # The header is set to None, so that the first line is not considered as the header
    bed = pd.read_csv(filename, sep='\s+', header=None)
    # Unpack chroms, starts, ends with the correct types
    chroms = bed[0].values.astype(str)
    starts = bed[1].values.astype(int)
    ends = bed[2].values.astype(int)
    # Try to unpack vals with the correct type
    try:
        vals = bed[3].values.astype(val_type)
    # If the file does not contain a fourth column, return chroms, starts, ends
    except IndexError:
        return chroms, starts, ends
    # Otherwise, return chroms, starts, ends, vals
    return chroms, starts, ends, vals


def adapt_haploid_to_index(hap_track: np.ndarray, index: Index) -> np.ndarray:
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
