import numpy as np
import alabtools.geo
from alabtools.utils import Index
from alabtools.analysis import HssFile

DEFAULT_WINDOW_SIZE = 5  # TODO: should be given in Mb and converted to number of beads using the resolution
    
def run(struct_id: int, hss: HssFile, params: dict) -> np.ndarray:
    """ Compute the radius of gyration for each bead in the structure.

    Args:
        struct_id (int): The index of the structure in the HSS file.
        hss (alabtools.analysis.HssFile)
        params (dict): A dictionary containing the parameters for the analysis.

    Returns:
        (np.ndarray): radius of gyration for each bead in the structure.
    """
    
    # Read the widow size for the gyration radius calculation
    try:
        window = params['window']
    except KeyError:
        window = DEFAULT_WINDOW_SIZE
    # If the window is not an odd number, add 1 to it
    if window % 2 == 0:
        window += 1
    assert window >= 3, 'Window size should be at least 3'
    assert isinstance(window, int), 'Window size should be an integer'
    
    # get coordinates of struct_id (only loading to memory the coordinates of the structure)
    coord = hss['coordinates'][:, struct_id, :]
    
    # get the radii of each bead
    radii = hss.radii
    
    # get the index
    index = hss.index
    index: Index
    
    # get the number of beads
    nbead = hss.nbead
    assert nbead == len(index)
    
    # initialize the gyration radius array
    gyr = []
    
    # loop over the beads and calculate the gyration radius
    for i in range(nbead):
        # get start and end of the window
        s = i - int((window - 1) / 2)
        e = i + int((window - 1) / 2)
        # check if the window is out of the structure
        if s < 0 or e > nbead - 1:
            gyr.append(np.nan)
            continue
        # check if the window is out of the chromosome
        if index.chrom[s] != index.chrom[e]:
            gyr.append(np.nan)
            continue
        # calculate the gyration radius
        gyr.append(alabtools.geo.RadiusOfGyration(coord[s:e, :], radii[s:e]))
    
    return np.array(gyr)
