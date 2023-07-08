import numpy as np
from scipy.spatial import distance
import alabtools.geo
    
def run(struct_id, hss, params):
    
    # Read the widow size for the gyration radius calculation
    try:
        window = params['window']
    except KeyError:
        window = 5  # should be given in Mb and converted to number of beads using the resolution
    # If the window is not an odd number, add 1 to it
    if window % 2 == 0:
        window += 1
    
    # get coordinates of struct_id
    coord = hss.coordinates[:, struct_id, :]
    
    # initialize the gyration radius array
    gyr = []
    
    # loop over the beads and calculate the gyration radius
    for i in range(hss.nbead):
        # get start and end of the window
        s = i - int((window - 1) / 2)
        e = i + int((window - 1) / 2)
        # check if the window is out of the structure
        if s < 0 or e > hss.nbead:
            gyr.append(np.nan)
            continue
        # check if the window is out of the chromosome
        if hss.index.chrom[s] != hss.index.chrom[e]:
            gyr.append(np.nan)
            continue
        # calculate the gyration radius
        gyr.append(alabtools.geo.RadiusOfGyration(coord[s:e, :], hss.radii[s:e]))
    
    return np.array(gyr)
