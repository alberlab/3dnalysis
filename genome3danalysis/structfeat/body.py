import numpy as np
from scipy.spatial.distance import cdist
import pickle
    
def run(struct_id, hss, params):
    
    # Read file name
    try:
        filename = params['filename']
    except KeyError:
        raise KeyError('Body filename must be specified')
    # Load the body file with pickle
    try:
        bodies = pickle.load(open(filename, 'rb'))
    except KeyError:
        raise KeyError('Body file not found')
    # Take the bodies of the structure
    try:
        bodies = bodies[struct_id]
    except KeyError:
        raise KeyError('Structure {} not found in body file'.format(struct_id))
    # Assert that bodies is a correct numpy array
    assert isinstance(bodies, np.ndarray), 'Bodies must be a numpy array'
    assert bodies.ndim == 2, 'Bodies must be a 2D array'
    assert bodies.shape[1] == 3, 'Bodies second dimension must be 3 (x,y,z)'
    
    # get coordinates of struct_id
    coord = hss.coordinates[:, struct_id, :]
    
    # If there are no bodies, return an array of NaNs
    if bodies.shape[0] == 0:
        return np.full(coord.shape[0], np.nan)
    
    # Compute the distance between each coordinate and each body
    dist_bodies = cdist(coord, bodies)
    
    # Return the minimum distance for each coordinate
    dist_closest_body = np.min(dist_bodies, axis=1)
    
    return dist_closest_body
    