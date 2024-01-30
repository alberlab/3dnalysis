import numpy as np

AVAILABLE_SHAPES = ['sphere', 'ellipsoid', 'experimental']
    
def run(struct_id, hss, params):
    
    # Read the center or set it to [0, 0, 0]
    try:
        center = params['center']
    except KeyError:
        center = [0, 0, 0]
    assert len(center) == 3, 'Center must be a 3D vector'
    
    # Read the shape
    try:
        shape = params['shape']
    except KeyError:
        raise KeyError('Shape must be specified')
    assert shape in AVAILABLE_SHAPES, 'Shape not available'
    
    # Read the radius (only for sphere and ellipsoid)
    if shape == 'sphere' or shape == 'ellipsoid':
        try:
            radius = params['radius']
        except KeyError:
            raise KeyError('Radius must be specified')
        if shape == 'sphere':
            assert isinstance(radius, (int, float)), 'Radius must be a number since shape is a sphere'
        elif shape == 'ellipsoid':
            assert len(radius) == 3, 'Radius must be a 3D vector since shape is an ellipsoid'
    
    # get coordinates of struct_id
    coord = hss.coordinates[:, struct_id, :]
    
    # compute radial distance
    # for sphere and ellipsoid
    if shape == 'sphere' or shape == 'ellipsoid':
        coord_scaled = np.array(coord) / np.array(radius)
        center_scaled = np.array(center) / np.array(radius)
        return np.linalg.norm(coord_scaled - center_scaled, axis=1)
    # for experimental
    elif shape == 'experimental':
        raise NotImplementedError('Experimental shape not implemented yet')
