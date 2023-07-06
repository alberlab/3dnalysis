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
            assert len(radius) == 1, 'Radius must be a scalar since shape is a sphere'
        elif shape == 'ellipsoid':
            assert len(radius) == 3, 'Radius must be a 3D vector since shape is an ellipsoid'
    
    # get coordinates of struct_id
    coord = hss.coordinates[:, struct_id, :]
    
    # compute lamin distance
    # for sphere and ellipsoid
    if shape == 'sphere' or shape == 'ellipsoid':
        # We use the formula:
        #       LamDist = √(X² + Y² + Z²) * (1 / √((X/a)² + (Y/b)² + (Z/c)²) - 1)
        # Compute X/a, Y/b, Z/c
        coord_scaled = np.array(coord) / np.array(radius)
        center_scaled = np.array(center) / np.array(radius)
        # Compute √((X/a)² + (Y/b)² + (Z/c)²)
        rad_dist_scaled = np.linalg.norm(coord_scaled - center_scaled, axis=1)
        # Compute √(X² + Y² + Z²)
        rad_dist = np.linalg.norm(coord - center, axis=1)
        # Compute LamDist
        lamin_dist = rad_dist * (1 / rad_dist_scaled - 1)
        return lamin_dist
    # for experimental
    elif shape == 'experimental':
        raise NotImplementedError('Experimental shape not implemented yet')
