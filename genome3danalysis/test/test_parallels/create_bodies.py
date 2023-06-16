import numpy as np
import pickle
from alabtools.analysis import HssFile
import os

# Set working directory as file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

hss = HssFile('igm-model_mcrb_2.5MB.hss', 'r')

# Create the fictional bodies file
bodies = []
for structID in np.arange(hss.nstruct):
    # Generate a random number of bodies
    nbody = np.random.randint(1, 20)
    bodies_struct = np.zeros((nbody, 3))
    # For each body, generate random 3D coordinates within a 4000x4000x4000nm cube centered at the origin
    for bodyID in np.arange(nbody):
        body = np.random.rand(3)*4000 - 2000
        bodies_struct[bodyID, :] = body
    bodies.append(bodies_struct)

# Save the fictional bodies file as a pickle
with open('bodies.pkl', 'wb') as f:
    pickle.dump(bodies, f)
