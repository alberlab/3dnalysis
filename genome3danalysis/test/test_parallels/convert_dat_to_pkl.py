import pickle
import os
import sys
from alabtools.analysis import HssFile
from alabtools.geo import CenterOfMass
import numpy as np
from tqdm import tqdm

# Set working directory as file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load hss file
hss = HssFile('igm-model.damid_0.2000.inter_sigma_0.0040.intra_sigma_0.0080.sprite_5.0.hss', 'r')

# define minimum cluster size
cluster_minsize = 3

# Load pickle file
filename_in = sys.argv[1]
with open(filename_in, 'rb') as f:
    clusters = pickle.load(f)

# Create list of bodies,
# where each body is the center of mass of the clusters
bodies = []
for s in tqdm(range(hss.nstruct)):
    bodies_s = []
    for cluster in clusters['IN'][s]:  # cluster is a list of indices (int)
        if len(cluster) <= cluster_minsize:
            continue
        xyz = hss.coordinates[np.array(cluster), s, :]
        r = hss.radii[np.array(cluster)]
        com = CenterOfMass(xyz, r**3)  # center of mass of cluster
        bodies_s.append(com)
    if len(bodies_s) == 0:
        bodies.append(np.array([]))
        continue
    bodies_s = np.array(bodies_s)
    bodies.append(bodies_s)

# Save bodies as pickle file
filename_out = filename_in.replace('.dat', '.pkl')
with open(filename_out, 'wb') as f:
    pickle.dump(bodies, f)
