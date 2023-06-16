from genome3danalysis.structfeat import SfFile
import os
from alabtools.utils import Genome, Index
from alabtools.analysis import HssFile
import numpy as np

# set working directory as file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load SF file
sf = SfFile('igm-model_mcrb_2.5MB.sf', 'r')

# Print data keys and types
print(sf.data.keys())
print(sf.data['radial'].keys())
print(type(sf.data['radial']['mean_arr']))
print(np.max(sf.data['radial']['freq_arr']))
print(type(sf.data['lamina']['mean_arr']))
print(np.min(sf.data['lamina']['mean_arr']))
print(np.max(sf.data['lamina']['mean_arr']))
print(type(sf.data['lamina_tsa']['mean_arr_lnorm_gwide']))
print(np.min(sf.data['lamina_tsa']['mean_arr_lnorm_gwide']))
print(np.max(sf.data['lamina_tsa']['mean_arr_lnorm_gwide']))

# Save radial data as bedgraph
genome = Genome('mm10', usechr=('#', 'X', 'Y'))
index = genome.bininfo(2500000).get_haploid()
index.add_custom_track('radial', sf.data['radial']['mean_arr'])
index.dump_bed(file='radial.bedgraph', header=False, include=['radial'])

# Load HSS file
hss = HssFile('igm-model_mcrb_2.5MB.hss', 'r')

def compute_radial_profile(hss, nuclear_radius):
    """
    Computes the average radial profile and its standard deviation of the input hss model.
    :return: avg_rad_haploid: average radial profile (haploid). np.array(n_beads_haploid)
             std_rad_haploid: std of the radial profile (haploid). np.array(n_beads_haploid)
    """
    # Compute the radial profile for the diploid genome in each structure
    beads = np.arange(0, len(hss.index))
    # (n_beads_diploid, n_structures)
    rad_diploid = hss.getBeadRadialPositions(beads, nuclear_radius)
    # Separate the two alleles
    # (n_beads_allele0, n_structures)
    rad_copy0 = rad_diploid[hss.index.copy == 0, :]
    # (n_beads_allele1, n_structures)
    rad_copy1 = rad_diploid[hss.index.copy == 1, :]
    # Since the sexual chromosomes only have copy 0, we extend the allele1 array with NaNs so that it has the same
    # size as allele0 array
    # (n_beads_allele0, n_structures)
    rad_copy1 = np.vstack((rad_copy1, np.nan * np.ones((rad_copy0.shape[0] - rad_copy1.shape[0], rad_copy0.shape[1]))))
    # Now we stack the two alleles horizontally (the beads of the sexual chromosomes have NaNs for allele 1
    # (n_beads_allele0, 2 * n_structures)
    rad_copies_combined = np.column_stack([rad_copy0, rad_copy1])
    # Average and Std for each bead (row) (notice that n_beads_haploid = n_beads_allele0)
    # (n_beads_haploid)
    avg_rad_haploid = np.nanmean(rad_copies_combined, axis=1)
    std_rad_haploid = np.nanstd(rad_copies_combined, axis=1)
    return avg_rad_haploid, std_rad_haploid
rad, std = compute_radial_profile(hss, (3050, 2350, 2350))

# Check that the radial avg and std are close
np.testing.assert_allclose(sf.data['radial']['mean_arr'], rad, rtol=1e-5)
np.testing.assert_allclose(sf.data['radial']['std_arr'], std, rtol=1e-5)
