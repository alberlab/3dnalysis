from genome3danalysis.structfeat import SfFile
import os
from alabtools.utils import Genome, Index

# set working directory as file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sf = SfFile('sf_file.sf', 'r')

print(sf.data['radial'].keys())

print(type(sf.data['radial']['bulk_arr']))

genome = Genome('mm10', usechr=('#', 'X', 'Y'))
index = genome.bininfo(2500000).get_haploid()
index.add_custom_track('radial', sf.data['radial']['bulk_arr'][0])
index.dump_bed(file='radial.bedgraph', header=False, include=['radial'])
