
from alabtools.analysis import HssFile
from genome3danalysis.structfeat import SfFile
import os

# Set working directory as file directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

hss_name = 'igm-model_mcrb_2.5MB.hss'
# append absolute path to hss_name
hss_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), hss_name)

sf = SfFile('sf_file.sf', 'w')

config = {'hss_name': hss_name,
          'features': {'radial': {'shape': 'ellipsoid',
                                  'radius': [3050, 2350, 2350]}
                       },
          'parallel': {'controller': 'ipyparallel'}
          }

sf.run(config)
sf.save()
