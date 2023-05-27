
from genome3danalysis.structfeat import SfFile
import os

# Set working directory as file directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

hss_name = 'igm-model_mcrb_2.5MB.hss'
# append absolute path to hss_name
hss_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), hss_name)

sf_name = hss_name.replace('.hss', '.sf')
sf = SfFile(sf_name, 'w')

config = {'hss_name': hss_name,
          'gap_name': '',  # ADD GAP FILE!
          'features': {'radial': {'shape': 'ellipsoid',
                                  'radius': [3050, 2350, 2350],
                                  'contact_threshold': 0.5}
                       },
          'parallel': {'controller': 'serial'}
          }

sf.run(config)
sf.save()
