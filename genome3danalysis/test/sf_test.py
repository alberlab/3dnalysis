
from alabtools.analysis import HssFile
from genome3danalysis.structfeat import SfFile

hss_name = 'igm-model_mcrb_2.5MB.hss'

sf = SfFile('sf_file.sf', 'w')

config = {'hss_name': hss_name,
          'features': {'radial': {'shape': 'ellipsoid',
                                  'radius': [3050, 2350, 2350]}
                       },
          'parallel': {'controller': 'serial'}
          }

sf.run(config)
sf.save()
