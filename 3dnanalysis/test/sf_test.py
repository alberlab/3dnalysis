
from alabtools.analysis import HssFile
import analysis3dn 

hss_name = 'igm-model_mcrb_2.5MB.hss'

sf = SfFile('sf_file.sf', 'w')

config = {'hss_name': 'hss_file.hss',
          'features': {'radial': {'shape': 'ellispoid',
                                  'radius': [3050, 2350, 2350]}
                       },
          'parallel': {'controller': 'serial'}
          }

sf.run(config)
sf.save()
