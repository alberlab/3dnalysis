
from genome3danalysis.structfeat import SfFile
import os

# Set working directory as file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

hss_name = 'igm-model.damid_0.2000.inter_sigma_0.0040.intra_sigma_0.0080.sprite_5.0.hss'
hss_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), hss_name)

sf_name = hss_name.replace('.hss', '.sf')
sf = SfFile(sf_name, 'w')

config = {'hss_name': hss_name,
          'gap_name': 'ADD GAP FILE!',
          'features': {'radial': {'shape': 'sphere',
                                  'radius': 5000.,
                                  'contact_threshold': 0.5},
                       'lamina': {'shape': 'sphere',
                                  'radius': 5000.,
                                  'contact_threshold': 0.2 * 5000.},
                       'lamina_tsa': {'shape': 'sphere',
                                      'radius': 5000.,
                                      'exterior_threshold': 0.85,
                                      'tsa_exponent': 0.004},
                       'speckle': {'filename': 'ADD SPECKLE FILE!'},
                       'speckle_tsa': {'filename': 'ADD SPECKLE FILE!',
                                       'tsa_exponent': 0.004},
                       'nucleoli': {'filename': 'ADD NUCLEOLI FILE!'},
                       'nucleoli_tsa': {'filename': 'ADD NUCLEOLI FILE!',
                                        'tsa_exponent': 0.004},
                       'transAB': {'filename': 'ADD AB FILE!',
                                   'dist_cutoff': 500},
                       'icp': {'dist_cutoff': 500},
                       'rg': {'window': 5}
                       },
          'parallel': {'controller': 'ipyparallel'}
          }

sf.run(config)
sf.save()
