import numpy as np
import pickle
from alabtools.analysis import HssFile
import warnings
import h5py
import tempfile
import os
import sys
from functools import partial
from alabtools.parallel import Controller
from . import radial

# Available features that can be extracted
AVAILABLE_FEATURES = ['radial']

class SfFile(object):
    """Generic class for extracting and storing Structural Features from HSS file.

    Attributes:
        ...
        
    """
    
    def __init__(self, filename, mode='r'):
        
        # assert the input filename
        assert isinstance(filename, str), "The input filename must be a string."
        assert filename.endswith('.sf'), "The input filename must end with .sf."
        
        # assert the input mode
        assert mode in ['r', 'w'], "The input mode must be 'r' or 'w'."
        
        # set the filename and mode attributes
        self.filename = filename
        self.mode = mode
        
        if mode == 'r':
            self.load()
        
        if mode == 'w':
            self.nstruct = None
            self.nbead = None
            self.features = None
            self.hss_filename = None
            self.fitted = False
            self.available_features = AVAILABLE_FEATURES
            self.data = {}
    
    def load(self):
        """Loads a SfFile from a pickle file.
        """
        
        with open(self.filename, 'rb') as f:
            loaded_sf = pickle.load(f)
                 
        assert hasattr(loaded_sf, 'nstruct'), "Loaded SfFile has no attribute 'nstruct'."
        assert hasattr(loaded_sf, 'nbead'), "Loaded SfFile has no attribute 'nbead'."
        assert hasattr(loaded_sf, 'features'), "Loaded SfFile has no attribute 'features'."
        assert hasattr(loaded_sf, 'hss_filename'), "Loaded SfFile has no attribute 'hss_filename'."
        
        if loaded_sf.fitted == False:
            warnings.warn('Loaded SfFile has not been fitted.')
        
        # update the attributes of the current object
        # (every object has a __dict__ attribute, which is a dictionary of its attributes.
        # In this way, we can update the attributes of the current object with the attributes
        # of the loaded object)
        self.__dict__.update(loaded_sf.__dict__)
        
        # Delete the loaded object
        del loaded_sf
    
    def save(self):
        """Saves a SfFile to a pickle file.
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)
    
    def run(self, cfg):
        """Compute the Structural Features specified in the config file.

        Args:
            cfg (dict): Configuration dictionary.
        """
        
        # Read features
        try:
            features = cfg['features']
        except KeyError:
            raise KeyError("No features found in the config file.")
        assert isinstance(features, dict), "Features must be a dict."
        self.features = list(features.keys())
        
        # Try to get hss_name from cfg
        try:
            hss_name = cfg['hss_name']
        except KeyError:
            "hss_name not found in cfg."
        hss = HssFile(hss_name, 'r')
        self.nstruct = hss.nstruct
        self.nbead = hss.nbead
        self.hss_filename = hss_name
        
        for feature in features:
            assert isinstance(feature, str), "Each feature must be a string."
            assert feature in self.available_features, "Feature {} is not available.".format(feature)

            self.run_feature(cfg, feature)
        
        self.fitted = True
    
    def run_feature(self, cfg, feature):
        """Extract a particular structural feature from an HSS file specified in the config file.
        
        Args:
            cfg: Configuration dictionary.
            feature: Feature name.
        """

        # open HSS file
        hss_name = cfg['hss_name']
        hss = HssFile(hss_name, 'r')
        
        # create a temporary directory to store nodes' results
        temp_dir = tempfile.mkdtemp(dir=os.getcwd())
        sys.stdout.write("Temporary directory for nodes' results: {}\n".format(temp_dir))
        
        # create a Controller
        controller = Controller(cfg)
        
        # set the parallel and reduce tasks
        parallel_task = partial(self.parallel_feature,
                                feature=feature,
                                cfg=cfg,
                                temp_dir=temp_dir)
        reduce_task = partial(self.reduce_feature,
                              cfg=cfg,
                              temp_dir=temp_dir,
                              feature=feature)

        # run the parallel and reduce tasks
        feat_mat = controller.map_reduce(parallel_task,
                                         reduce_task,
                                         args=np.arange(hss.nstruct))

        # delete the temporary directory
        os.system('rm -rf {}'.format(temp_dir))
        
        # update the data of the current object
        self.data[feature] = {'ss_mat': feat_mat,
                              'bulk_arr': compute_bulk_quantities(feat_mat,
                                                                  hss.index.copy_index)}
        
        return None
 
    @staticmethod
    def parallel_feature(struct_id, feature, cfg, temp_dir):
        
        # get data from the configuration
        try:
            hss_name = cfg['hss_name']
        except KeyError:
            raise KeyError("hss_name not found in cfg.")
        
        # open hss file
        hss = HssFile(hss_name, 'r')
        
        # get the feature parameters from the configuration
        try:
            params = cfg['features'][feature]
        except KeyError:
            raise KeyError("No parameters found for feature {}.".format(feature))

        # compute the feature array for the current structure
        feat_arr = structfeat_computation(feature, struct_id, hss, params)
        
        # save the feature array in the temporary directory
        out_name = os.path.join(temp_dir, feature + '_' + str(struct_id) + '.npy')
        np.save(out_name, feat_arr)
        
        return out_name
    
    @staticmethod
    def reduce_feature(out_names, cfg, temp_dir, feature):
        
        # get hss_name from cfg
        try:
            hss_name = cfg['hss_name']
        except KeyError:
            "hss_name not found in cfg."
        
        # open ct file
        hss = HssFile(hss_name, 'r')
        
        # check that the output size is correct
        assert len(out_names) == hss.nstruct,\
            "Number of output files does not match number of structures."
        
        # initialize the structure matrix and bulk arrays
        feat_mat = np.zeros((hss.nbead, hss.nstruct))
        
        # Loop over the structures
        for structID in np.arange(hss.nstruct):
            # try to open the output file associated to structID
            try:
                out_name = os.path.join(temp_dir, feature + '_' + str(structID) + '.npy')
                feat_mat[:, structID] = np.load(out_name)
            except IOError:
                raise IOError("File {} not found.".format(out_name))
        
        return feat_mat
    
def compute_bulk_quantities(feat_mat, copy_index):
    """Compute the bulk quantities of the structural features.
    """
    feat_mean_arr = []
    feat_std_arr = []
    for i in copy_index:
        feat_submat_i = feat_mat[copy_index[i], :]
        feat_mean_arr.append(np.mean(feat_submat_i))
        feat_std_arr.append(np.std(feat_submat_i))
    return feat_mean_arr, feat_std_arr


def structfeat_computation(feature, struct_id, hss, params):
        if feature == 'radial':
            return radial.run(struct_id, hss, params)
    