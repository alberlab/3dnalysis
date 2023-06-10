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
from alabtools.utils import Genome, Index
import pandas as pd
from . import radial
from . import lamina

# Available features that can be extracted
AVAILABLE_FEATURES = ['radial', 'lamina']

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
            self.genome = None
            self.index = None
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
        self.genome = hss.genome
        self.index = hss.index
        
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
        # feat_mat is a matrix of shape (nbead_multiploid, nstruct)
        feat_mat = controller.map_reduce(parallel_task,
                                         reduce_task,
                                         args=np.arange(hss.nstruct))
        
        # Mask the gaps (or "non-domain regions") with NaNs
        try:
            # Try to read the gap BED file from the config file
            # It must be a 4-column BED file, where the 4th column is a boolean,
            # and with no header
            # (True if the region is a gap, False otherwise)
            gap_file = cfg['gap_file']
            gap_hap = pd.read_csv(gap_file, sep='\t', header=None)[3].values.astype(bool)
        except KeyError:
            # If the gap file is not specified, we assume that there are no gaps
            gap_hap = np.zeros(len(self.index.copy_index), dtype=bool)
        # Convert gap to multiploid version
        gap_mtp = np.zeros(len(self.index), dtype=bool)
        for i in self.index.copy_index:
            gap_mtp[self.index.copy_index[i]] = gap_hap[i]
        # Mask the gaps
        feat_mat[gap_mtp, :] = np.nan

        # delete the temporary directory
        os.system('rm -rf {}'.format(temp_dir))
        
        # Compute the HAPLOID bulk quantities (mean, std and log normalizations)
        feat_mean_arr, feat_std_arr = self.compute_feature_mean_std(feat_mat)
        feat_mean_arr_lnorm_gwide = self.compute_log_normalization(feat_mean_arr)
        feat_mean_arr_lnorm_cwide = self.compute_log_normalization(feat_mean_arr,
                                                                   method='chromosome-wide')
        feat_std_arr_lnorm_gwide = self.compute_log_normalization(feat_std_arr)
        feat_std_arr_lnorm_cwide = self.compute_log_normalization(feat_std_arr,
                                                                  method='chromosome-wide')
        
        # update the data of the current object
        self.data[feature] = {'ss_mat': feat_mat,
                              'mean_arr': feat_mean_arr,
                              'std_arr': feat_std_arr,
                              'mean_arr_lnorm_gwide': feat_mean_arr_lnorm_gwide,
                              'mean_arr_lnorm_cwide': feat_mean_arr_lnorm_cwide,
                              'std_arr_lnorm_gwide': feat_std_arr_lnorm_gwide,
                              'std_arr_lnorm_cwide': feat_std_arr_lnorm_cwide}
        
        if 'contact_threshold' not in cfg['features'][feature]:
            return None
        
        # Compute the single-structure contact frequency matrix
        cnt_thresh = cfg['features'][feature]['contact_threshold']
        cnt_mat = feat_mat <= cnt_thresh
        # Compute the HAPLOID average contact frequency array
        freq_arr, _ = self.compute_feature_mean_std(cnt_mat)
        # Update the data of the current object
        self.data[feature]['freq_arr'] = freq_arr
 
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
    
    def compute_feature_mean_std(self, feat_mat):
        """Compute the bulk quantities of the structural features.
        """
        
        # Initialize the arrays
        feat_mean_arr = []
        feat_std_arr = []
        
        # Loop over the haploid indices
        for i in self.index.copy_index:
            # copy_index is a Dictionary, where:
            # - keys are haploid indices (0, 1, 2, ..., nbead_hap - 1)
            # - values are the multiploid indices for the corresponding haploid index
            # for examples {0: [0, 1000], 1: [1, 1001], ...}
            feat_submat_i = feat_mat[self.index.copy_index[i], :]
            feat_mean_arr.append(np.nanmean(feat_submat_i))
            feat_std_arr.append(np.nanstd(feat_submat_i))
        
        # Cast to arrays
        feat_mean_arr = np.array(feat_mean_arr)
        feat_std_arr = np.array(feat_std_arr)
        
        return feat_mean_arr, feat_std_arr

    def compute_log_normalization(self, arr, method='genome-wide'):
        """Compute the log2 normalization of an array,
        i.e.  log2(arr / mean(arr)).
        
        It can be computed genome-wide (one global mean)
        or chromosome-wide (one mean per chromosome)

        Args:
            arr (np.array): Array to normalize.
            method (str, optional): Either 'genome-wide' or 'chromosome-wide'.
                                    Defaults to 'genome-wide'.

        Returns:
            np.array: Normalized array.
        """
        
        # Get the haploid chromstr array
        chromstr_hap = self.index.chromstr[self.index.copy == 0]
        
        # Check that the array and the chromstr_hap array have the same length
        assert len(arr) == len(chromstr_hap),\
            "Array and chromstr_hap array must have the same length."
        
        # Compute the log2 normalization genome-wide
        if method == 'genome-wide':
            return np.log2(arr / np.nanmean(arr))
        
        # Compute the log2 normalization chromosome-wide
        elif method == 'chromosome-wide':
            arr_norm = []
            # Compute the unique chroms preserving the order
            chroms_unique, chroms_index = np.unique(chromstr_hap, return_index=True)
            chroms_unique = chroms_unique[np.argsort(chroms_index)]
            # Loop over the unique chromosomes
            for chrom in chroms_unique:
                arr_chrom = arr[chromstr_hap == chrom]
                arr_norm.append(np.log2(arr_chrom / np.nanmean(arr_chrom)))
            arr_norm = np.concatenate(arr_norm)
            return arr_norm
        
        else:
            raise ValueError("Method {} not recognized.".format(method))


def structfeat_computation(feature, struct_id, hss, params):
        if feature == 'radial':
            return radial.run(struct_id, hss, params)
        if feature == 'lamina':
            return lamina.run(struct_id, hss, params)
    