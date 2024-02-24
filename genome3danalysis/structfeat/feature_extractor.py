import numpy as np
import h5py
from alabtools.analysis import HssFile
from alabtools.utils import Index
import tempfile
import os
import sys
from functools import partial
from alabtools.parallel import Controller
import pandas as pd
from . import _radial
from . import _lamina
from . import _lamina_tsa
from . import _body
from . import _transAB
from . import _icp
from . import _rg

# Available features that can be extracted
AVAILABLE_FEATURES = ['radial', 'lamina', 'lamina_tsa',
                      'speckle', 'nucleoli', 'speckle_tsa', 'nucleoli_tsa', 
                      'transAB', 'icp', 'rg']

class SfFile(object):
    """Generic class for extracting and storing Structural Features from HSS file.

    Attributes:
        ...
        
    """
    
    def __init__(self, h5_name: str, mode: str = 'r') -> None:
        """ Initialize a SfFile object.
        
        The data of the object is encapsulated in a HDF5 file,
        which can be opened in multiple modes.

        Args:
            h5_name (str): Path to the HDF5 file.
            mode (str, optional): File access mode. Defaults to 'r'.
        """
        
        # Extend the name with its absolute path
        h5_name = os.path.abspath(h5_name)
        
        # Check that h5_name has a valid path
        if not os.path.exists(os.path.dirname(h5_name)):
            raise FileNotFoundError("The path of the HDF5 file does not exist.")
        
        # Check that mode is valid
        if not mode in ['r', 'r+', 'w', 'w-', 'x', 'a']:
            raise ValueError("mode must be one of 'r', 'r+', 'w', 'w-', 'x', 'a'.")
        
        # If the file doesn't exists, make sure that mode is write (w, w-, x, r+, a)
        if not os.path.exists(h5_name) and mode not in ['w', 'w-', 'x', 'r+', 'a']:
            raise FileNotFoundError("The HDF5 file does not exist. Use mode 'w', 'w-', 'x', 'r+', 'a' to create it.")
        
        # Open the HDF5 file
        self.h5_name = h5_name
        self.h5 = h5py.File(h5_name, mode)
    
    
    # CONTAIN METHOD
    def __contains__(self, name: str) -> bool:
        """ Check if a dataset exists in the h5 file."""
        return name in self.h5
    
    
    # DELETE METHOD
    def __delitem__(self, name: str) -> None:
        """ Delete a dataset from the h5 file."""
        del self.h5[name]
    
    
    # SETTER FUNCTIONS
    
    def set_index(self, index: Index) -> None:
        """ Set the Index object in the h5 file."""
        index.save(self.h5)
    
    def set_feature(self, feature_name: str, matrix: np.ndarray) -> None:
        """ Set a feature, along with its matrix, in the h5 file.
        
        The feature creates a group at the root level of the h5 file,
        with the matrix as a dataset.
        
        Other datasets can be added to the group, e.g. mean, std,
        but this is not mandatory, and thus is not implemented here.

        Args:
            feature_name (str)
            matrix (np.ndarray): 2D feature matrix of shape (nstruct, nbead).
        """
        
        # Check that the feature is not already in the h5 file
        if feature_name in self:
            raise ValueError("The feature '{}' already exists in the h5 file.".format(feature_name))

        # Create a group for the feature (at the root level)
        h5_group = self.h5.create_group(f'/{feature_name}')
        
        # Add the feature matrix to the group
        h5_group.create_dataset('matrix', data=matrix, dtype=matrix.dtype)
    
    
    # GETTER FUNCTIONS
    
    def get_index(self) -> Index:
        """ Get the Index object from the h5 file."""
        return Index(self.h5)
    
    def get_feature(self, feature_name: str, format: str = 'matrix') -> np.ndarray:
        """ Get the feature data from the h5 file.
        The 'format' key specifies what to retrieve, e.g.
        - 'matrix': the feature matrix, 2D array of shape (nstruct, nbead)
        - 'mean': the mean array, 1D array of shape (nbead,)
        - ...
        The available formats depend on the feature."""
        
        # Check that the feature exists in the h5 file
        if feature_name not in self.h5:
            raise ValueError("Feature {} not found in the h5 file.".format(feature_name))
        # Check that the format is valid for the feature
        if format not in self.h5[feature_name]:
            raise ValueError("Format {} is not valid for feature {}.".format(format, feature_name))
        
        return self.h5[feature_name][format][:]
    
    def get_feature_list(self) -> list:
        """ Get the list of feature matrices in the h5 file."""
        # Get the list of keys in the h5 file
        h5_keys = list(self.h5.keys())
        # Remove the keys that are not feature matrices
        remove_keys = ['index', 'genome']
        for key in remove_keys:
            if key in h5_keys:
                h5_keys.remove(key)
        return h5_keys
    
    def get_feature_dataset_list(self, feature_name: str) -> list:
        """ Get the list of datasets in the feature group."""
        return list(self.h5[feature_name].keys())
    
    
    # DEFINE PROPERTIES (READ ONLY)
    index = property(get_index, doc="Index object.")
    feature_list = property(get_feature_list, doc="List of feature matrices.")
    
    
    # CLOSE METHOD
    def close(self) -> None:
        """ Close the h5 file."""
        self.h5.close()
    
    
    # STRUCTURAL FEATURES EXTRACTION METHODS
    
    def run(self, cfg: dict) -> None:
        """Compute the Structural Features specified in the config file.

        Args:
            cfg (dict): Configuration dictionary.
        """
        
        # Save the config file in the current object
        self.config = cfg
        
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
        self.set_index(hss.index)
        
        for feature in features:
            assert isinstance(feature, str), "Each feature must be a string."
            assert feature in AVAILABLE_FEATURES, "Feature {} is not available.".format(feature)

            self.run_feature(cfg, feature)
        
        self.fitted = True
    
    def run_feature(self, cfg, feature):
        """Extract a particular structural feature from an HSS file specified in the config file.
        
        Args:
            cfg: Configuration dictionary.
            feature: Feature name.
        """
        
        sys.stdout.write("Extracting feature: {}\n".format(feature))

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
        sys.stdout.write("Running parallel and reduce tasks...\n")
        feat_mat = controller.map_reduce(parallel_task,
                                         reduce_task,
                                         args=np.arange(hss.nstruct))
        sys.stdout.write("Parallel and reduce tasks completed.\n")
        
        # Mask the gaps (or "non-domain regions") with NaNs
        try:
            # Try to read the gap BED file from the config file
            # It must be a 4-column BED file, where the 4th column is a boolean,
            # and with no header
            # (True if the region is a gap, False otherwise)
            gap_file = os.path.abspath(cfg['gap_file'])
            gap_hap = pd.read_csv(gap_file, sep='\s+', header=None)[3].values.astype(bool)
        except KeyError:
            # If the gap file is not specified, we assume that there are no gaps
            gap_hap = np.zeros(len(self.index.copy_index), dtype=bool)
        # Convert gap to multiploid version
        gap_mtp = np.zeros(len(self.index), dtype=bool)
        for i in self.index.copy_index:
            gap_mtp[self.index.copy_index[i]] = gap_hap[i]
        # Mask the gaps
        feat_mat[gap_mtp, :] = np.nan
        sys.stdout.write("Gaps masked.\n")

        # delete the temporary directory
        os.system('rm -rf {}'.format(temp_dir))
        
        # Set the feature matrix in the h5 file
        self.set_feature(feature, feat_mat)
        sys.stdout.write("Feature matrix set in the h5 file.\n")
        
        # Compute the HAPLOID bulk quantities (mean, std and log normalizations)
        feat_mean_arr, feat_std_arr = self.compute_feature_mean_std(feature)
        feat_mean_arr_lnorm_gwide = self.compute_log_normalization(feat_mean_arr, self.index, method='genome-wide')
        feat_mean_arr_lnorm_cwide = self.compute_log_normalization(feat_mean_arr, self.index, method='chromosome-wide')
        feat_std_arr_lnorm_gwide = self.compute_log_normalization(feat_std_arr, self.index, method='genome-wide')
        feat_std_arr_lnorm_cwide = self.compute_log_normalization(feat_std_arr, self.index, method='chromosome-wide')
        sys.stdout.write("Bulk quantities computed.\n")
        
        # Add the bulk quantities to feature group of the h5 file
        h5_group = self.h5[feature]
        h5_group.create_dataset('mean_arr', data=feat_mean_arr)
        h5_group.create_dataset('std_arr', data=feat_std_arr)
        h5_group.create_dataset('mean_arr_lnorm_gwide', data=feat_mean_arr_lnorm_gwide)
        h5_group.create_dataset('mean_arr_lnorm_cwide', data=feat_mean_arr_lnorm_cwide)
        h5_group.create_dataset('std_arr_lnorm_gwide', data=feat_std_arr_lnorm_gwide)
        h5_group.create_dataset('std_arr_lnorm_cwide', data=feat_std_arr_lnorm_cwide)
        sys.stdout.write("Bulk quantities added to feature group.\n")
        
        if 'contact_threshold' not in cfg['features'][feature]:
            return None
        
        # Compute the single-structure contact frequency matrix
        cnt_thresh = cfg['features'][feature]['contact_threshold']
        # Compute the HAPLOID average contact frequency array
        freq_arr, _ = self.compute_feature_mean_std(feature, threshold=cnt_thresh)
        # Add the association frequency array to feature group of the h5 file
        h5_group.create_dataset('association_freq', data=freq_arr)
 
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
        
        # Save an empty file to indicate that the computation is done
        out_name = os.path.join(temp_dir, feature + '_' + str(struct_id) + '.done')
        with open(out_name, 'w') as f:
            pass
        
        return out_name
    
    @staticmethod
    def reduce_feature(out_names, cfg, temp_dir, feature):
        
        sys.stdout.write("Reducing feature started.\n")
        
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
        sys.stdout.write("Feature matrix initialized in reduce_feature.\n")
        
        # Loop over the structures
        for structID in np.arange(hss.nstruct):
            # try to open the output file associated to structID
            try:
                out_name = os.path.join(temp_dir, feature + '_' + str(structID) + '.npy')
                feat_mat[:, structID] = np.load(out_name)
            except IOError:
                raise IOError("File {} not found.".format(out_name))
        
        return feat_mat
    
    
    # CALCULATIONS FROM FEATURE MATRICES METHODS
    
    def compute_feature_mean_std(self, feature_name: str, threshold: float = None) -> tuple:
        """ Compute the mean and standard deviation of a feature.
        
        If a threshold is specified, the feature is thresholded before computing the mean and std,
        thus computing the mean and std of the association frequency.

        Args:
            feature_name (str)
            threshold (float, optional): Threshold value for Association Frequency.
                                         If None, the normal distances are computed.

        Returns:
            (np.array): Mean array of the feature of shape (nbead_hapoloid,)
            (np.array): Standard deviation array of the feature of shape (nbead_haploid,)
        """
        
        # Get the feature matrix
        feat_mat = self.get_feature(feature_name, format='matrix')
        
        # Binarize the feature matrix if a threshold is specified
        if threshold is not None:
            feat_mat = feat_mat <= threshold
        
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

    @staticmethod
    def compute_log_normalization(arr: np.ndarray, index: Index, method: str = 'genome-wide') -> np.ndarray:
        """Compute the log2 normalization of an array,
        i.e.  log2(arr / mean(arr)).
        
        It can be computed genome-wide (one global mean)
        or chromosome-wide (one mean per chromosome)

        Args:
            arr (np.array): Array to normalize.
            index (Index): Index object.
            method (str, optional): Either 'genome-wide' or 'chromosome-wide'.
                                    Defaults to 'genome-wide'.

        Returns:
            np.array: Normalized array.
        """
        
        # Get the haploid chromstr array
        chromstr_hap = index.chromstr[index.copy == 0]
        
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
            return _radial.run(struct_id, hss, params)
        if feature == 'lamina':
            return _lamina.run(struct_id, hss, params)
        if feature == 'lamina_tsa':
            return _lamina_tsa.run(struct_id, hss, params)
        if feature == 'speckle':
            return _body.run(struct_id, hss, params, what_to_measure='dist')
        if feature == 'speckle_tsa':
            return _body.run(struct_id, hss, params, what_to_measure='tsa')
        if feature == 'nucleoli':
            return _body.run(struct_id, hss, params, what_to_measure='dist')
        if feature == 'nucleoli_tsa':
            return _body.run(struct_id, hss, params, what_to_measure='tsa')
        if feature == 'transAB':
            return _transAB.run(struct_id, hss, params)
        if feature == 'icp':
            return _icp.run(struct_id, hss, params)
        if feature == 'rg':
            return _rg.run(struct_id, hss, params)
    