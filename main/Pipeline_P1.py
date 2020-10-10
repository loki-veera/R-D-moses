import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class pipeline_Part_One():
    """
    This class implements the first part of the pipeine, 
    i.e., implementing the algoriothm to detect the anomalous data in a testrun.
    The pipeline consists of the preprocessing of the testruns, followed by PCA then
    the results from PCA is fed into DBSCAN to detect the anomalous datapoints.
    """
    
    def __init__(self, path):
        """
        Constructor to read the path to the measurement sequence.
        Input : 
            path : Path to the measurement sequence
        Output :

        """
        self.path = path
    
    def read_Textfiles(self):
        """
        This method reads the measurement sequence from the given path.
        Input :

        Output :
            data : List of dataframes containing the data for measurement sequence 
        """
        files = sorted(os.listdir(self.path))
        print("No. of files to be processed are: ", len(files), end ="\n")
        count  = 0
        data = []
        dup_path = self.path
        for each_file in files:
            dup_path += each_file
            print(count+1, end=',')
            data.append(pd.read_csv(dup_path, sep = '\t', engine = 'python', decimal = ',', skiprows = 40)[2: ].apply(pd.to_numeric))
            count += 1
            dup_path = self.path
        print()
        print("No. of files processed are: ", count)
        return data

    def preprocess_Testruns(self, file_data):
        """
        This method preprocesses the testruns based on the features which are don't
        convey any information. This method is only part of the pipeline which needs 
        a replacement for the new maneuver. The needed features are selected based on 
        the importance from the manual, and also standard deviation from the data.
        Input : 
            file_data : The list of raw dataframes
        Output : 
            file_data : List of dataframes with preprocessed data
        """
        needed_features = ['Lenkradwin', 'Lenkmoment', 'Fahrgeschw', 'Schwimmwin', 'F_Spur_VL', 'F_Spur_VR', 'Querbeschl', 
                           'Giergeschw', 'Nickwinkel', 'Wankwinkel', 'Gierwinkel', 'Nickgeschw', 'Wankgeschw', 'Hochbeschl',
                           'Fahrge_DIS', 'Schwim_MSP', 'Schwim_MHA', 'Radius', 'Fdiff_Spur', 'Lichtschra', 'LenkgeschM', 
                           'StWhl_Angl', 'VehSpd_Disp', 'VehAccel_X_V2', 'VehAccel_Y_V2', 'VehYawRate_Raw', 'WhlRPM_FL', 
                           'WhlRPM_FR', 'WhlRPM_RL', 'WhlRPM_RR', 'EngRPM']
        
        for each_file in range(len(file_data)):
            cols = file_data[each_file]
            for each_col in cols:
                if each_col not in needed_features:
                    file_data[each_file] = file_data[each_file].drop(columns=each_col).dropna(axis='index')
        return file_data

    def perform_PCA(self, input_values):
        """
        This method applies PCA over the data since lot of correlation is observed.
        Steps include:
            1. Standardizing the data
            2. Compute the covariance matrix
            3. Calulcate the eigen values and eigen vectors
            4. Calculate explained variance to estimate the k-number of principal components
            5. Prepare weight matrix from the k largest eigen vectors.
            6. Reproject the data on the weight matrix.
        Input :
            input_values : DPreprocessed dataframe of single test run containing the measurements.
        Output : 
            np_data_std_projected : Dataframe constituting the reduced number of dimensions
        """
        np_data = input_values.to_numpy()
        np_data_std = (np_data-np.mean(np_data, axis = 0))/np.std(np_data, axis = 0)
        np_data_std = np_data
        cov_matrx = (np_data_std-np.mean(np_data_std, axis =0)).T.dot(np_data_std-np.mean(np_data_std, axis =0)) / (np_data_std.shape[0]-1)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_matrx)
        eigen_tuples = [(eigen_vals[idx], eigen_vecs[:, idx]) for idx in range(len(eigen_vals))]
        eigen_tuples.sort(key = lambda x:x[0], reverse=True)
        eigen_vals_sum = sum(eigen_vals)
        exp_variance = [(eigen_val*100)/eigen_vals_sum for eigen_val in sorted(eigen_vals, reverse=True)]
        cum_exp_variance = np.cumsum(exp_variance)
        no_dim = 0
        for val in cum_exp_variance:
            if val < 96:
                no_dim += 1
            else:
                break
        no_dim = 9
        W_matrix = np.zeros(shape = (len(eigen_vals),1))
        for idx, each_tuple in enumerate(eigen_tuples):
            if idx < no_dim:
                W_matrix = np.hstack((W_matrix, each_tuple[1].reshape(len(eigen_vals),1)))
        W_matrix = W_matrix[: ,1:]
        np_data_std_projected = np_data_std.dot(W_matrix)
        return np_data_std_projected

    def perform_DBSCAN(self, values):
        """
        This method applies the DBSCAN algorithm to the PCA resulted output testruns 
        and results the labels whether as anomaly or not and number of unique labels, number of datapoints for each labels 
        Input : 
            values : Dataframe constituing the data withe reduced dimensions
        Output :
            labels : Labels data for all the datapoints
            unique : Number of unique cluster the DBSCAN found
            counts : Number of datapoints for each and every cluster
        """
        clustering = DBSCAN(eps=3, min_samples=10).fit(values)
        labels = clustering.labels_
        unique, counts = np.unique(labels, return_counts=True)
        return labels, unique, counts

    def process_Pipeline_One(self):
        """
        This method acts as meta method for all the previous.
        i.e., this method acts as backbone which connects all other methods in this class.
        Input :

        Output :

        """
        file_data = self.read_Textfiles()
        file_data = self.preprocess_Testruns(file_data)
        labels = []
        for each_testrun in file_data:
            pca_values = self.perform_PCA(each_testrun)
            lbls, _, _ = self.perform_DBSCAN(pca_values)
            labels.append(lbls)
        return file_data, labels
