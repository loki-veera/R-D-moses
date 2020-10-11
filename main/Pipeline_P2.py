from Pipeline_P1 import pipeline_Part_One
import numpy as np

class pipeline_Part_Two(pipeline_Part_One):
    """
    This class implements the second stage of the pipeline and it inherits the propertied from the 
    first stage of the pipeline. This stage of the pipeline is is again specific to the maneuver.
    In SK maneuver, we use Querbeschleunigung as the standard feature. We split the data into 3 sectors
    at the standard values of the Querbeschleunigung. Then the number of invalid data in the sector is 
    considered and then the weighted average is considered. 
    """
    def __init__(self, path, split_values):
        """
        Constructor to store the indices of the values to split the data
        Input :
            split_values : List of indices to split the data
        Output :

        """
        super().__init__(path)
        self.split_values = split_values
    
    def cut_lables(self, labels):
        """
        This method is used to cut the testruns at particular values from the constructor
        Input : 
            labels : List of labels for the testruns
        Output :
            proc_labels : List of list, where the testruns are split at the split_values 
        """
        proc_labels = []
        self.split_values.append(len(labels))
        proc_labels.append(labels[0 : self.split_values[0]])
        proc_labels.append(labels[self.split_values[0] : self.split_values[1]])
        proc_labels.append(labels[self.split_values[1] : self.split_values[-1]])
        return proc_labels

    def get_Testrun_Percentage(self, testrun_lbls):
        """
        This method is used to calculate the percentage of valid data in the sector of the testrun
        Input :
            testrun_lbls : List containing the labels of the particular sectors.
        Output :
            percentage : Percentage of invalid data in the testruns
        """
        unique, counts = np.unique(testrun_lbls, return_counts=True)
        index = np.where(unique == -1)
        is_empty = index[0].size == 0
        sum_val = 0
        for each_unq in unique:
            if each_unq != 0:
                index = np.where(unique == each_unq)
                sum_val += counts[index]
        if is_empty == False:
            percentage = sum_val/len(testrun_lbls)
            percentage *= 100
            return percentage[0]
        else:
            return 0

    def get_Percentage_Measurement_Sequence(self, file_data, proc_lbls_1, proc_lbls_2, proc_lbls_3):
        """
        This method is used to calculate the percentage of invalid data in all the sectors of a testrun.
        At present this method can only implemented over the 3 testruns in a measurement sequence.
        Input : 
            proc_labels_1 : Split labels into sectors for the first testrun
            proc_labels_2 : Split labels into sectors for the second testrun
            proc_labels_3 : Split labels into sectors for the third testrun
        Output :
            valid_percent :Float values specifying the amount of valid data in a testrun
        """
        if len(proc_lbls_1) == len(proc_lbls_2) == len(proc_lbls_3):
            pass
        else:
            print("Cutting is not the same")
            return 0
        avg_percent = []
        for index in range(0, len(proc_lbls_1)):
            percent_1 = self.get_Testrun_Percentage(proc_lbls_1[index])
            percent_2 = self.get_Testrun_Percentage(proc_lbls_2[index])
            percent_3 = self.get_Testrun_Percentage(proc_lbls_3[index])
            print("Sector ",str(index+1)," percentage in measurement sequence: ")
            print("\tT1: ",percent_1)
            print("\tT2: ",percent_2)
            print("\tT3: ",percent_3)
            print("\tAverage percentage (invalid): ")
            avg = (percent_1+percent_2+percent_3)/3
            print("\tAvg: ", avg)
            avg_percent.append(avg)
        print()
        print("Total average percentages (invalid): ")
        print(avg_percent)
        print("Total validity in measurement sequence is [defined as weighted average]: ")
        total_values = np.mean(self.split_values[2: ])
        t1 = self.split_values[0]/total_values
        t2 = (self.split_values[1]-self.split_values[0])/total_values
        t3 = (total_values-self.split_values[1])/total_values
        valid_percent = ((t1*avg_percent[0])+(t2*avg_percent[1])+(t3*avg_percent[2]))
        print(100-valid_percent)
        print("Avegrae value is: ")
        print(100-np.mean(avg_percent))
        valid_percent = 100-(np.mean(avg_percent))
        return valid_percent

    def process_Pipeline_Two(self):
        """
        This method act as meta method to calculate the amount of valid measurements in a measurement
        sequence.
        Input :

        Output :
            valid_percentage : Percentage of valid measurements in the measurement sequence
        """
        filedata, labels = self.process_Pipeline_One()
        proc_lbls_0 = self.cut_lables(labels[0])
        proc_lbls_2 = self.cut_lables(labels[1])
        proc_lbls_4 = self.cut_lables(labels[2])
        valid_percentage = self.get_Percentage_Measurement_Sequence(filedata, proc_lbls_0, proc_lbls_2, proc_lbls_4)
        return valid_percentage

