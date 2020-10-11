from Pipeline_P2 import pipeline_Part_Two
import sys

def main():
    """
    This method is for testing the pipeline.
    This method takes the command line argument as path, 
    and indices to split the values is specified here
    Input :

    Output :
    
    """
    split_values = [2000, 7000]
    path = sys.argv[1]
    pipeline = pipeline_Part_Two(path, split_values)
    meas_seq_percentage = pipeline.process_Pipeline_Two()

if __name__ == "__main__":
    main()