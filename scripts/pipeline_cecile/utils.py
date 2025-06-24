import os
import pandas as pd
import sys

from torch.utils.data import DataLoader, Dataset

cwd = os.path.dirname(os.getcwd())
mother_dir = os.path.dirname(cwd) + os.sep
# print(mother_dir)
sys.path.append(os.path.abspath(os.path.join(mother_dir , "fairmast-data-preprocessing/scripts")))
sys.path.append(mother_dir)
sys.path.append(cwd)
sys.path.append(os.path.join( os.path.dirname(cwd) ) )

from torch.utils.data._utils.collate import default_collate


#================================================================
        ##########   DATA-SPLIT  ##########
#================================================================
def read_data_split_csv(csv_path="/home/ir-rous1/hncdi-fusion-plasma/fairmast-data-preprocessing/metadata/2025-05-12/data_splits.csv"):
    """Read the csv file containing the lists of shot IDs for 
    training, validation and testing.

    Parameters
    ----------
    csv_path : str, optional
        by default "metadata/2025-05-12/data_splits.csv"

    Returns
    -------
    Three lists containing shot IDs for training, validation and testing sets
    """

    df = pd.read_csv(csv_path)

    # Filter rows where the 'train' column is True
    shot_ids_for_train = df[df['train'] == True]['shot_id'].tolist() 
    shot_ids_for_test = df[df['test'] == True]['shot_id'].tolist() 
    shot_ids_for_val = df[df['val'] == True]['shot_id'].tolist() 

    return shot_ids_for_train, shot_ids_for_test, shot_ids_for_val
#================================================================
        ##########   END OF DATA-SPLIT   ##########
#================================================================



#================================================================
        ##########  COLLATE FUNCTION  ##########
#================================================================
def flatten_then_collate(batch):

    print(f"Collating batch of size {len(batch)}")
    
    # Flatten the batch of lists into a single list
    if isinstance(batch[0], list) :
        flattened_batch = [item for sublist in batch for item in sublist]
        print(f'Number of samples from batch = {len(batch)} shots is N = {len(flattened_batch)}')
    # Use the default collate function
    return default_collate(flattened_batch)

#================================================================
        ##########  END OF COLLATE FUNCTION  ##########
#================================================================