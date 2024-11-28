import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random

def read_table_from_file(file_path, cols, sep="", dtype=dict(col: object for col in cols)):
    df = pd.read_table(file_path, sep=sep, names=cols, header=None, dtype=dtype, na_values=["\\N", "nan", "null", "NaN", ""])
    return df

def read_table_from_dir(dir_path, cols, sep="", filetype="txt", num=None, shuffle=False):
    if not num:
        num = len(os.listdir(dir_path))
    
    dtype = {col: object for col in cols}
    df_list = []
    
    if os.path.isfile(dir_path):
        file_list = [dir_path]
    else:
        file_list = os.listdir(dir_path)
    
    if shuffle:
        random.shuffle(file_list)
    
    file_list = file_list[:num]  # Limit to num files if specified
    
    # Using ProcessPoolExecutor for concurrent processing
    with ProcessPoolExecutor() as executor:
        # Create a list of futures
        futures = [executor.submit(read_table_from_file, os.path.join(dir_path, filename), cols, sep, dtype) for filename in file_list if filename.endswith(filetype)]
        
        # As each process completes, append the result to df_list
        for future in tqdm(futures, total=len(futures), desc='Reading files'):
            df_list.append(future.result())
    
    df = pd.concat(df_list).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = read_table_from_dir("data/raw/train", ["id", "text", "label"], sep="\t", filetype="txt", num=1000, shuffle=True)
    print(df.head())