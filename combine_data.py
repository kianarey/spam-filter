import numpy as np
import pandas as pd

def shuffle_and_get_data(data):
    shuffle_data_idx = np.random.randint(low=0, high=len(data)-1, size=len(data))
    ret_data = data[shuffle_data_idx]
    return ret_data

ns_df = pd.read_csv('ns_data.txt', header=None, keep_default_na=False)
s_df = pd.read_csv('s_data.txt',header=None, keep_default_na=False)

ns_df = pd.DataFrame.to_numpy(ns_df)
s_df = pd.DataFrame.to_numpy(s_df) 

ns_df = ns_df.ravel().reshape(len(ns_df),1)
s_df = s_df.ravel().reshape(len(s_df),1)

ns_label = np.zeros((len(ns_df),1))
s_label = np.ones((len(s_df),1))

ns = np.append(ns_df, ns_label, axis=1)
s = np.append(s_df, s_label, axis=1)

all_data = np.concatenate((ns, s))
all_data = shuffle_and_get_data(all_data)

all_data_df = pd.DataFrame(all_data)
all_data_df.to_csv('all_data.txt', header=False, index=False)