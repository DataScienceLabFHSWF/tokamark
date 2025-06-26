import numpy as np 
import warnings

def get_mean_shot(dataset):
    
    dict_mean_list={var:[] for var in dataset[0].keys()}
    for data in dataset:
        for var, data_var in data.items():
            # print(var)
            if data_var['values'] is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dict_mean_list[var].append(np.nanmean(data_var['values'], axis=-1))
    
    dict_mean={}
    for var, list in dict_mean_list.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dict_mean[var]=np.nanmean(list)

    return dict_mean

def get_std_shot(dataset):
    
    dict_std_list={var:[] for var in dataset[0].keys()}
    for data in dataset:
        for var, data_var in data.items():
            # print(var)
            if data_var['values'] is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dict_std_list[var].append(np.nanstd(data_var['values'], axis=-1))
    
    dict_std={}
    for var, list in dict_std_list.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dict_std[var]=np.nanmean(list)

    return dict_std

    