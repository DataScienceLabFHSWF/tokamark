import numpy as np 
import warnings


# ----------------------------------------------------------------------------------------------------------------------
def get_mean_shot(dataset):
    
    dict_mean_list = {var: [] for var in dataset[0].keys()}
    for data in dataset:
        for var, data_var in data.items():
            # print(var)
            if data_var['values'] is not None:
                if len(dict_mean_list[var]) != 0 :
                    with warnings.catch_warnings():
                        if dict_mean_list[var][0].shape == np.nanmean(data_var['values'], axis=-1).shape :
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            dict_mean_list[var].append(np.nanmean(data_var['values'], axis=-1))
                        else:
                            print(f'Shape different for variable "{var}" of one shot! Skipping')
                else : 
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        dict_mean_list[var].append(np.nanmean(data_var['values'], axis=-1))
    
    dict_mean = {}
    for var, list_ in dict_mean_list.items():
        print(f'Shapes in dict_mean_list for var {var}', [arr.shape for arr in list_])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dict_mean[var] = np.nanmean(list_)

    return dict_mean


# ----------------------------------------------------------------------------------------------------------------------
def get_std_shot(dataset):
    
    dict_std_list = {var: [] for var in dataset[0].keys()}
    for data in dataset:
        for var, data_var in data.items():
            # print(var)
            if data_var['values'] is not None:
                if len(dict_std_list[var]) != 0 :
                    with warnings.catch_warnings():
                        print(dict_std_list[var][0].shape)
                        print(np.nanstd(data_var['values'], axis=-1).shape)
                        if dict_std_list[var][0].shape == np.nanstd(data_var['values'], axis=-1).shape :
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            dict_std_list[var].append(np.nanstd(data_var['values'], axis=-1))
                        else:
                            print(f'Shape different for variable "{var}" of one shot! Skipping')
                else : 
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        dict_std_list[var].append(np.nanstd(data_var['values'], axis=-1))
    
    dict_std = {}
    for var, list_ in dict_std_list.items():
        print(f'Shapes in dict_std_list for var {var}', [arr.shape for arr in list_])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dict_std[var] = np.nanmean(list_)

    return dict_std
    