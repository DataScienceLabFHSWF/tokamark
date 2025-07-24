import joblib
import numpy as np

class PCATransform(object):
    """Use a pre-fitted PCA function to transform input data.

    Parameters
    ----------
    pca_model_path : str
        Path to fitted pca model joblib file.
    """

    def __init__(self, models):
        self.models = models

    def __call__(self, sample):
        vals, time, source_signal = sample["values"], sample["time"], sample["source-signal"]
        
        if vals is None or len(vals) == 0:
            return None

        # Select model type and signal from those available in the dictionary.
        pca_model =  self.models["pca"][source_signal]
                                
        # Apply Scaler and PCA
        if not np.isnan(vals).any():
            
            #Import models
            pca = pca_model["pca"]
            scaler = pca_model["scaler"]
            
            # Transoform data in PCA space
            x_scaled = scaler.transform(vals.T) 
            x_transform = pca.transform(x_scaled)
            #vals = scaler.inverse_transform(x_transform).T
            vals = x_transform.T
           
            
            return {"values":vals, "times":time, "source-signal":source_signal}
        else:
            return None
