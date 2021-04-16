import wavelet_cpl_extractor as wce
import grimax_sigma_extractor as gse
import grimax_gaus_sigma_extractor as gse_gaus
import pandas as pd
import phantom_generator as pg
import data_manager as dm
import random

import timeit
import numpy as np
from progress.bar import Bar


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output


if __name__ == "__main__":

    df = pd.DataFrame(columns = ['method_name',
                                 'phantom_porosity',
                                 'phantom_sigma',
                                 'calculated_sigma',
                                 'deviation_percent',
                                 'processing_time_seconds'])


    phantom_shape = [1, 1_000_000]
    porosities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sigmas = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    bar = Bar('Processing', max=len(porosities)*len(sigmas))

    for phantom_porosity in porosities:
        for phantom_sigma in sigmas:
            phantom = ~pg.generate_phantom(phantom_shape, phantom_porosity, phantom_sigma)
            phantom = sp_noise(phantom, 0.01).astype(bool)
            
            for method_name in ["anvar", 
                                "grimax",
                                "grimax_gaus"]:
                #### main body ####
                start = timeit.default_timer()

                if method_name == "anvar":
                    extractor_func = wce.extract_cpl
                    phantom = phantom if phantom_porosity <=0.5 else ~phantom.astype(bool)
                elif method_name == "grimax":
                    phantom = phantom.T
                    extractor_func = gse.sigma_estimate_2
                elif method_name == "grimax_gaus":
                    extractor_func = gse_gaus.sigma_estimate_2

                #TODO: put magic_coef in other script
                magic_coef = 1.6 if method_name == "anvar" else 1
                calculated_sigma = extractor_func(phantom) / magic_coef
                
                stop = timeit.default_timer()
                #### main body ####

                df = df.append({'method_name': method_name,
                                'phantom_porosity': phantom_porosity,
                                'phantom_sigma': phantom_sigma,
                                'calculated_sigma': calculated_sigma,
                                'deviation_percent': np.abs(calculated_sigma - phantom_sigma) / phantom_sigma * 100,
                                'processing_time_seconds': stop-start}, ignore_index=True)
                
                dm.save_dataframe(df, "sigma_extraction_contest.csv")
            
            bar.next()