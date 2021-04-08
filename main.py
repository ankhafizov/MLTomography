import wavelet_cpl_extractor as wce
import grimax_sigma_extractor as gse
import grimax_gaus_sigma_extractor as gse_gaus
import pandas as pd
import phantom_generator as pg
import data_manager as dm

import timeit
import numpy as np

if __name__ == "__main__":

    df = pd.DataFrame(columns = ['method_name',
                                 'phantom_porosity',
                                 'phantom_sigma',
                                 'calculated_sigma',
                                 'deviation_percent',
                                 'processing_time_seconds'])


    phantom_shape = [1, 1_000_000]

    for phantom_porosity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for phantom_sigma in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            phantom = ~pg.gen_phantom(phantom_shape, phantom_porosity, phantom_sigma)
            
            for method_name in ["anvar", "grimax", "grimax_gaus"]:
                #### main body ####
                start = timeit.default_timer()

                if method_name == "anvar":
                    extractor_func = wce.extract_cpl
                elif method_name == "grimax":
                    phantom = phantom.T
                    gse.sigma_estimate_2
                elif method_name == "grimax_gaus":
                    phantom = phantom.T
                    gse_gaus.sigma_estimate_2

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