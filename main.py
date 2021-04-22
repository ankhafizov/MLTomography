import wavelet_sigma_extractor as wse
import grimax_sigma_extractor as gse
import pandas as pd
import phantom_generator as pg
import data_manager as dm
import random

import timeit
import numpy as np
from progress.bar import Bar


class InputParameters():
    def __init__(self):
        self.shape = [1, 1_000_000]
        self.porosities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.sigmas = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.noise = 0.01 # set 0 for no noise


if __name__ == "__main__":

    df = pd.DataFrame(columns = ['noise',
                                 'method_name',
                                 'phantom_porosity',
                                 'phantom_sigma',
                                 'calculated_sigma',
                                 'deviation_percent',
                                 'processing_time_seconds'])

    parameters = InputParameters()
    bar = Bar('Processing', max=len(parameters.porosities)*len(parameters.sigmas))

    for phantom_porosity in parameters.porosities:
        for phantom_sigma in parameters.sigmas:
            phantom = ~pg.generate_phantom(parameters.shape, phantom_porosity, phantom_sigma)
            phantom = pg.sp_noise(phantom, prob=parameters.noise).astype(bool)
            
            for method_name in ["anvar", 
                                "grimax_h",
                                "grimax_smoothed",
                                "grimax_gaus"]:
                #### main body ####
                start = timeit.default_timer()

                if method_name == "anvar":
                    calculated_sigma = wse.get_sigma(phantom, phantom_porosity)
                elif method_name == "grimax_h":
                    calculated_sigma = gse.sigma_estimate_smoothed_histogram(phantom, mode="default")
                elif method_name == "grimax_smoothed":
                    calculated_sigma = gse.sigma_estimate_smoothed_histogram(phantom, mode="smoothed")
                elif method_name == "grimax_gaus":
                    calculated_sigma = gse.sigma_estimate_smoothed_histogram(phantom, mode="gaus")
                
                stop = timeit.default_timer()
                #### main body ####

                df = df.append({'noise': parameters.noise>0,
                                'method_name': method_name,
                                'phantom_porosity': phantom_porosity,
                                'phantom_sigma': phantom_sigma,
                                'calculated_sigma': calculated_sigma,
                                'deviation_percent': np.abs(calculated_sigma - phantom_sigma) / phantom_sigma * 100,
                                'processing_time_seconds': stop-start}, ignore_index=True)
                
                dm.save_dataframe(df, "sigma_extraction_contest.csv")
            
            bar.next()