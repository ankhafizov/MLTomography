import wavelet_cpl_extractor as wce
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
                                 'processing_time'])


    phantom_shape = [500, 500]

    for phantom_porosity in [0.2, 0.4]:
        for phantom_sigma in [10, 15]:
            
            for method_name in ["anvar"]:
                #### main body ####
                start = timeit.default_timer()
                
                #TODO: add grimax
                extractor_func = wce.extract_sigma if method_name == "anvar" else wce.extract_sigma
                phantom = ~pg.gen_phantom(phantom_shape, phantom_porosity, phantom_sigma)
                calculated_sigma = extractor_func(phantom)
                
                stop = timeit.default_timer()
                #### main body ####

                df = df.append({'method_name': method_name,
                                'phantom_porosity': phantom_porosity,
                                'phantom_sigma': phantom_sigma,
                                'calculated_sigma': calculated_sigma,
                                'deviation_percent': np.abs(calculated_sigma - phantom_sigma) / phantom_sigma * 100,
                                'processing_time_seconds': stop-start}, ignore_index=True)
                
                dm.save_dataframe(df, "sigma_extraction_contest.csv")