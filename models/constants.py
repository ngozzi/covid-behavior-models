from datetime import datetime

## EPIDEMIOLOGICAL PARAMETERS COMMON TO ALL SIMULATIONS

# 1 / infectious period
mu = 1. / 2.5

# 1 / latent period
eps = 1. / 4.

# infection fatality rate
IFR = [0.00161 / 100, # 0-9 
        0.00695 / 100, # 10-19
        0.0309  / 100, # 20-29
        0.0844  / 100, # 30-39
        0.161   / 100, # 40-49
        0.595   / 100, # 50-59
        1.93    / 100, # 60-69
        4.28    / 100, # 70-79
        7.80    / 100] # 80+

# infection fatality rate up to 75+
IFR_75plus = [0.00161 / 100, # 0-9 
        0.00695 / 100, # 10-19
        0.0309  / 100, # 20-29
        0.0844  / 100, # 30-39
        0.161   / 100, # 40-49
        0.595   / 100, # 50-59
        1.93    / 100, # 60-69
        4.28    / 100, # 70-74
        6.04    / 100] # 75+


# infection fatality rate for 10 age groups
IFR_10age = [0.00161 / 100, # 0-9 
             0.00695 / 100, # 10-19
             0.0309  / 100, # 20-24
             0.0309  / 100, # 25-29
             0.0844  / 100, # 30-39
             0.161   / 100, # 40-49
             0.595   / 100, # 50-59
             1.93    / 100, # 60-69
             4.28    / 100, # 70-79
             7.80    / 100] # 80+


simulation_dates = {"london": [datetime(year=2020, month=3, day=8), datetime(year=2020, month=7, day=5)],
                    "new_york": [datetime(year=2020, month=3, day=8), datetime(year=2020, month=8, day=2)],
                    "chicago": [datetime(year=2020, month=3, day=15), datetime(year=2020, month=9, day=13)],
                    "santiago": [datetime(year=2020, month=3, day=29), datetime(year=2021, month=11, day=1)],
                    "rio_de_janeiro": [datetime(year=2020, month=3, day=15), datetime(year=2020, month=11, day=8)],
                    "bogota": [datetime(year=2020, month=4, day=28), datetime(year=2020, month=11, day=1)],
                    "gauteng": [datetime(year=2020, month=3, day=29), datetime(year=2020, month=11, day=15)],
                    "jakarta": [datetime(year=2020, month=3, day=1), datetime(year=2020, month=5, day=31)],
                    "madrid": [datetime(year=2020, month=3, day=1), datetime(year=2020, month=7, day=12)]}