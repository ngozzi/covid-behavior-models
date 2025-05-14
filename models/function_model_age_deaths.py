import numpy as np 
from numba import jit
from typing import List
from datetime import datetime
from .utils import get_beta, apply_seasonality, get_gamma_params

def SEIR_function(inf_t0 : int, 
                  rec_t0 : int,
                  Nk : List[float], 
                  T : int, 
                  R0 : float, 
                  eps : float, 
                  mu : float,
                  ifr : List[float],
                  Delta : float, 
                  dc : float, 
                  Dc : float, 
                  C : List[List[float]], 
                  detection_rate : float,
                  dates : List[datetime], 
                  hemisphere : int, 
                  seasonality_min : float,
                  long_term : bool = False,
                  short_term : bool = True, 
                  k : int = 1, 
                  Delta_std: float = 1,
                  deaths_delay : str = "fixed", 
                  daily_steps : float = 12, 
                  seed = None) -> dict: 
    """
    SEIR model with mobility data used to modulate the force of infection.
    Parameters
    ----------
        @param inf_t0 (int): initial number of infected 
        @param rec_t0 (int): initial number of recovered
        @param Nk (List[float]): number of individuals in different age groups
        @param T (int): simulation steps 
        @param R0 (float): basic reproductive number 
        @param eps (float): inverse of latent period
        @param mu (float): inverse of infectious period
        @param ifr (List[float]): infection fatality rate by age groups
        @param Delta (float): delay in deaths (mean)
        @param Delta_std (float): delay in deaths (std)   
        @param ic (float): short-term risk parameter
        @param Ic (float): long-term risk parameter
        @param C (List[List[float]]): contact matrix
        @param detection_rate (float): fraction of deaths that are reported
        @param dates (List[datetime]): list of simulation dates
        @param hemisphere (int): hemisphere (0: north, 1: tropical, 2: south)
        @param seasonality_min (float): seasonality parameter
        @param long_term (bool, optional): if True also long-term memory mechanism is used. Defaults to False
        @param short_term (bool, optional): if True short-term memory mechanism is used. Defaults to True
        @param k (int, optional): parameter that sets the speed of behaviour change. Defaults to 1
        @param Delta_std (float): delay in deaths (std). Defaults to 1
        @param deaths_delay (str): method to calculate deaths delay. Defaults to "fixed" (alternative is "gamma")
        @param daily_steps (int): simulation time step. Defaults to 12
        @param seed (int): random seed
    Return 
    ------
        @return: dictionary of compartments and deaths
    """
    
    if seed is not None: 
        np.random.seed(seed)

    # number of age groups 
    n_age = len(Nk)
    
    # compute beta
    beta = get_beta(R0, mu, Nk, C)

    # compute deaths delay parameters
    shape, scale = get_gamma_params(Delta, Delta_std)
    
    # initialize compartments and set initial conditions (S: 0, E: 1, I: 2, R: 3)
    compartments, deaths = np.zeros((4, n_age, T)), np.zeros((n_age, T))

    # compute delta t
    dt = 1. / daily_steps

    # log unif priors
    dc = 10**dc
    Dc = 10**Dc

    # distribute intial infected and recovered among age groups
    for age in range(n_age):
        # I
        inf_t0_age = int(inf_t0 * Nk[age] / np.sum(Nk))
        compartments[2, age, 0] = int(int(inf_t0_age) * (1 / mu) / ((1 / mu) + (1 / eps)))
        # E 
        compartments[1, age, 0] = inf_t0_age - compartments[2, age, 0]
        # R 
        compartments[3, age, 0] = int(rec_t0 * Nk[age] / np.sum(Nk))
        # S
        compartments[0, age, 0] = Nk[age] - (compartments[1, age, 0] + compartments[2, age, 0] + compartments[3, age, 0])

    # simulate
    for t in np.arange(1, T, 1): 

        # get seasonal forcing 
        st = apply_seasonality(day=dates[t-1], seasonality_min=seasonality_min, basin_hemispheres=hemisphere)

        # iterate over daily steps
        compartments_next_day = compartments[:, :, t-1].copy()
        new_R_day = np.zeros(n_age, dtype=int)
        new_infections_day = 0

        # quantities needed for behaviour function
        last_day_deaths = np.sum(deaths[:, t-1])
        cumulative_deaths = np.sum(deaths[:, :t-1])
        
        for _ in range(int(daily_steps)):   

            # compute behaviour function 
            bf = 1 / (1 + short_term * (last_day_deaths / dc)**k + long_term * (cumulative_deaths / Dc)**k)

            # compute force of infection  
            force_inf = np.sum(st * bf * beta * C * compartments_next_day[2, :] / Nk, axis=1)

            # compute transitions
            new_E = np.random.binomial(compartments_next_day[0, :].astype(int), 1 - np.exp(-force_inf * dt))
            new_I = np.random.binomial(compartments_next_day[1, :].astype(int), 1 - np.exp(-eps * dt))
            new_R = np.random.binomial(compartments_next_day[2, :].astype(int), 1 - np.exp(-mu * dt))

            # update next step solution
            # S
            compartments_next_day[0, :] = compartments_next_day[0, :] - new_E
            # E
            compartments_next_day[1, :] = compartments_next_day[1, :] + new_E - new_I
            # I 
            compartments_next_day[2, :] = compartments_next_day[2, :] + new_I - new_R
            # R
            compartments_next_day[3, :] = compartments_next_day[3, :] + new_R

            # store new_R for deaths computation and new_I for FoI computation
            new_R_day += new_R
            new_infections_day += np.sum(new_I)

        # update compartments
        compartments[:, :, t] = compartments_next_day

        # compute deaths 
        if deaths_delay == "fixed":
            if (t - 1) + Delta < deaths.shape[1]:
                deaths[:, (t - 1) + int(Delta)] = (detection_rate * np.random.binomial(new_R_day, ifr)).astype(int)
                
        elif deaths_delay == "gamma":
            new_deaths = np.random.binomial(new_R_day, ifr)
            delays = np.random.gamma(shape=shape, scale=scale, size=np.sum(new_deaths))
            count = 0
            for i, d in enumerate(new_deaths):
                for _ in range(d):
                    # sample delay
                    delay = int(delays[count])
                    if t + delay < deaths.shape[1]:
                        deaths[i, t + delay] += 1
                    count += 1
        
    # apply deaths undereporting
    deaths_observed = (deaths.sum(axis=0)).astype(int)
    
    return {'compartments': compartments, 
            'S': compartments[0].sum(axis=0),  
            'E': compartments[1].sum(axis=0),  
            'I': compartments[2].sum(axis=0),  
            'R': compartments[3].sum(axis=0), 
            'deaths_age': deaths, 
            'deaths_actual': ((1. / detection_rate) * deaths.sum(axis=0)).astype(int), 
            'deaths': deaths_observed}