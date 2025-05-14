from numba import jit
import numpy as np 
from typing import List
from .utils import get_beta, apply_seasonality, get_gamma_params
from datetime import datetime

def SEIR_compartment(inf_t0 : int, 
                     rec_t0 : int,
                     Nk : List[float], 
                     T : int, 
                     R0 : float, 
                     eps : float, 
                     mu : float, 
                     beta_B : float, 
                     mu_B : float, 
                     r : float, 
                     ifr : List[float],
                     Delta : float, 
                     C : List[List[float]], 
                     detection_rate : float,
                     dates : List[datetime], 
                     hemisphere : int, 
                     seasonality_min : float,
                     gamma : float = 0.0, 
                     deaths_delay : str = "fixed", 
                     daily_steps : float = 12, 
                     Delta_std: float = 1,
                     behavioral_mechanism : str = "global",
                     seed = None) -> dict: 
    """
    SEIR model with additional compartments to model behavioural reaction.
    Parameters
    ----------
        @param inf_t0 (int): initial number of infected 
        @param rec_t0 (int): initial number of recovered
        @param Nk (List[float]): number of individuals in different age groups
        @param T (int): simulation steps 
        @param R0 (float): basic reproductive number 
        @param eps (float): inverse of latent period
        @param mu (float): inverse of infectious period
        @param beta_B (float): transmission rate for behaviour
        @param mu_B (float): rate at which susceptibles give up behaviour change
        @param r (float): protection factor for behaviour change
        @param ifr (List[float]): infection fatality rate by age groups
        @param Delta (float): delay in deaths (mean)    
        @param C (List[List[float]]): contact matrix
        @param detection_rate (float): fraction of deaths that are reported
        @param dates (List[datetime]): list of simulation dates
        @param hemisphere (int): hemisphere (0: north, 1: tropical, 2: south)
        @param seasonality_min (float): seasonality parameter
        @param gamma (float): rate that regulates global behaviour change. Defaults to 0.0
        @param deaths_delay (str): method to calculate deaths delay. Defaults to "fixed" (alternative is "gamma")aily_steps (int): simulation time step. Defaults to 12
        @param daily_steps (int): simulation time step. Defaults to 12
        @param Delta_std (float): delay in deaths (std). Defaults to 1
        @param behavioral_mechanism (str): mechanism to compute the behavioral change. Defaults to "global" (alternative is "local")
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

    # initialize compartments and set initial conditions (S: 0, SB: 1, E: 2, I: 3, R: 4)
    compartments, deaths = np.zeros((5, n_age, T)), np.zeros((n_age, T))

    # compute delta t
    dt = 1. / daily_steps

    # log uniform prior
    gamma = 10**gamma

    # distribute intial infected and recovered among age groups
    for age in range(n_age):
        # I
        inf_t0_age = int(inf_t0 * Nk[age] / np.sum(Nk))
        compartments[3, age, 0] = int(int(inf_t0_age) * (1 / mu) / ((1 / mu) + (1 / eps)))
        # E 
        compartments[2, age, 0] = inf_t0_age - compartments[3, age, 0]
        # R 
        compartments[4, age, 0] = int(rec_t0 * Nk[age] / np.sum(Nk))
        # S
        compartments[0, age, 0] = Nk[age] - (compartments[2, age, 0] + compartments[3, age, 0] + compartments[4, age, 0])

    # simulate
    for t in np.arange(1, T, 1): 

        # get seasonal forcing 
        st = apply_seasonality(day=dates[t-1], seasonality_min=seasonality_min, basin_hemispheres=hemisphere)

        # iterate over daily steps
        compartments_next_day = compartments[:, :, t-1].copy()
        new_R_day = np.zeros(n_age, dtype=int)
        for _ in range(int(daily_steps)):   
            # decide global or local mechanism
            if behavioral_mechanism == "global":
                prob_S_to_SB = beta_B * (1 - np.exp(-gamma * np.sum(deaths[:, t-1])))
                prob_S_to_SB = np.full(n_age, prob_S_to_SB) # resize to array
            elif behavioral_mechanism == "local": 
                prob_S_to_SB = np.sum(beta_B * C * deaths[:, t-1] / Nk, axis=1)
            else: 
                raise ValueError("Invalid behavioral mechanism. Choose between 'global' and 'local'.")
    
            # multinomial sample from S 
            prob_S_to_I = np.sum(st * beta * C * compartments_next_day[3, :] / Nk, axis=1)
            total_leaving = np.random.binomial(compartments_next_day[0, :].astype(int), 1 - np.exp(-(prob_S_to_SB + prob_S_to_I) * dt))
            new_E = np.random.binomial(total_leaving, np.array([a / b if b != 0 else 0 for a, b in zip(prob_S_to_I, prob_S_to_SB + prob_S_to_I)]))
            new_SB = total_leaving - new_E

            # multinomial sample from SB 
            prob_SB_to_S = mu_B * (np.sum(compartments_next_day[0, :]) + np.sum(compartments_next_day[4, :])) / np.sum(Nk)
            prob_SB_to_S = np.full(n_age, prob_SB_to_S) # resize to array
            prob_SB_to_I = np.sum(st * r * beta * C * compartments_next_day[3, :] / Nk, axis=1)
            total_leaving = np.random.binomial(compartments_next_day[1, :].astype(int), 1 - np.exp(-(prob_SB_to_S + prob_SB_to_I) * dt))
            new_E_fromSB = np.random.binomial(total_leaving, np.array([a / b if b != 0 else 0 for a, b in zip(prob_SB_to_I, prob_SB_to_S + prob_SB_to_I)]))
            new_S = total_leaving - new_E_fromSB

            new_I = np.random.binomial(compartments_next_day[2, :].astype(int), 1 - np.exp(-eps * dt))
            new_R = np.random.binomial(compartments_next_day[3, :].astype(int), 1 - np.exp(-mu * dt))

            # update compartments_next_day
            # S
            compartments_next_day[0, :] = compartments_next_day[0, :] - new_E - new_SB + new_S
            # SB 
            compartments_next_day[1, :] = compartments_next_day[1, :] - new_E_fromSB - new_S + new_SB
            # E
            compartments_next_day[2, :] = compartments_next_day[2, :] + new_E + new_E_fromSB - new_I
            # I 
            compartments_next_day[3, :] = compartments_next_day[3, :] + new_I - new_R 
            # R
            compartments_next_day[4, :] = compartments_next_day[4, :] + new_R

            # store new_R for deaths computation
            new_R_day += new_R

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
            'SB': compartments[1].sum(axis=0),  
            'E': compartments[2].sum(axis=0),  
            'I': compartments[3].sum(axis=0),  
            'R': compartments[4].sum(axis=0), 
            'deaths_age': deaths, 
            'deaths_actual': ((1. / detection_rate) * deaths.sum(axis=0)).astype(int), 
            'deaths': deaths_observed}
