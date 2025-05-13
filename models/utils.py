import numpy as np 
import pandas as pd
from typing import List
from numba import jit
from .constants import IFR
import pyabc
import os
import uuid
import pickle as pkl
from datetime import timedelta, datetime
from typing import Callable, List
from pyabc.sampler import MulticoreEvalParallelSampler


def run_calibration(model_name, basin, deaths_delay, run_name):
    """
    This function runs a calibration file
    Parameters
    ----------
        @param model_name: model name
        @param basin: basin name
        @param deaths_delay: delay in deaths type
        @param run_name: name of the calibration run
        @param model_type: model type (only used if model_name is compartment or function)
    """
    # format
    command = f"python -m calibrate_{model_name}.py --basin {basin} --run_name {run_name} --deaths_delay {deaths_delay}"

    # run
    os.system(command)


def run_forecast(model_name, basin, run_name, tmax, model_type=None, forecast=1):
    """
    This function runs a forecast file
    Parameters
    ----------
        @param model_name: model name
        @param basin: basin name
        @param run_name: name of the calibration run
        @param tmax: maximum number of steps to use for the calibration
        @param model_type: model type (only used if model_name is compartment or function)
    """
    # format
    command = f"python -m forecast_{model_name}.py --basin {basin} --tmax {tmax} --run_name {run_name} --forecast {forecast}"
    if model_type != None:
        command += f" --model_type {model_type}"
    # run
    os.system(command)


def get_beta(R0, mu, Nk, C):
    """
    Compute the transmission rate beta for a SEIR model with age groups
    Parameters
    ----------
        @param R0: basic reproductive number
        @param mu: recovery rate
        @param Nk: number of individuals in different age groups
        @param C: contact matrix
    Return
    ------
        @return: the transmission rate beta
    """
    # get seasonality adjustment
    C_hat = np.zeros((C.shape[0], C.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C_hat[i, j] = (Nk[i] / Nk[j]) * C[i, j]
    return R0 * mu / (np.max([e.real for e in np.linalg.eig(C_hat)[0]]))


def calibration(epimodels : List[Callable], 
                priors : List[pyabc.Distribution], 
                params : List[dict], 
                distance : Callable,
                observations : List[float],
                model_name : str, 
                basin_name : str,
                transitions : List[pyabc.AggregatedTransition],
                dates : List[datetime],
                evaluation_dates : List[datetime],
                max_walltime : timedelta = None,
                population_size : int = 1000,
                minimum_epsilon : float = 0.3, 
                max_nr_populations : int = 10, 
                n_procs : int = 4, 
                filename : str = '', 
                folder_name : str = "calibration_runs"):

    """
    Run ABC calibration on given model and prior 
    Parameters
    ----------
        @param epimodel (Callable): epidemic model 
        @param prior (pyabc.Distribution): prior distribution
        @param params (dict): dictionary of fixed parameters value
        @param distance (Callable): distance function to use 
        @param observations (List[float]): real observations 
        @param model_name (str): model name
        @param basin_name (str): name of the basin
        @param transition (pyabc.AggregatedTransition): next gen. perturbation transitions
        @param dates (List[datetime]): list of simulation dates 
        @param evaluation_dates (List[datetime]): start/end dates to consider for evaluation
        @param max_walltime (timedelta): maximum simulation time
        @param population_size (int): size of the population of a given generation
        @param minimum_epsilon (float): minimum tolerance (if reached calibration stops)
        @param max_nr_population (int): maximum number of generations
        @param filename (str): name of the files used to store ABC results
    Returns
    -------
        @return: returns ABC history
    """


    def make_model(epimodel, param): 
        def model(p): 
            # run model 
            results = epimodel(**p, **param)
            # resample deaths weekly
            df_deaths = pd.DataFrame(data={"deaths": results['deaths']}, index=pd.to_datetime(dates))
            df_deaths = df_deaths.loc[(df_deaths.index >= evaluation_dates[0]) & (df_deaths.index <= evaluation_dates[1])]
            df_deaths = df_deaths.resample("W").sum()
            return {'data': df_deaths.deaths.values, 
                    "incidence": np.concatenate(([0], np.diff(results['I']) + np.diff(results['R'])))}
        return model

    if filename == '':
        filename = str(uuid.uuid4())

    abc = pyabc.ABCSMC([make_model(m, p) for m, p in zip(epimodels, params)], 
                       priors, 
                       distance, 
                       transitions=transitions, 
                       population_size=population_size,
                       sampler=MulticoreEvalParallelSampler(n_procs=n_procs))
    
    db_path = os.path.join(f'./{folder_name}/{basin_name}/{model_name}/dbs/', f"{filename}.db")
    abc.new("sqlite:///" + db_path, {"data": observations})
    history = abc.run(minimum_epsilon=minimum_epsilon, 
                      max_nr_populations=max_nr_populations,
                      max_walltime=max_walltime)
    
    with open(os.path.join(f'./{folder_name}/{basin_name}/{model_name}/abc_history/', f"{filename}.pkl"), 'wb') as file:
        pkl.dump(history, file)

    # save posterior distribution
    history.get_distribution()[0].to_csv(os.path.join(f"./{folder_name}/{basin_name}/{model_name}/posteriors/", f"{filename}_posterior_distributions.csv"))  

    # compute quantiles and save samples
    raw_samples = np.array([d["data"] for d in history.get_weighted_sum_stats()[1]])
    df_samples = compute_quantiles(raw_samples)
    df_dates = pd.DataFrame(index=pd.to_datetime(dates))
    df_dates = df_dates.loc[(df_dates.index >= evaluation_dates[0]) & (df_dates.index <= evaluation_dates[1])]
    df_samples["date"] = df_dates.resample("W").sum().index
    df_samples["data"] = history.observed_sum_stat()["data"]
    df_samples.to_csv(os.path.join(f"./{folder_name}/{basin_name}/{model_name}/posteriors/", f"{filename}_deaths_posterior_quantiles.csv"), index=False)  
    np.savez_compressed(os.path.join(f"./{folder_name}/{basin_name}/{model_name}/posteriors/", f"{filename}_deaths_posterior_samples.npz"), raw_samples)

    # compute incidence quantiles and save samples  
    raw_samples_incidence = np.array([d["incidence"] for d in history.get_weighted_sum_stats()[1]])
    df_samples_incidence = compute_quantiles(raw_samples_incidence)
    df_dates = pd.DataFrame(index=pd.to_datetime(dates))
    #df_dates = df_dates.loc[(df_dates.index >= evaluation_dates[0]) & (df_dates.index <= evaluation_dates[1])]
    df_samples_incidence["date"] = df_dates.index
    df_samples_incidence.to_csv(os.path.join(f"./{folder_name}/{basin_name}/{model_name}/posteriors/", f"{filename}_incidence_posterior_quantiles.csv"), index=False)  
    np.savez_compressed(os.path.join(f"./{folder_name}/{basin_name}/{model_name}/posteriors/", f"{filename}_incidence_posterior_samples.npz"), raw_samples_incidence)

    return history, abc


def calibration_single_model(epimodel : Callable, 
                            prior : pyabc.Distribution, 
                            params : dict, 
                            distance : Callable,
                            observations : List[float],
                            model_name : str, 
                            basin_name : str,
                            transition : pyabc.AggregatedTransition,
                            dates : List[datetime],
                            max_walltime : timedelta = None,
                            population_size : int = 1000,
                            minimum_epsilon : float = 0.3, 
                            max_nr_populations : int = 10, 
                            filename : str = ''):

    """
    Run ABC calibration on given model and prior 
    Parameters
    ----------
        @param epimodel (Callable): epidemic model 
        @param prior (pyabc.Distribution): prior distribution
        @param params (dict): dictionary of fixed parameters value
        @param distance (Callable): distance function to use 
        @param observations (List[float]): real observations 
        @param model_name (str): model name
        @param basin_name (str): name of the basin
        @param transition (pyabc.AggregatedTransition): next gen. perturbation transitions
        @param dates (List[datetime]): list of simulation dates 
        @param max_walltime (timedelta): maximum simulation time
        @param population_size (int): size of the population of a given generation
        @param minimum_epsilon (float): minimum tolerance (if reached calibration stops)
        @param max_nr_population (int): maximum number of generations
        @param filename (str): name of the files used to store ABC results
    Returns
    -------
        @return: returns ABC history
    """
    
    def model(p): 
        # run model 
        results = epimodel(**p, **params)
        # resample deaths weekly
        df_deaths = pd.DataFrame(data={"deaths": results['deaths']}, index=pd.to_datetime(dates))
        df_deaths = df_deaths.resample("W").sum()
        return {'data': df_deaths.deaths.values}

    if filename == '':
        filename = str(uuid.uuid4())

    abc = pyabc.ABCSMC(model, prior, distance, transitions=transition, population_size=population_size)
    db_path = os.path.join(f'./calibration_runs/{basin_name}/{model_name}/dbs/', f"{filename}.db")
    abc.new("sqlite:///" + db_path, {"data": observations})
    history = abc.run(minimum_epsilon=minimum_epsilon, 
                      max_nr_populations=max_nr_populations,
                      max_walltime=max_walltime)
    
    with open(os.path.join(f'./calibration_runs/{basin_name}/{model_name}/abc_history/', f"{filename}.pkl"), 'wb') as file:
        pkl.dump(history, file)
    return history


def compute_IFR(ages : List[float], ifr=IFR) -> float:
    """
    Compute effective IFR given the population pyramid
    Parameters
    ----------
        @param ages (List[float]): number of people in different age groups
        @param ifr (float): list of IFR by age groups
    Returns
    -------
        @return: returns effective IFR
    """
    return np.dot(ages, ifr) / np.sum(ages)


@jit(fastmath=True)
def compute_deaths(daily_recovered : List[float],
                   ifr : float, 
                   Delta : int) -> List[float]: 
    """
    Shift simulated deaths in time according to fixed delay 
    Parameters
    ----------
        @param daily_recovered (List[float]): number od daily recovered
        @param ifr (float): effective IFR
        @param Delta (int): delay in deaths
    Return
    ------
        @return: returns the list of daily deaths
    """
    # compute deaths
    daily_deaths = np.array([np.random.binomial(rec, ifr) for rec in daily_recovered])
    # shift forward deaths by Delta days
    daily_deaths_shifted = np.concatenate((np.zeros(Delta), daily_deaths[:-Delta]))
    return daily_deaths_shifted


def wmape_pyabc(sim_data : dict,
                actual_data : dict,) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE) to use for pyabc calibratiob
    Parameters
    ----------
        @param actual_data (dict): dictionary of actual data
        @param sim_data (dict): dictionary of simulated data 
    Return
    ------
        @return: returns wmape between actual and simulated data
    """
    return np.sum(np.abs(actual_data['data'] - sim_data['data'])) / np.sum(np.abs(actual_data['data']))
    

def wmape_pyabc_weekly(sim_data : dict, 
                       actual_data : dict) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE) to use for pyabc calibratiob
    Parameters
    ----------
        @param sim_data (dict): dictionary of simulated data 
        @param actual_data (dict): dictionary of actual data
    Return
    ------
        @return: returns wmape between actual and simulated data
    """
    sim_data_week, actual_data_week, i = [], [], 0
    while i + 7 < len(actual_data['data']):
        actual_data_week.append(np.sum(actual_data['data'][i:i + 7]))
        sim_data_week.append(np.sum(sim_data['data'][i:i + 7]))
        i += 7 
    sim_data_week, actual_data_week = np.array(sim_data_week), np.array(actual_data_week)
    return np.sum(np.abs(actual_data_week - sim_data_week)) / np.sum(np.abs(actual_data_week))
    

def import_projections(model_name, run_name, basin_name): 
    """
    This function imports the calibration data for a given model
    Parameters
    ----------
        @param model_name: model name
        @param run_name: run name
        @basin_name: name of the basin
    Return
    ------
        @return: returns pyabc calibration history 
    """
    with open(f'./calibration_runs/{basin_name}/{model_name}/abc_history/{run_name}.pkl', 'rb') as file: 
        data = pkl.load(file)
    return data


def import_parameters(model_name, run_name, basin_name): 
    """
    This function imports the parameters sampled during the calibration for a given model
    Parameters
    ----------
        @param model_name: model name
        @param run_name: run name
        @basin_name: name of the basin
    Return
    ------
        @return: returns sampled parameters
    """
    with open(f'./calibration_runs/{basin_name}/{model_name}/abc_history/{run_name}.pkl', 'rb') as file: 
        data = pkl.load(file)
    best_model_index = np.argmax(data.get_model_probabilities().iloc[-1])
    params = data.get_distribution(best_model_index)[0]
    return params


def get_median_CI(samples, 
                  levels=[0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 
                          0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 
                          0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 
                          0.975, 0.990]): 
    """
    This function returns the predictive median and quantiles simulated data
    Parameters
    ----------
        @param samples: stochastic samples
        @param levels: quantile levels
    Return
    ------
        @return: returns the predictive median and quantiles of the simulated data
    """

    quantiles = dict(median=np.quantile(samples, axis=0, q=0.5))
    for q in levels:
        quantiles[np.round(q, 3)] = np.quantile(samples, axis=0, q=q)
    return quantiles


def apply_seasonality(day: datetime, 
                      seasonality_min: float, 
                      basin_hemispheres: int, 
                      seasonality_max: float = 1) -> float:
    """
    Applies seasonality adjustment to a given day based on the specified parameters.

    Parameters
    - day (datetime): The specific day for which seasonality adjustment is applied.
    - seasonality_min (float): The minimum value of the seasonality adjustment.
    - basin_hemispheres (int): The indicator of the basin hemisphere (0 for northern, 1 for tropical, 2 for southern).
    - seasonality_max (float, optional): The maximum value of the seasonality adjustment. Defaults to 1.

    Returns:
    - float: The seasonality adjustment value for the specified day and basin hemisphere.

    Raises:
    - None
    """

    s_r = seasonality_min / seasonality_max
    day_max_north = datetime(day.year, 1, 15)
    day_max_south = datetime(day.year, 7, 15)

    seasonal_adjustment = np.empty(shape=(3,), dtype=np.float64)

    # north hemisphere
    seasonal_adjustment[0] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_north).days + 0.5 * np.pi) + 1 + s_r)

    # tropical hemisphere
    seasonal_adjustment[1] = 1.0

    # south hemisphere
    seasonal_adjustment[2] = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_south).days + 0.5 * np.pi) + 1 + s_r)

    return seasonal_adjustment[basin_hemispheres]


def get_gamma_params(mu, std):
    """
    Compute the parameters of a gamma distribution given its mean and standard deviation.
    """
    shape = (mu/std)**2
    scale = std**2 / mu
    return shape, scale


def get_simulation_dates(df):
    """
    Compute the simulation dates. Start date is the last day with 0 deaths, while the end date is the first minimum 
    after the peak.
    """
    imax = np.argmax(df.new_deaths)
    start_date = df.iloc[:imax].index[np.argwhere(df.iloc[:imax].new_deaths == np.min(df.iloc[:imax].new_deaths))[-1][0]]
    end_date = df.iloc[imax:].index[np.argmin(df.iloc[imax:].new_deaths)]
    return start_date, end_date


def check_folder(path: str, basin_name: str, folder_name : str = "calibration_runs") -> None:
    """
    Check if a folder exists, and create it if it doesn't exist.

    Parameters:
    - path (str): Path to the folder.
    - basin_name (str): Name of the basin

    Returns:
    - None
    """
    if not os.path.exists(os.path.join(path, f"{folder_name}/{basin_name}")):
        os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}"))
        for model in ["compartment_model", "mobility_model", "function_model"]:
            os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}/{model}"))
            os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}/{model}", "dbs"))
            os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}/{model}", "abc_history"))
            os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}/{model}", "posteriors"))
            os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}/{model}", "forecast"))


def check_folder_simulations(path: str, basin_name: str, folder_name : str = "simulations") -> None:
    """
    Check if a folder exists, and create it if it doesn't exist.

    Parameters:
    - path (str): Path to the folder.
    - basin_name (str): Name of the basin

    Returns:
    - None
    """
    if not os.path.exists(os.path.join(path, f"{folder_name}/{basin_name}")):
        os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}"))
        for model in ["compartment_model", "mobility_model", "function_model"]:
            os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}/{model}"))
            os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}/{model}", "samples"))
            os.system("mkdir " + os.path.join(path, f"{folder_name}/{basin_name}/{model}", "posteriors"))


def compute_quantiles(samples: np.ndarray, quantiles: np.ndarray = np.arange(0.01, 1.0, 0.01)) -> pd.DataFrame:
    """
    Compute quantiles and aggregated measures from the given samples.

    Parameters:
    - samples (np.ndarray): Array of samples.
    - quantiles (np.ndarray): Array of quantiles to compute. Default is np.arange(0.01, 1.0, 0.01).

    Returns:
    - pd.DataFrame: DataFrame containing the computed quantiles and aggregated measures.
    """
    df_samples = pd.DataFrame() 
    for q in quantiles:
        df_samples[str(np.round(q, 2))] = np.quantile(samples, axis=0, q=np.round(q, 2))
    
    # additional quantiles and aggregated measures
    df_samples["0.025"] = np.quantile(samples, axis=0, q=0.025)
    df_samples["0.975"] = np.quantile(samples, axis=0, q=0.975)
    df_samples["min"] = np.min(samples, axis=0)
    df_samples["max"] = np.max(samples, axis=0)

    return df_samples