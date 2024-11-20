# =============================================================================
# This script provides tests that go further than the ones done in the
# "Run_test_battery" script. Here the results for more than one model can be
# calculated at the same time.
# =============================================================================
##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import thorns as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as itl

##### import functions
import functions.stimulation as stim
import functions.create_plots_for_model_comparison as plot
import functions.model_tests as test
import functions.tests_for_analyses as aly
import functions.calculations as calc

##### import models
import models.Rattay_2001 as rattay_01
import models.Briaire_2005 as briaire_05
import models.Smit_2010 as smit_10

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### list of all models
models = ["rattay_01", "briaire_05", "smit_10"]

##### initialize clock
dt = 5*us

##### define way of processing
backend = "serial"

##### define if plots should be generated
generate_plots = False
theses_image_path = "" # add path here

##### define which tests to run
all_tests = True
latency_over_stim_amp_test = False
thresholds_for_pulse_trains_over_rate_and_dur = False
thresholds_for_pulse_trains_over_nof_pulses = False
thresholds_for_sinus = False
voltage_courses_comparison = False
computational_efficiency_test = False
pulse_train_refractory_test = False
stochastic_properties_test = False
somatic_delay_measurement = False

if any([all_tests, stochastic_properties_test]):
    # =============================================================================
    # Get thresholds for certain stimulation types and stimulus durations
    # =============================================================================
    ##### define phase durations (in us) and pulse forms to test
    phase_durations = [50]
    inter_phase_gap = [0]
    pulse_forms = ["mono"]
    
    ##### define varied parameters 
    params = [{"model_name" : model,
               "phase_duration" : phase_durations[jj]*1e-6,
               "inter_phase_gap" : inter_phase_gap[jj]*1e-6,
               "pulse_form" : pulse_forms[jj]}
                for model in models
                for jj in range(len(phase_durations))]
    
    ##### get thresholds
    threshold_table = th.util.map(func = test.get_threshold,
                                  space = params,
                                  backend = backend,
                                  cache = "no",
                                  kwargs = {"dt" : 5*us,
                                            "delta" : 0.001*uA,
                                            "stimulation_type" : "extern",
                                            "upper_border" : 800*uA,
                                            "add_noise" : False})
    
    ##### change index to column
    threshold_table.reset_index(inplace=True)
    
    ##### change column names
    threshold_table = threshold_table.rename(index = str, columns={"model_name" : "model",
                                                                   "phase_duration" : "phase duration (us)",
                                                                   "inter_phase_gap" : "inter phase gap (us)",
                                                                   "pulse_form" : "pulse form",
                                                                   0:"threshold"})
    
    ##### add unit to phase duration and inter phase gap
    threshold_table["phase duration (us)"] = [round(ii*1e6) for ii in threshold_table["phase duration (us)"]]
    threshold_table["inter phase gap (us)"] = [round(ii*1e6,1) for ii in threshold_table["inter phase gap (us)"]]
    
if all_tests or thresholds_for_pulse_trains_over_rate_and_dur:
    # =============================================================================
    # Measure thresholds for pulse trains with different pulse rates and train durations
    # =============================================================================
    ##### define stimulation parameters
    phase_durations = [25,45,100]*us
    inter_phase_gap = 8*us
    
    ##### use different phase durations
    for phase_duration in phase_durations:
        
        ##### measure threshold over pulse rate and over train duration
        for method in ["thr_over_rate", "thr_over_dur"]:
            
            ##### define pulse rates
            if method == "thr_over_rate":
                pulses_per_second = [250,500,1000,2000,3000,5000,10000]
            else:
                pulses_per_second = [200,3000,10000]
            
            ##### define pulse train durations in ms
            if method == "thr_over_rate":
                pulse_train_durations = [0.5,2,50]
            else:
                pulse_train_durations = [0.1,0.2,0.4,0.7,1,2,5,10,20]
            
            ##### define number of stochastic runs per pulse train type
            nof_runs_per_pulse_train = 1
            
            ##### initialize dataframe with all stimulation information
            stim_table = pd.DataFrame(list(itl.product(pulse_train_durations, pulses_per_second)), columns=["pulse train durations (ms)", "pulses per second"])
            
            ##### calculate number of pulses for each train duration - pulse rate combination
            stim_table["number of pulses"] = np.floor(stim_table["pulse train durations (ms)"]*1e-3 * stim_table["pulses per second"]).astype(int)
                
            ##### calculate inter_pulse_gap
            stim_table["inter pulse gap (us)"] = np.round(1e6/stim_table["pulses per second"] - phase_duration*2/us - inter_phase_gap/us).astype(int)
            
            ##### drop rows with no pulse
            stim_table = stim_table[stim_table["number of pulses"] != 0]
            
            ##### get thresholds
            params = [{"model_name" : model,
                       "nof_pulses" : stim_table["number of pulses"].iloc[ii],
                       "inter_pulse_gap" : stim_table["inter pulse gap (us)"].iloc[ii]*1e-6,
                       "run_number" : jj}
                        for model in models
                        for ii in range(len(stim_table))
                        for jj in range(nof_runs_per_pulse_train)]
            
            pulse_train_thresholds = th.util.map(func = test.get_threshold,
                                                 space = params,
                                                 backend = backend,
                                                 cache = "no",
                                                 kwargs = {"dt" : 1*us,
                                                           "phase_duration" : phase_duration,
                                                           "inter_phase_gap" : inter_phase_gap,
                                                           "delta" : 0.005*uA,
                                                           "upper_border" : 1000*uA,
                                                           "pulse_form" : "bi",
                                                           "add_noise" : False})
            
            ##### change index to column
            pulse_train_thresholds.reset_index(inplace=True)
            
            ##### change column names
            pulse_train_thresholds = pulse_train_thresholds.rename(index = str, columns={"model_name" : "model",
                                                                                         "nof_pulses" : "number of pulses",
                                                                                         "inter_pulse_gap" : "inter pulse gap (us)",
                                                                                         0:"threshold (uA)"})
            
            ##### convert threshold to uA and inter pulse gap to us
            pulse_train_thresholds["threshold (uA)"] = [ii*1e6 for ii in pulse_train_thresholds["threshold (uA)"]]
            pulse_train_thresholds["inter pulse gap (us)"] = [round(ii*1e6) for ii in pulse_train_thresholds["inter pulse gap (us)"]]
            
            ##### exclude spontaneous APs
            pulse_train_thresholds = pulse_train_thresholds[pulse_train_thresholds["threshold (uA)"] > 1e-9]
            
            ##### get mean thresholds
            pulse_train_thresholds = pulse_train_thresholds.groupby(["model","number of pulses", "inter pulse gap (us)"])["threshold (uA)"].mean().reset_index()
                
            ##### add information of stim_table
            pulse_train_thresholds = pd.merge(pulse_train_thresholds, stim_table, on=["number of pulses","inter pulse gap (us)"])
            
            ##### order dataframe
            pulse_train_thresholds = pulse_train_thresholds.sort_values(by=['model', 'pulses per second'])
            
            ##### save dataframe as csv    
            pulse_train_thresholds.to_csv("results/Analyses/pulse_train_{}_{}us.csv".format(method, phase_duration), index=False, header=True)
    
    ##### plot thresholds over pulse rate and pulse train duration      
    if generate_plots:
        
        ##### load dataframes
        pulse_train_thr_over_rate = pd.read_csv("results/Analyses/pulse_train_thr_over_rate_45us.csv")
        pulse_train_thr_over_dur = pd.read_csv("results/Analyses/pulse_train_thr_over_dur_45us.csv")
        
        ##### generate plot
        thresholds_for_pulse_trains_plot = plot.thresholds_for_pulse_trains(plot_name = "Thresholds for pulse trains",
                                                                            pulse_train_thr_over_rate = pulse_train_thr_over_rate,
                                                                            pulse_train_thr_over_dur = pulse_train_thr_over_dur)
    
        ##### save plot
        thresholds_for_pulse_trains_plot.savefig("{}/thresholds_for_pulse_trains.pdf".format(theses_image_path), bbox_inches='tight')
        
if all_tests or thresholds_for_pulse_trains_over_nof_pulses:
    # =============================================================================
    # Measure thresholds for pulse trains over number of pulses
    # =============================================================================
    ##### define stimulation parameters
    phase_duration = 15*us # 20*us
    inter_phase_gap = 2*us
    
    ##### define pulse train durations in ms
    pulse_train_durations = [0.1,0.2,0.5,1,2,5,10,20] #[0.1,0.3,1,3,10,30]
    
    ##### define pulse rates
    pulses_per_second = [1200,3000,5000,25000] #[1500,3000,5000,11000] 
    
    ##### define number of stochastic runs per pulse train tye
    nof_runs_per_pulse_train = 1
    
    ##### initialize dataframe with all stimulation information
    stim_table = pd.DataFrame(list(itl.product(pulse_train_durations, pulses_per_second)), columns=["pulse train durations (ms)", "pulses per second"])
    
    ##### calculate number of pulses for each train duration - pulse rate combination
    stim_table["number of pulses"] = np.floor(stim_table["pulse train durations (ms)"]*1e-3 * stim_table["pulses per second"]).astype(int)
        
    ##### calculate inter_pulse_gap
    stim_table["inter pulse gap (us)"] = np.round(1e6/stim_table["pulses per second"] - phase_duration*2/us - inter_phase_gap/us).astype(int)
    
    ##### drop rows with no pulse
    stim_table = stim_table[stim_table["number of pulses"] != 0]
    
    ##### get thresholds
    params = [{"model_name" : model,
               "nof_pulses" : stim_table["number of pulses"].iloc[ii],
               "inter_pulse_gap" : stim_table["inter pulse gap (us)"].iloc[ii]*1e-6,
               "run_number" : jj}
                for model in models
                for ii in range(len(stim_table))
                for jj in range(nof_runs_per_pulse_train)]
    
    pulse_train_thresholds = th.util.map(func = test.get_threshold,
                                         space = params,
                                         backend = backend,
                                         cache = "no",
                                         kwargs = {"dt" : 1*us,
                                                   "phase_duration" : phase_duration,
                                                   "inter_phase_gap" : inter_phase_gap,
                                                   "delta" : 0.005*uA,
                                                   "upper_border" : 1500*uA,
                                                   "pulse_form" : "bi",
                                                   "add_noise" : False})
    
    ##### change index to column
    pulse_train_thresholds.reset_index(inplace=True)
    
    ##### change column names
    pulse_train_thresholds = pulse_train_thresholds.rename(index = str, columns={"model_name" : "model",
                                                                                 "nof_pulses" : "number of pulses",
                                                                                 "inter_pulse_gap" : "inter pulse gap (us)",
                                                                                 0:"threshold (uA)"})
    
    ##### convert threshold to uA and inter pulse gap to us
    pulse_train_thresholds["threshold (uA)"] = [ii*1e6 for ii in pulse_train_thresholds["threshold (uA)"]]
    pulse_train_thresholds["inter pulse gap (us)"] = [round(ii*1e6) for ii in pulse_train_thresholds["inter pulse gap (us)"]]
    
    ##### exclude spontaneous APs
    pulse_train_thresholds = pulse_train_thresholds[pulse_train_thresholds["threshold (uA)"] > 1e-9]
    
    ##### get mean thresholds
    pulse_train_thresholds = pulse_train_thresholds.groupby(["model","number of pulses", "inter pulse gap (us)"])["threshold (uA)"].mean().reset_index()
        
    ##### add information of stim_table
    pulse_train_thresholds = pd.merge(pulse_train_thresholds, stim_table, on=["number of pulses","inter pulse gap (us)"])
    
    ##### order dataframe
    pulse_train_thresholds = pulse_train_thresholds.sort_values(by=['model', 'pulses per second'])
    
    ##### save dataframe as csv    
    pulse_train_thresholds.to_csv("results/Analyses/pulse_train_thresholds_15us.csv", index=False, header=True)
    
    ##### plot thresholds over pulse rate and pulse train duration      
    if generate_plots:
        
        ##### load dataframes
        pulse_train_thresholds = pd.read_csv("results/Analyses/pulse_train_thresholds_15us.csv")
        
        ##### plot thresholds over number of pulses
        thresholds_for_pulse_trains_plot = plot.thresholds_for_pulse_trains_over_nof_pulses(plot_name = "Thresholds for pulse trains",
                                                                                            threshold_data = pulse_train_thresholds)
        
        ##### save plot
        thresholds_for_pulse_trains_plot.savefig("results/Analyses/pulse_train_thresholds.pdf", bbox_inches='tight')

if all_tests or thresholds_for_sinus:
    # =============================================================================
    # Measure thresholds for sinusodial stimulation
    # =============================================================================
    ##### define frequencies (in kHz)
    frequencies = [0.125,0.25,0.5,1,2,4,8,16]
    
    ##### sinus durations (in ms)
    stim_lengths = [0.5,2,10]
    
    ##### get thresholds
    params = [{"model_name" : model,
               "frequency" : frequency*1e3,
               "stim_length" : stim_length*1e-3} 
                for model in models
                for frequency in frequencies
                for stim_length in stim_lengths]
    
    sinus_thresholds = th.util.map(func = test.get_threshold_for_sinus,
                                   space = params,
                                   backend = backend,
                                   cache = "no",
                                   kwargs = {"dt" : 1*us,
                                             "delta" : 0.005*uA,
                                             "upper_border" : 1500*uA,
                                             "add_noise" : False})
    
    ##### change index to column
    sinus_thresholds.reset_index(inplace=True)
    
    ##### change column names
    sinus_thresholds = sinus_thresholds.rename(index = str, columns={"model_name" : "model",
                                                                     "stim_length" : "stimulus length (ms)",
                                                                     0:"threshold (uA)"})
    
    ##### convert threshold to uA and stimulus length to ms and frequency to kHz
    sinus_thresholds["threshold (uA)"] = [ii*1e6 for ii in sinus_thresholds["threshold (uA)"]]
    sinus_thresholds["stimulus length (ms)"] = [ii*1e3 for ii in sinus_thresholds["stimulus length (ms)"]]
    sinus_thresholds["frequency"] = [ii*1e-3 for ii in sinus_thresholds["frequency"]]
    
    ##### exclude spontaneous APs
    sinus_thresholds = sinus_thresholds[sinus_thresholds["threshold (uA)"] > 1e-9]
    
    ##### save dataframe as csv    
    sinus_thresholds.to_csv("results/Analyses/sinus_thresholds.csv", index=False, header=True)
                
    ##### plot thresholds over pulse rate and pulse train duration      
    if generate_plots:
        
        ##### load dataframes
        sinus_thresholds = pd.read_csv("results/Analyses/sinus_thresholds.csv")
        
        ##### generate plot
        thresholds_for_sinus = plot.thresholds_for_sinus(plot_name = "Thresholds for pulse trains",
                                                         sinus_thresholds = sinus_thresholds)
    
        ##### save plot
        thresholds_for_sinus.savefig("{}/thresholds_for_sinus.pdf".format(theses_image_path), bbox_inches='tight')
        
if all_tests or latency_over_stim_amp_test:
    # =============================================================================
    # Measure latencies for different stimulus amplitudes and electrode distances
    # =============================================================================
    ##### define stimulus parameters
    phase_duration = 45*us
    inter_phase_gap = 2*us
    pulse_form = "bi"
    stim_node = 2
    
    ##### electrode distance ratios (to be multiplied with the models original electrode distance of 300*um)
    electrode_distances = [200*um,300*um,500*um,700*um,1000*um,1500*um,2000*um,3000*um,5000*um]

    electrode_distance_ratios = [electrode_distance/(300*um) for electrode_distance in electrode_distances]
    
    ##### stimulus amplitude levels (to be multiplied with threshold stimulus)
    stim_amp_levels = np.linspace(1,1.5,10, endpoint = False).tolist() + np.linspace(1.5,5,15).tolist()
    
    ##### get thresholds of models for different electrode distances
    params = [{"model_name" : model,
               "parameter_ratio" : electrode_distance_ratio}
                for model in models\
                for electrode_distance_ratio in electrode_distance_ratios]
    
    thresholds = th.util.map(func = test.get_threshold,
                             space = params,
                             backend = backend,
                             cache = "no",
                             kwargs = {"dt" : 1*us,
                                       "parameter": "electrode_distance",
                                       "phase_duration" : phase_duration,
                                       "inter_phase_gap" : inter_phase_gap,
                                       "delta" : 0.0001*uA,
                                       "pulse_form" : pulse_form,
                                       "upper_border" : 400*mA})
    
    ##### change index to column
    thresholds.reset_index(inplace=True)
    
    ##### change column names
    thresholds = thresholds.rename(index = str, columns={"model_name" : "model",
                                                         "parameter_ratio" : "parameter ratio",
                                                         0:"threshold (uA)"})
    
    ##### calculate electrode distance in um
    thresholds["electrode distance (um)"] = [np.round(ratio*300).astype(int) for ratio in thresholds["parameter ratio"]]
    
    ##### get latencies for all stimulus amplitudes and electrode distances
    params = [{"model_name" : model,
               "stim_amp" : thresholds["threshold (uA)"][thresholds["model"] == model][thresholds["parameter ratio"] == electrode_distance_ratio][0] * stim_amp_level,
               "electrode_distance" : thresholds["electrode distance (um)"][thresholds["model"] == model][thresholds["parameter ratio"] == electrode_distance_ratio][0]*1e-6}
                for model in models\
                for electrode_distance_ratio in electrode_distance_ratios\
                for stim_amp_level in stim_amp_levels]
    
    latency_table = th.util.map(func = aly.get_latency,
                                space = params,
                                backend = backend,
                                cache = "no",
                                kwargs = {"dt" : 1*us,
                                          "phase_duration" : 45*us,
                                          "inter_phase_gap" : 2*us,
                                          "stimulus_node" : stim_node,
                                          "measurement_node" : 100,
                                          "time_after" : 10*ms,
                                          "pulse_form" : "bi"})
    
    ##### change index to column
    latency_table.reset_index(inplace=True)
    
    ##### change column names
    latency_table = latency_table.rename(index = str, columns={"model_name" : "model",
                                                               "stim_amp" : "stimulus amplitude (uA)",
                                                               "electrode_distance": "electrode distance (um)",
                                                               0:"latency (ms)"})
    
    ##### exclude rows where no AP was elicited
    latency_table = latency_table[latency_table["latency (ms)"] != 0]
    
    ##### convert electrode distances to um
    latency_table["electrode distance (um)"] = [np.round(distance*1e6).astype(int) for distance in latency_table["electrode distance (um)"]]
    
    ##### add amplitude level (factor by which threshold is multiplied)
    latency_table["amplitude level"] = latency_table["stimulus amplitude (uA)"] / \
                                                    [thresholds["threshold (uA)"][thresholds["model"] == latency_table["model"][ii]]\
                                                    [thresholds["electrode distance (um)"] == latency_table["electrode distance (um)"][ii]][0]\
                                                    for ii in range(len(latency_table))]
    
    ##### convert latency values to ms and stimulus amplitude to uA
    latency_table["latency (ms)"] = [ii*1e3 for ii in latency_table["latency (ms)"]]
    latency_table["stimulus amplitude (uA)"] = [ii*1e6 for ii in latency_table["stimulus amplitude (uA)"]]
    
    ##### order dataframe
    latency_table = latency_table.sort_values(by=['model', 'electrode distance (um)'])
    
    ##### Save dataframe as csv    
    latency_table.to_csv("results/Analyses/latencies_over_stim_dur.csv", index=False, header=True)
    
    ##### get experimental data
    latency_measurements = pd.read_csv("Measurements/Latency_data/latency_measurements.csv")
    
    ##### add stimulus amplitude levels to latency_measurements
    latency_measurements["amplitude level"] = latency_measurements["stimulus amplitude (uA)"] / latency_measurements["threshold"]
    
    #latency_table = pd.read_csv("results/Analyses/latencies_over_stim_dur_var_dist.csv")
    
    ##### plot latencies over stimulus amplitudes
    latencies_over_stimulus_duration_plot = plot.latencies_over_stimulus_duration(plot_name = "Latencies over stimulus durations",
                                                                                  latency_models = latency_table,
                                                                                  latency_measurements = latency_measurements)
    
    if generate_plots:
        ##### save plot
        latencies_over_stimulus_duration_plot.savefig("results/Analyses/latencies_over_stim_dur.pdf", bbox_inches='tight')

if all_tests or computational_efficiency_test:
    # =============================================================================
    # Get computational efficiencies
    # =============================================================================
    ##### stimulus duration
    stimulus_duration = 50*ms
    
    ##### define runs per model
    nof_runs = 10
    
    ##### get computation times
    computation_times = aly.computational_efficiency_test(model_names = models,
                                                           dt = 1*us,
                                                           stimulus_duration = stimulus_duration,
                                                           nof_runs = nof_runs)
    
    ##### save dataframe to csv
    computation_times.to_csv("results/Analyses/computational_efficiency.csv", index=False, header=True)

if all_tests or pulse_train_refractory_test:
    # =============================================================================
    # Get refractory periods for pulse trains
    # =============================================================================
    ##### define pulse rate of masker and second stimulus (in pulses per second)
    pulse_rate = 1200/second
    
    ##### define phase durations and inter_pulse_gap
    t_phase = 23*us
    t_ipg = 2*us
    
    ##### define pulse train duration
    t_pulse_train = 100*ms
    
    ##### calculate number of pulses
    nof_pulses = int(t_pulse_train * pulse_rate)
    
    ##### calculate inter pulse gap
    inter_pulse_gap = t_pulse_train/nof_pulses - 2*t_phase - t_ipg
    
    ##### define pulse rates
    pulse_rates = [1200,1500,18000,25000]
    
    ##### define varied parameters
    params = {"model_name" : models,
              "pulse_rate" : pulse_rates}
    
    ##### get thresholds
    refractory_table = th.util.map(func = aly.get_refractory_periods_for_pulse_trains,
                                   space = params,
                                   backend = backend,
                                   cache = "no",
                                   kwargs = {"dt" : 1*us,
                                             "delta" : 1*us,
                                             "stimulation_type" : "extern",
                                             "pulse_form" : "bi",
                                             "phase_durations" : [t_phase/us,t_ipg/us,t_phase/us]*us,
                                             "pulse_train_duration" : t_pulse_train})
    
    ##### change index to column
    refractory_table.reset_index(inplace=True)
    
    ##### change column names
    refractory_table = refractory_table.rename(index = str, columns={"model_name" : "model name",
                                                                     "pulse_rate" : "pulse rate",
                                                                     0 : "absolute refractory period (us)",
                                                                     1 : "relative refractory period (ms)"})
    
    ##### convert refractory periods to ms
    refractory_table["absolute refractory period (us)"] = refractory_table["absolute refractory period (us)"]*1e6
    refractory_table["relative refractory period (ms)"] = refractory_table["relative refractory period (ms)"]*1e3
    
    ##### round columns to 4 significant digits
    for ii in ["absolute refractory period (us)","relative refractory period (ms)"]:
        refractory_table[ii] = ["%.4g" %refractory_table[ii][jj] for jj in range(refractory_table.shape[0])]
    
    ##### Save dataframe as csv    
    refractory_table.to_csv("results/Analyses/refractory_table_pulse_trains.csv", index=False, header=True)

if all_tests or stochastic_properties_test:
    
    # =============================================================================
    # Get relative spread for different k_noise values
    # =============================================================================
    ##### define k_noise values to test
    k_noise_factor = np.append(np.round(np.arange(0,1,0.1),1), np.arange(1,5,0.25)).tolist()
    
    ##### define test parameters
    phase_duration = 50*us
    pulse_form = "mono"
    runs_per_k_noise = 1000
    
    ##### define varied parameters
    params = {"model_name" : models,
              "parameter_ratio": k_noise_factor,
              "run_number" : [ii for ii in range(runs_per_k_noise)]}
    
    ##### get stochastic thresholds
    relative_spreads = th.util.map(func = test.get_threshold,
                                   space = params,
                                   backend = backend,
                                   cache = "no",
                                   kwargs = {"dt" : dt,
                                             "parameter": "k_noise",
                                             "phase_duration" : phase_duration,
                                             "delta" : 0.0005*uA,
                                             "pulse_form" : pulse_form,
                                             "stimulation_type" : "extern",
                                             "upper_border" : 800*uA,
                                             "time_before" : 2*ms,
                                             "time_after" : 2*ms,
                                             "add_noise" : True})
    
    ##### change index to column
    relative_spreads.reset_index(inplace=True)
    
    ##### change column names
    relative_spreads = relative_spreads.rename(index = str, columns={"model_name" : "model",
                                                                     "parameter_ratio" : "knoise ratio",
                                                                     0:"threshold"})
    
    ##### exclude spontaneous APs
    relative_spreads = relative_spreads[relative_spreads["threshold"] > 1e-9]
    
    ##### delete run_number column
    relative_spreads = relative_spreads.drop(columns = ["run_number"])
    
    ##### calculate relative spread values
    thresholds = relative_spreads.groupby(["model", "knoise ratio"])
    relative_spreads = round(thresholds.std()/thresholds.mean()*100, 2)
    relative_spreads.reset_index(inplace=True)
    relative_spreads = relative_spreads.rename(index = str, columns={"threshold" : "relative spread (%)"})
    
    ##### Save relative spread dataframe as csv    
    relative_spreads.to_csv("results/Analyses/relative_spreads_k_noise_comparison.csv", index=False, header=True)
    
    # =============================================================================
    # Get jitter for different k_noise values
    # =============================================================================
    ##### define k_noise values to test
    k_noise_factor = np.append(np.round(np.arange(0,1,0.1),1), np.arange(1,5,0.25)).tolist()
    
    ##### define test parameters
    phase_duration = 50*us
    pulse_form = "mono"
    runs_per_k_noise = 1000
    
    ##### look up deterministic thresholds
    thresholds = threshold_table[threshold_table["phase duration (us)"] == phase_duration/us]
    thresholds = thresholds[threshold_table["pulse form"] == pulse_form][["model","threshold"]]
    
    ##### define varied parameters 
    params = [{"model_name" : model,
               "stim_amp" : threshold_table.set_index("model").transpose()[model]["threshold"],
               "parameter_ratio" : k_noise_factor[ii],
               "run_number" : jj}
                for model in models
                for ii in range(len(k_noise_factor))
                for jj in range(runs_per_k_noise)]
    
    ##### get single node response properties
    single_node_response_table = th.util.map(func = test.get_single_node_response,
                                             space = params,
                                             backend = backend,
                                             cache = "no",
                                             kwargs = {"dt" : 1*us,
                                                       "parameter": "k_noise",
                                                       "phase_duration" : phase_duration,
                                                       "pulse_form" : pulse_form,
                                                       "stimulation_type" : "extern",
                                                       "time_before" : 3*ms,
                                                       "time_after" : 2*ms,
                                                       "add_noise" : True})
    
    ##### change index to column
    single_node_response_table.reset_index(inplace=True)
    
    ##### change column names
    single_node_response_table = single_node_response_table.rename(index = str, columns={"model_name" : "model",
                                                                                         "run_number" : "run",
                                                                                         "parameter_ratio" : "knoise ratio"})
    
    ##### exclude data, where no action potential was elicited 
    single_node_response_table = single_node_response_table[single_node_response_table["AP height (mV)"] > 0.06]
    
    ##### exclude latencies smaller than zero
    single_node_response_table = single_node_response_table[single_node_response_table["latency (us)"] > 0]
    
    ##### build subset of relevant columns
    single_node_response_table = single_node_response_table[["model","knoise ratio","run","latency (us)"]]
    
    ##### change units from second to us
    single_node_response_table["latency (us)"] = single_node_response_table["latency (us)"]*1e6
    
    ##### calculate jitter
    single_node_response_table = single_node_response_table.groupby(["model","knoise ratio"])["latency (us)"].std().reset_index()
    single_node_response_table = single_node_response_table.rename(index = str, columns={"latency (us)" : "jitter (us)"})
    
    ##### Save jitter dataframe as csv    
    single_node_response_table.to_csv("results/Analyses/single_node_response_table_k_noise_comparison.csv", index=False, header=True)
    
    # =============================================================================
    # Plot relative spread over jitter for different models and k_noise values
    # =============================================================================
    relative_spreads = pd.read_csv("results/Analyses/relative_spreads_k_noise_comparison.csv")
    single_node_response_table = pd.read_csv("results/Analyses/single_node_response_table_k_noise_comparison.csv")
    
    
    ##### Combine relative spread and jitter information and exclude rows with na values
    stochasticity_table = pd.merge(relative_spreads, single_node_response_table, on=["model","knoise ratio"]).dropna()
    
    ##### Exclude relative spreads bigger than 30% and jitters bigger than 200 us
    stochasticity_table = stochasticity_table[(stochasticity_table["relative spread (%)"] < 30) & (stochasticity_table["jitter (us)"] < 200)]
    
    ##### plot table
    stochasticity_plot = plot.stochastic_properties_comparison(plot_name = "Comparison of stochastic properties",
                                                               stochasticity_table = stochasticity_table)
    
    ##### save plot
    stochasticity_plot.savefig("{}/stochasticity_plot.pdf".format(theses_image_path), bbox_inches='tight')

if all_tests or voltage_courses_comparison:
    # =============================================================================
    # Plot voltage course for all models
    # =============================================================================
    stim_amps = [0.2, 1.5, 0.3]
    max_node = [7,15,15]
    max_comp = [0,0,0]
    
    ##### initialize list to save voltage courses
    voltage_courses =  [ [] for i in range(len(models)) ]
    
    for ii, model_name in enumerate(models):
        
        ##### get model
        model = eval(model_name)
        
        ##### just save voltage values for a certain compartment range
        max_comp[ii] = np.where(model.structure == 2)[0][max_node[ii]]
        
        ##### set up the neuron
        neuron, model = model.set_up_model(dt = dt, model = model)
        
        ##### record the membrane voltage
        M = StateMonitor(neuron, 'v', record=True)
        
        ##### save initialization of the monitor(s)
        store('initialized')
    
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stimulus_current(model = model,
                                                    dt = dt,
                                                    pulse_form = "mono",
                                                    stimulation_type = "intern",
                                                    time_before = 0.2*ms,
                                                    time_after = 1*ms,
                                                    stimulated_compartment = np.where(model.structure == 2)[0][1],
                                                    ##### monophasic stimulation
                                                    amp_mono = stim_amps[ii]*nA,
                                                    duration_mono = 100*us)
        
        ##### get TimedArray of stimulus currents
        stimulus = TimedArray(np.transpose(I_stim), dt = dt)
                
        ##### run simulation
        run(runtime)
        
        ##### save M.v in voltage_courses
        voltage_courses[ii] = M.v[:max_comp[ii],:]
    
    ##### Plot membrane potential of all compartments over time
    voltage_course_comparison = plot.voltage_course_comparison_plot(plot_name = "Voltage courses all models",
                                                                    model_names = models,
                                                                    time_vector = M.t,
                                                                    max_comp = max_comp,
                                                                    voltage_courses = voltage_courses)
    
    if generate_plots:
        ##### save plot
#        voltage_course_comparison.savefig("results/Analyses/voltage_course_comparison_plot {}.png", bbox_inches='tight')
        voltage_course_comparison.savefig("{}/voltage_course_comparison_plot.pdf".format(theses_image_path), bbox_inches='tight')

if somatic_delay_measurement:
    # =============================================================================
    # Measure the somatic delay of models with a soma
    # =============================================================================
    stim_amps = [0.2, 1.5, 0.3]
    
    ##### initialize list to save voltage courses
    somatic_delays =  [ 0*second for i in range(len(models)) ]
    
    for ii, model_name in enumerate(models):
        
        ##### get model
        model = eval(model_name)
        
        ##### set up the neuron
        neuron, model = model.set_up_model(dt = dt, model = model)
        
        ##### record the membrane voltage
        M = StateMonitor(neuron, 'v', record=True)
        
        ##### save initialization of the monitor(s)
        store('initialized')
    
        ##### define how the ANF is stimulated
        I_stim, runtime = stim.get_stimulus_current(model = model,
                                                    dt = dt,
                                                    pulse_form = "mono",
                                                    stimulation_type = "intern",
                                                    time_before = 0.2*ms,
                                                    time_after = 1*ms,
                                                    stimulated_compartment = np.where(model.structure == 2)[0][1],
                                                    ##### monophasic stimulation
                                                    amp_mono = stim_amps[ii]*nA,
                                                    duration_mono = 100*us)
        
        ##### get TimedArray of stimulus currents
        stimulus = TimedArray(np.transpose(I_stim), dt = dt)
                
        ##### run simulation
        run(runtime)
        
        ##### get indexes of the compartments before and after the soma
        idx_before_soma = np.where(model.structure == 3)[0][0]-2
        idx_after_soma = np.where(model.structure == 4)[0][-1]+1
        
        ##### AP amplitude
        AP_amp_before_soma = max(M.v[idx_before_soma,:]-model.V_res)
        AP_amp_after_soma = max(M.v[idx_after_soma,:]-model.V_res)
        
        ##### AP time
        AP_time_before_soma = M.t[M.v[idx_before_soma,:]-model.V_res == AP_amp_before_soma]
        AP_time_after_soma = M.t[M.v[idx_after_soma,:]-model.V_res == AP_amp_after_soma]
        
        ##### save somatic delay
        somatic_delays[ii] = AP_time_after_soma - AP_time_before_soma
