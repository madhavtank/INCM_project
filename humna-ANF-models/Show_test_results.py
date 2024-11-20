# =============================================================================
# This script collects the test battery results of one (!) model and generates
# and saves plots which show these data. Furthermore tables are generated, that
# compare the results of this single model to experimental data.
# =============================================================================
##### don't show warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##### import packages
from brian2 import *
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import os

##### import models
import models.Rattay_2001 as rattay_01
import models.Briaire_2005 as briaire_05
import models.Smit_2010 as smit_10

##### import functions
import functions.create_plots as plot

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### choose model
model_name = "smit_10"
model = eval(model_name)

##### save plots
save_plots = False
save_plots_for_report = False
interim_report_path = "" # add path here

# =============================================================================
# Load data
# =============================================================================
##### load dataframes for tables
strength_duration_data = pd.read_csv("results/{}/Strength_duration_data_cathodic {}.csv".format(model.display_name,model.display_name))
strength_duration_data_ano = pd.read_csv("results/{}/Strength_duration_data_anodic {}.csv".format(model.display_name,model.display_name))
threshold_table = pd.read_csv("results/{}/Threshold_table {}.csv".format(model.display_name,model.display_name))
relative_spreads = pd.read_csv("results/{}/Relative_spreads {}.csv".format(model.display_name,model.display_name))
conduction_velocity_table = pd.read_csv("results/{}/Conduction_velocity_table {}.csv".format(model.display_name,model.display_name))
single_node_response_deterministic  = pd.read_csv("results/{}/Single_node_response_deterministic {}.csv".format(model.display_name,model.display_name))
single_node_response_stochastic = pd.read_csv("results/{}/Single_node_response_stochastic {}.csv".format(model.display_name,model.display_name))
refractory_table = pd.read_csv("results/{}/Refractory_table {}.csv".format(model.display_name,model.display_name))

##### load dataframes for plots
strength_duration_plot_table = pd.read_csv("results/{}/Strength_duration_plot_table_cathodic {}.csv".format(model.display_name,model.display_name))
strength_duration_plot_table_ano = pd.read_csv("results/{}/Strength_duration_plot_table_anodic {}.csv".format(model.display_name,model.display_name))
relative_spread_plot_table = pd.read_csv("results/{}/Relative_spread_plot_table {}.csv".format(model.display_name,model.display_name))
voltage_course_dataset_det = pd.read_csv("results/{}/Single_node_response_plot_data_deterministic {}.csv".format(model.display_name,model.display_name))
voltage_course_dataset_stoch = pd.read_csv("results/{}/Single_node_response_plot_data_stochastic {}.csv".format(model.display_name,model.display_name))
refractory_curve_table = pd.read_csv("results/{}/Refractory_curve_table {}.csv".format(model.display_name,model.display_name))
psth_table = pd.read_csv("results/{}/PSTH_table {}.csv".format(model.display_name,model.display_name))

# =============================================================================
# Generate plots and save them
# =============================================================================
##### strength duration curve (cathodic)
strength_duration_curve = plot.strength_duration_curve(plot_name = "Strength duration curve cathodic {}".format(model.display_name),
                                                       threshold_data = strength_duration_plot_table,
                                                       rheobase = strength_duration_data["rheobase (uA)"].iloc[0]*uA,
                                                       chronaxie = strength_duration_data["chronaxie (us)"].iloc[0]*us)

##### strength duration curve (anodic)
strength_duration_curve_ano = plot.strength_duration_curve(plot_name = "Strength duration curve anodic {}".format(model.display_name),
                                                           threshold_data = strength_duration_plot_table_ano,
                                                           rheobase = strength_duration_data_ano["rheobase (uA)"].iloc[0]*uA,
                                                           chronaxie = strength_duration_data_ano["chronaxie (us)"].iloc[0]*us)

##### relative spreads plot
relative_spread_plot = plot.relative_spread(plot_name = "Relative spreads {}".format(model.display_name),
                                            threshold_data = relative_spread_plot_table)

##### single node response plot
single_node_response = plot.single_node_response_voltage_course(plot_name = "Voltage courses {}".format(model.display_name),
                                                                voltage_data = voltage_course_dataset_stoch)

##### refractory curve
refractory_curve = plot.refractory_curve(plot_name = "Refractory curve {}".format(model.display_name),
                                         refractory_table = refractory_curve_table)

##### poststimulus time histogram plot
post_stimulus_time_histogram = plot.post_stimulus_time_histogram(plot_name = "PSTH {}".format(model.display_name),
                                                                 psth_dataset = psth_table.copy(),
                                                                 plot_style = "pulses_per_timebin")

if save_plots:
    strength_duration_curve.savefig("results/{}/Strength_duration_curve {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    relative_spread_plot.savefig("results/{}/Relative_spreads_plot {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    single_node_response.savefig("results/{}/Single_node_response {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    refractory_curve.savefig("results/{}/Refractory_curve {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    post_stimulus_time_histogram.savefig("results/{}/PSTH {}.png".format(model.display_name,model.display_name), bbox_inches='tight')

if save_plots_for_report:
    strength_duration_curve.savefig("results/{}/Strength_duration_curve {}.png".format(interim_report_path,model.display_name), bbox_inches='tight')
    relative_spread_plot.savefig("results/{}/Relative_spreads_plot {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    single_node_response.savefig("results/{}/Single_node_response {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    refractory_curve.savefig("results/{}/Refractory_curve {}.png".format(model.display_name,model.display_name), bbox_inches='tight')
    post_stimulus_time_histogram.savefig("results/{}/PSTH {}.png".format(model.display_name,model.display_name), bbox_inches='tight')

# =============================================================================
# Add experimental results to tables
# =============================================================================
##### Strength duration data
strength_duration_data = strength_duration_data.transpose()
strength_duration_data = strength_duration_data.rename(index = str, columns={0:"model"})
strength_duration_data["model"] = ["%.3g" %strength_duration_data["model"][i] for i in range(0,2)]
strength_duration_data["Van den Honert and Stypulkowski 1984"] = ["95.8", "247"]

##### Relative spread of thresholds
relative_spreads = relative_spreads.rename(index = str, columns={"relative spread":"model"})
relative_spreads["experiments"] = ["6.3%","5-10%","12%","11%"]
relative_spreads["reference"] = ["Miller et al. 1999","Dynes 1996","Javel et al. 1987","Javel et al. 1987"]
relative_spreads = relative_spreads.set_index(["phase duration (us)","pulse form"])

##### Conduction velocity
conduction_velocity_table = conduction_velocity_table.transpose()
conduction_velocity_table = conduction_velocity_table.rename(index = str, columns={0:"model"})
if hasattr(model, "index_soma"):
    conduction_velocity_table["Hursh 1939"] = ["-","-","6","-","-","-"]
    if model.dendrite_outer_diameter < 12*um:
        dendrite_ratio = 4.6
    else:
        dendrite_ratio = 5.66
    conduction_velocity_table["Boyd and Kalu 1979"] = ["-","-","{}".format(dendrite_ratio),"-","-","-"]
    conduction_velocity_table["Czèh et al 1976"] = ["-","-","-","v_ax = 0.9*v_den-6.9*m/s","-","-"]

else:
    conduction_velocity_table["Hursh 1939"] = ["-","-","6"]
    if model.fiber_outer_diameter < 12*um:
        dendrite_ratio = 4.6
    else:
        dendrite_ratio = 5.66
    conduction_velocity_table["Boyd and Kalu 1979"] = ["-","-","{}".format(dendrite_ratio)]

##### Latency and jitter
latency = single_node_response_deterministic[["phase duration (us)", "pulse form", "amplitude level", "latency (us)"]]
jitter = single_node_response_stochastic[["phase duration (us)", "pulse form", "amplitude level", "jitter (us)"]]
latency_jitter = pd.merge(single_node_response_deterministic, single_node_response_stochastic, on=["phase duration (us)","pulse form","amplitude level"], how="left")

latency_jitter = pd.melt(latency_jitter, id_vars=["phase duration (us)", "pulse form", "amplitude level"], value_vars=["latency (us)",  "jitter (us)"])
latency_jitter["value"] = ["%.3g" %latency_jitter["value"][i] for i in range(0,latency_jitter.shape[0])]
latency_jitter["phase duration (us)"] = ["%g us" %latency_jitter["phase duration (us)"][i] for i in range(0,latency_jitter.shape[0])]
latency_jitter = latency_jitter.rename(index = str, columns={"phase duration (us)":"phase duration"})

latency_jitter_th = latency_jitter[latency_jitter["amplitude level"] == "1*threshold"]
latency_jitter_th = latency_jitter_th.rename(index = str, columns={"value":"model (threshold)"})
latency_jitter_th = latency_jitter_th.drop(columns = ["amplitude level"])

latency_jitter_2th = latency_jitter[latency_jitter["amplitude level"] == "2*threshold"]
latency_jitter_2th = latency_jitter_2th.rename(index = str, columns={"value":"model (2*threshold)"})
latency_jitter_2th = latency_jitter_2th.drop(columns = ["amplitude level"])

latency_jitter = latency_jitter.drop(columns = ["amplitude level", "value"]).drop_duplicates()

latency_jitter = pd.merge(latency_jitter, latency_jitter_th, on=["phase duration","pulse form","variable"], how = 'left')
latency_jitter = pd.merge(latency_jitter, latency_jitter_2th, on=["phase duration","pulse form","variable"], how = 'left')

latency_jitter = latency_jitter.sort_values(by=["pulse form", "phase duration"], ascending = [False, True])

latency_jitter["Miller et al. 1999"] = ["650", "100", "-", "-", "-", "-", "-", "-"]
latency_jitter["Van den Honert and Stypulkowski 1984 (threshold)"] = ["-", "-", "-", "-", "685", "352", "-", "-"]
latency_jitter["Van den Honert and Stypulkowski 1984 (2*threshold)"] = ["-", "-", "-", "-", "352", "8", "-", "-"]
latency_jitter["Hartmann and al. 1984"] = ["-", "-", "-", "-", "-", "-", "300-400", "-"]
latency_jitter["Cartee et al. 2000 (threshold)"] = ["-", "-", "440", "80", "-", "-", "-", "-"]

latency_jitter = latency_jitter.rename(index = str, columns={"variable":"property"})
latency_jitter = latency_jitter.set_index(["phase duration","pulse form", "property"])

##### AP shape
AP_shape = single_node_response_deterministic[["AP height (mV)", "rise time (us)", "fall time (us)", "AP duration (us)"]].iloc[[5]].transpose()
AP_shape = AP_shape.rename(index = str, columns={5:"model"})

t_rise = round(-0.000625*conduction_velocity_table["model"].iloc[[0]] + 0.14, 3).tolist()[0]
t_fall = round(-0.002083*conduction_velocity_table["model"].iloc[[0]] + 0.3933, 3).tolist()[0]
AP_duration = t_rise + t_fall
AP_shape["Paintal 1966"] = ["-", int(t_rise*1e3), int(t_fall*1e3), int(AP_duration*1e3)]

##### Refractory periods
absolute_refractory_periods = refractory_table.drop(columns = ["relative refractory period (ms)"])
absolute_refractory_periods = absolute_refractory_periods.rename(index = str, columns={"absolute refractory period (us)":"ARP model (us)"})
absolute_refractory_periods["ARP Experiments (us)"] = ["334","300","500-700","400-500","-"]
absolute_refractory_periods["reference"] = ["Miller et al. 2001","Stypulkowski and Van den Honert 1984","Dynes 1996","Brown and Abbas 1990", "-"]
absolute_refractory_periods = absolute_refractory_periods[absolute_refractory_periods["ARP Experiments (us)"] != "-"]
absolute_refractory_periods = absolute_refractory_periods.set_index(["phase duration (us)","pulse form"])

relative_refractory_periods = refractory_table.drop(columns = ["absolute refractory period (us)"])
relative_refractory_periods = relative_refractory_periods.rename(index = str, columns={"relative refractory period (ms)":"RRP model (ms)"})
relative_refractory_periods["RRP Experiments (ms)"] = ["-","3-4; 4-5","5","-","5"]
relative_refractory_periods["reference"] = ["-","Stypulkowski and Van den Honert 1984; Cartee et al. 2000","Dynes 1996","-", "Hartmann et al. 1984"]
relative_refractory_periods = relative_refractory_periods[relative_refractory_periods["RRP Experiments (ms)"] != "-"]
relative_refractory_periods = relative_refractory_periods.set_index(["phase duration (us)","pulse form"])
