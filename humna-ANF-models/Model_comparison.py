# =============================================================================
# This script collects the test battery results of all (!) models and generates
# and saves plots that compare the results among each other and with experimental
# data. Furthermore dataframes are generated and saved in a latex-compatibel
# format, which contain both model and experimental data.
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
import itertools as itl
from string import ascii_uppercase as letters

##### import models
import models.Rattay_2001 as rattay_01
import models.Briaire_2005 as briaire_05
import models.Smit_2010 as smit_10

##### import functions
import functions.create_plots_for_paper as plot
import functions.pandas_to_latex as ptol
import functions.stimulation as stim

##### makes code faster and prevents warning
prefs.codegen.target = "numpy"

# =============================================================================
# Initializations
# =============================================================================
##### list of all models
models = [rattay_01, briaire_05, smit_10]

##### save plots
save_plots = True
save_tables = True
paper_image_path = "" # add path here
paper_table_path = "" # add path here

# =============================================================================
# Conduction velocity tables
# =============================================================================
##### table for models with soma
for ii,model in enumerate(models):
    
    ##### get strength duration data
    data = pd.read_csv("results/{}/Conduction_velocity_table {}.csv".format(model.display_name,model.display_name)).transpose()
    
    ##### round for three significant digits
    data[0] = ["%.3g" %data[0][jj] for jj in range(data.shape[0])]
    
    if ii == 0:
        ##### use model name as column header
        conduction_velocity_table = data.rename(index = str, columns={0:model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        conduction_velocity_table[model.display_name_plots] = data[0]

# =============================================================================
# Conduction velocity plot
# =============================================================================
##### built subset with relevant columns of datasetb for dendritic part
cond_vel_dendrite = conduction_velocity_table.transpose()[["velocity dendrite (m/s)","outer diameter dendrite (um)"]]
cond_vel_dendrite["section"] = "dendrite"
cond_vel_dendrite = cond_vel_dendrite.rename(index = str, columns={"velocity dendrite (m/s)":"velocity (m/s)",
                                                 "outer diameter dendrite (um)":"outer diameter (um)"})

##### built subset with relevant columns of datase for axonal part
cond_vel_axon = conduction_velocity_table.transpose()[["velocity axon (m/s)","outer diameter axon (um)"]]
cond_vel_axon["section"] = "axon"
cond_vel_axon = cond_vel_axon.rename(index = str, columns={"velocity axon (m/s)":"velocity (m/s)",
                                                           "outer diameter axon (um)":"outer diameter (um)"})

##### connect dataframes
cond_vel = pd.concat((cond_vel_dendrite, cond_vel_axon), axis=0)

##### index to column
cond_vel.reset_index(inplace=True)
cond_vel = cond_vel.rename(index = str, columns={"index":"model_name"})

##### add short name of models to table
cond_vel["short_name"] = ""
cond_vel["short_name"][cond_vel["model_name"] == "Briaire and Frijns (2005)"] = "BF"
cond_vel["short_name"][cond_vel["model_name"] == "Rattay et al. (2001)"] = "RA"
cond_vel["short_name"][cond_vel["model_name"] == "Smit et al. (2010)"] = "SH"

##### order dataframe
cond_vel = cond_vel.sort_values("model_name")

##### Plot conduction velocity comparison
cond_vel_plot = plot.conduction_velocity_comparison(plot_name = "Comparison of conduction velocities with experimental data",
                                                    velocity_data = cond_vel)

##### save plots
if save_plots:
    cond_vel_plot.savefig("{}/conduction_velocity_plot.pdf".format(paper_image_path), bbox_inches='tight')

# =============================================================================
# Generate conduction velocity table in an appropriate format for latex
# =============================================================================
##### transpose table
cond_vel_table = conduction_velocity_table.transpose()

##### order columns
cond_vel_table = cond_vel_table[["velocity dendrite (m/s)", "outer diameter dendrite (um)","velocity/diameter dendrite",
                                                       "velocity axon (m/s)","outer diameter axon (um)","velocity/diameter axon"]]

##### change column names again
cond_vel_table = cond_vel_table.rename(index = str, columns={"velocity dendrite (m/s)":"$v_{\T{c}}$/$ms^{-1}$",
                                                                                   "outer diameter dendrite (um)":"$D$/\SI{}{\micro\meter}",
                                                                                   "velocity/diameter dendrite":"$k$",
                                                                                   "velocity axon (m/s)":"$v_{\T{c}}$/$ms^{-1}$",
                                                                                   "outer diameter axon (um)":"$D$/\SI{}{\micro\meter}",
                                                                                   "velocity/diameter axon":"$k$"})
    
##### fill NA values with ""
conduction_velocity_table = conduction_velocity_table.fillna("-")

##### define captions and save tables as tex
if save_tables:
    caption_top = "Comparison of conduction velocities, outer diameters and scaling factors predicted by the ANF models."    
    with open("{}/cond_vel_table.tex".format(paper_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(cond_vel_table, label = "tbl:con_vel_table",
                                         caption_top = caption_top, vert_line = [2], upper_col_names = ["dendrite","axon"]))

# =============================================================================
# Single node response plot
# =============================================================================
##### initialize list of dataframes to save voltage courses
voltage_courses = [pd.DataFrame()]*len(models)

##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"

##### loop over models
for ii,model in enumerate(models):
    
    ##### get voltage course of model
    voltage_course_dataset = pd.read_csv("results/{}/Single_node_response_plot_data_deterministic {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    voltage_course_dataset["model"] = model.display_name_plots
    
    ##### write subset of dataframe in voltage courses list
    voltage_courses[ii] = voltage_course_dataset[["model", "membrane potential (mV)","time (ms)", "amplitude level"]]\
                                                [voltage_course_dataset["pulse form"] == pulse_form]\
                                                [voltage_course_dataset["phase duration (us)"] == phase_duration/us]

##### connect dataframes to one dataframe
voltage_courses = pd.concat(voltage_courses,ignore_index = True)

##### add short name of models to table
voltage_courses["short_name"] = ""
voltage_courses["short_name"][voltage_courses["model"] == "Briaire and Frijns (2005)"] = "BF"
voltage_courses["short_name"][voltage_courses["model"] == "Rattay et al. (2001)"] = "RA"
voltage_courses["short_name"][voltage_courses["model"] == "Smit et al. (2010)"] = "SH"

##### plot voltage courses
single_node_response = plot.single_node_response_comparison(plot_name = "Voltage courses model comparison",
                                                            voltage_data = voltage_courses)

##### save plot
if save_plots:
    single_node_response.savefig("{}/single_node_response_comparison.pdf".format(paper_image_path), bbox_inches='tight')

# =============================================================================
# AP shape table axons
# =============================================================================
##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"
amplitude_level = "2*threshold"

##### loop over models
for ii,model in enumerate(models):
    
    ##### get node response summery table
    data  = pd.read_csv("results/{}/Single_node_response_deterministic {}.csv".format(model.display_name,model.display_name))
    
    ##### built subset of relevant rows and columns and transpose dataframe
    data = data[["AP height (mV)", "rise time (us)", "fall time (us)"]]\
               [data["pulse form"] == pulse_form]\
               [data["phase duration (us)"] == phase_duration/us]\
               [data["amplitude level"] == amplitude_level].transpose()
    
    if ii == 0:
        ##### use model name as column header
        AP_shape_axon = data.rename(index = str, columns={data.columns.values[0]:model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        AP_shape_axon[model.display_name_plots] = data[data.columns.values[0]]
    
##### transpose dataframe
AP_shape_axon = AP_shape_axon.transpose()

##### round columns to 3 significant digits
for ii in ["AP height (mV)","rise time (us)","fall time (us)"]:
    AP_shape_axon[ii] = ["%.4g" %AP_shape_axon[ii][jj] for jj in range(AP_shape_axon.shape[0])]

##### change column names for latex export
AP_shape_latex = AP_shape_axon.rename(index = str, columns={"AP height (mV)":"AP height/mV",
                                                 "rise time (us)":"rise time/\SI{}{\micro\second}",
                                                 "fall time (us)":"fall time/\SI{}{\micro\second}"})

##### define caption and save table as tex
if save_tables:
    caption_top = "Comparison of AP shapes, measured with the axons of the ANF models for a stimulation with a monophasic \SI{100}{\micro\second}\
                   cathodic current pulse with amplitude $2I_{\T{th}}$."
    with open("{}/AP_shape_models.tex".format(paper_table_path), "w") as tf:
         tf.write(ptol.dataframe_to_latex(AP_shape_latex, label = "tbl:AP_shape_comparison", caption_top = caption_top))

# =============================================================================
# AP shapes dendrite
# =============================================================================
##### define which data to show
phase_duration = 100*us
pulse_form = "monophasic"
amplitude_level = "2*threshold"

##### loop over models with a soma
for ii,model in enumerate(models):
    
    ##### get node response summery table
    data  = pd.read_csv("results/{}/Single_node_response_deterministic dendrite {}.csv".format(model.display_name,model.display_name))
    
    ##### built subset of relevant rows and columns and transpose dataframe
    data = data[["AP height (mV)", "rise time (us)", "fall time (us)"]]\
               [data["pulse form"] == pulse_form]\
               [data["phase duration (us)"] == phase_duration/us]\
               [data["amplitude level"] == amplitude_level].transpose()
    
    if ii == 0:
        ##### use model name as column header
        AP_shape_dendrite = data.rename(index = str, columns={data.columns.values[0]:"{} dendrite".format(model.display_name_plots)})
        
    else:
        ##### add column with AP shape data of current model
        AP_shape_dendrite["{} dendrite".format(model.display_name_plots)] = data[data.columns.values[0]]
    
##### transpose dataframe
AP_shape_dendrite = AP_shape_dendrite.transpose()

##### round columns to 3 significant digits
for ii in ["AP height (mV)","rise time (us)","fall time (us)"]:
    AP_shape_dendrite[ii] = ["%.4g" %AP_shape_dendrite[ii][jj] for jj in range(AP_shape_dendrite.shape[0])]

# =============================================================================
# Plot experimental results for rise and fall time
# =============================================================================
##### add axon label in row names for human ANF models
AP_shape_axon = AP_shape_axon.rename(index={"Rattay et al. (2001)":"Rattay et al. (2001) axon",
                                             "Briaire and Frijns (2005)":"Briaire and Frijns (2005) axon",
                                             "Smit et al. (2010)":"Smit et al. (2010) axon"})

##### connect conduction velocities with AP shape values
AP_shape_cond_vel_table = pd.concat([AP_shape_axon[["rise time (us)","fall time (us)"]], AP_shape_dendrite[["rise time (us)","fall time (us)"]]])
AP_shape_cond_vel_table["conduction velocity axon (m/s)"] = 0.0
AP_shape_cond_vel_table["conduction velocity dendrite (m/s)"] = 0.0
AP_shape_cond_vel_table["section"] = ""

##### Fill in conduction velocities
for ii,model in enumerate(models):
    
    #### write velocity in table
    AP_shape_cond_vel_table["conduction velocity dendrite (m/s)"]["{} dendrite".format(model.display_name_plots)] = conduction_velocity_table[model.display_name_plots]["velocity dendrite (m/s)"]
    AP_shape_cond_vel_table["conduction velocity axon (m/s)"]["{} axon".format(model.display_name_plots)] = conduction_velocity_table[model.display_name_plots]["velocity axon (m/s)"]
    #### write section in table
    AP_shape_cond_vel_table["section"]["{} dendrite".format(model.display_name_plots)] = "dendrite"
    AP_shape_cond_vel_table["section"]["{} axon".format(model.display_name_plots)] = "axon"
        
##### change index to column
AP_shape_cond_vel_table.reset_index(inplace=True)
AP_shape_cond_vel_table = AP_shape_cond_vel_table.rename(index = str, columns={"index" : "model_name"})

##### change rise time column type to float
AP_shape_cond_vel_table["rise time (us)"] = AP_shape_cond_vel_table["rise time (us)"].astype(float)
AP_shape_cond_vel_table["fall time (us)"] = AP_shape_cond_vel_table["fall time (us)"].astype(float)

##### Plot rise and fall time comparison
rise_and_fall_time_comparison_paintal = plot.rise_and_fall_time_comparison(plot_name = "Comparison of rise times with data from Paintal 1966",
                                                                           model_data = AP_shape_cond_vel_table)

##### save plots
if save_plots:
    rise_and_fall_time_comparison_paintal.savefig("{}/rise_and_fall_time_comparison_paintal.pdf".format(paper_image_path), bbox_inches='tight')

# =============================================================================
# Latency table
# =============================================================================
##### define which data to show
phase_durations = [40, 50, 50, 100]
amplitude_level = ["1*threshold", "1*threshold", "2*threshold", "1*threshold"]
pulse_forms = ["monophasic", "monophasic", "monophasic", "monophasic"]

##### create dataframe, that defines which data to show
stimulations = pd.DataFrame([phase_durations,amplitude_level,pulse_forms]).transpose()
stimulations = stimulations.rename(index = str, columns={0:"phase duration (us)",
                                                         1:"amplitude level",
                                                         2:"pulse form"})

##### loop over models
for ii,model in enumerate(models):
    
    ##### get node response summery table
    data = pd.read_csv("results/{}/Single_node_response_deterministic {}.csv".format(model.display_name,model.display_name))
    
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","amplitude level", "pulse form"])["latency (us)"].astype(int))
    
    if ii == 0:
        ##### use model name as column header
        latency_table = data.rename(index = str, columns={"latency (us)":model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        latency_table[model.display_name_plots] = data["latency (us)"].tolist()

##### Add experimental data
latency_table["\cite{Cartee2000}"] = ["440", "-", "-", "-"]
latency_table["\cite{VandenHonert1984}"] = ["-", "685", "352", "-"]
latency_table["\cite{Miller1999}"] = ["-", "-", "-", "650"]

##### change column names / model names
latency_table = latency_table.rename(index = str, columns={"Rattay et al. (2001)":"Rattay model",
                                                         "Briaire and Frijns (2005)":"Briaire-Frijns model",
                                                         "Smit et al. (2010)":"Smit-Hanekom model"})

##### Transpose dataframe
latency_table = latency_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(latency_table.columns)]):
    latency_table = latency_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### define caption and save table as tex
if save_tables:
    caption_top = "Action potential latency of ANF models measured with four different stimuli (Unit: \SI{}{\micro\second}). Latency values from exicting feline studies are also included (italicized)."
    caption_bottom = ""
    for ii,letter in enumerate(letters[:len(stimulations)]):
        if stimulations["amplitude level"][ii][0] == "1": stim_amp = ""
        else: stim_amp = stimulations["amplitude level"][ii][0]
        caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                         + "}{\micro\second}" + " cathodic current pulse with amplitude {}".format(stim_amp) + "$I_{\T{th}}$\\\\\n"
    italic_range = range(len(models),len(latency_table))
    with open("{}/latency_table.tex".format(paper_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(latency_table, label = "tbl:latency_comparison",
                                         caption_top = caption_top, caption_bottom = caption_bottom, italic = italic_range))

# =============================================================================
# Strength duration table
# =============================================================================
##### cathodic stimulus
for ii,model in enumerate(models):
    
    # get strength duration data
    data = pd.read_csv("results/{}/Strength_duration_data_cathodic {}.csv".format(model.display_name,model.display_name)).transpose()
    
    if ii == 0:
        # use model name as column header
        strength_duration_table_cat = data.rename(index = str, columns={0:model.display_name_plots})
        
    else:
        # add column with AP shape data of current model
        strength_duration_table_cat[model.display_name_plots] = data[0]

# round for three significant digits
for ii in strength_duration_table_cat.columns.values.tolist():
    strength_duration_table_cat[ii] = ["%.3g" %strength_duration_table_cat[ii][jj] for jj in range(strength_duration_table_cat.shape[0])]

# rename indices
strength_duration_table_cat = strength_duration_table_cat.rename(index={"rheobase (uA)":"rheobase cat",
                                                                        "chronaxie (us)":"chronaxie cat"})

##### anodic stimulus
for ii,model in enumerate(models):
    
    # get strength duration data
    data = pd.read_csv("results/{}/Strength_duration_data_anodic {}.csv".format(model.display_name,model.display_name)).transpose()
    
    if ii == 0:
        # use model name as column header
        strength_duration_table_ano = data.rename(index = str, columns={0:model.display_name_plots})
        
    else:
        # add column with AP shape data of current model
        strength_duration_table_ano[model.display_name_plots] = data[0]

# round for three significant digits
for ii in strength_duration_table_ano.columns.values.tolist():
    strength_duration_table_ano[ii] = ["%.3g" %strength_duration_table_ano[ii][jj] for jj in range(strength_duration_table_ano.shape[0])]

# rename indices
strength_duration_table_ano = strength_duration_table_ano.rename(index={"rheobase (uA)":"rheobase ano",
                                                                        "chronaxie (us)":"chronaxie ano"})

##### connect dataframes
strength_duration_table = pd.concat([strength_duration_table_cat,strength_duration_table_ano])

##### change column names / model names
strength_duration_table = strength_duration_table.rename(index = str, columns={"Rattay et al. (2001)":"Rattay model",
                                                                               "Briaire and Frijns (2005)":"Briaire-Frijns model",
                                                                               "Smit et al. (2010)":"Smit-Hanekom model"})

##### transpose dataframe
strength_duration_table = strength_duration_table.transpose()

##### change order of columns
strength_duration_table = strength_duration_table[["rheobase cat", "rheobase ano", "chronaxie cat", "chronaxie ano"]]

##### change column names again
strength_duration_table = strength_duration_table.rename(index = str, columns={"rheobase cat":"cathodic",
                                                                               "rheobase ano":"anodic",
                                                                               "chronaxie cat":"cathodic",
                                                                               "chronaxie ano":"anodic"})

#### define caption and save table as tex
if save_tables:
    caption_top = "Rheobase $I_{\T{rh}}$ and chronaxie $\tau_{\T{chr}}$ of ANF models for monophasic cathodic and anodic stimulations."
    with open("{}/strength_duration_table.tex".format(paper_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(strength_duration_table, label = "tbl:strength_duration_comparison",
                                         caption_top = caption_top, vert_line = [1], upper_col_names = ["$I_{\T{rh}}$/\SI{}{\micro\ampere}","$\tau_{\T{chr}}$/\SI{}{\micro\second}"]))

# =============================================================================
# Strength duration curve
# =============================================================================
##### initialize list of dataframes to save strength duration curves
stength_duration_curves_cat = [pd.DataFrame()]*len(models)
stength_duration_curves_ano = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model in enumerate(models):
    
    ##### read strength duration curves
    stength_duration_curves_cat[ii] = pd.read_csv("results/{}/Strength_duration_plot_table_cathodic {}.csv".format(model.display_name,model.display_name))
    stength_duration_curves_ano[ii] = pd.read_csv("results/{}/Strength_duration_plot_table_anodic {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    stength_duration_curves_cat[ii]["model"] = model.display_name_plots
    stength_duration_curves_ano[ii]["model"] = model.display_name_plots

##### connect list of dataframes to one dataframe
stength_duration_curves_cat = pd.concat(stength_duration_curves_cat,ignore_index = True)
stength_duration_curves_ano = pd.concat(stength_duration_curves_ano,ignore_index = True)

##### add short name of models to tables
stength_duration_curves_cat["short_name"] = ""
stength_duration_curves_cat["short_name"][stength_duration_curves_cat["model"] == "Briaire and Frijns (2005)"] = "BF"
stength_duration_curves_cat["short_name"][stength_duration_curves_cat["model"] == "Rattay et al. (2001)"] = "RA"
stength_duration_curves_cat["short_name"][stength_duration_curves_cat["model"] == "Smit et al. (2010)"] = "SH"

stength_duration_curves_ano["short_name"] = ""
stength_duration_curves_ano["short_name"][stength_duration_curves_ano["model"] == "Briaire and Frijns (2005)"] = "BF"
stength_duration_curves_ano["short_name"][stength_duration_curves_ano["model"] == "Rattay et al. (2001)"] = "RA"
stength_duration_curves_ano["short_name"][stength_duration_curves_ano["model"] == "Smit et al. (2010)"] = "SH"

##### plot strength duration curve
strength_duration_curve = plot.strength_duration_curve_comparison(plot_name = "Strength duration curve model comparison",
                                                                  threshold_data_cat = stength_duration_curves_cat,
                                                                  threshold_data_ano = stength_duration_curves_ano)

##### save plot
if save_plots:
    strength_duration_curve.savefig("{}/strength_duration_curve_comparison.pdf".format(paper_image_path), bbox_inches='tight')

# =============================================================================
# Refractory curves
# =============================================================================
##### initialize list of dataframes to save voltage courses
refractory_curves = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model in enumerate(models):
    
    ##### get voltage course of model
    refractory_curves[ii] = pd.read_csv("results/{}/Refractory_curve_table {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    refractory_curves[ii]["model"] = model.display_name_plots

##### connect dataframes to one dataframe
refractory_curves = pd.concat(refractory_curves,ignore_index = True)

##### remove rows where no second spikes were obtained
refractory_curves = refractory_curves[refractory_curves["minimum required amplitude"] != 0]
    
##### calculate the ratio of the threshold of the second spike and the masker
refractory_curves["threshold ratio"] = refractory_curves["minimum required amplitude"]/refractory_curves["threshold"]

##### convert interpulse intervals to ms
refractory_curves["interpulse interval"] = refractory_curves["interpulse interval"]*1e3

##### add short name of models to tables
refractory_curves["short_name"] = ""
refractory_curves["short_name"][refractory_curves["model"] == "Briaire and Frijns (2005)"] = "BF"
refractory_curves["short_name"][refractory_curves["model"] == "Rattay et al. (2001)"] = "RA"
refractory_curves["short_name"][refractory_curves["model"] == "Smit et al. (2010)"] = "SH"

##### plot voltage courses
refractory_curves_plot = plot.refractory_curves_comparison(plot_name = "Refractory curves model comparison",
                                                           refractory_curves = refractory_curves)

##### save plot
if save_plots:
    refractory_curves_plot.savefig("{}/refractory_curves_plot_comparison.pdf".format(paper_image_path), bbox_inches='tight')

# =============================================================================
# Absolute refractory table model comparison
# =============================================================================
##### define which data to show
phase_durations = [40, 50, 100, 50]
pulse_forms = ["monophasic", "monophasic", "monophasic", "biphasic"]

##### create dataframe, that defines which data to show
stimulations = pd.DataFrame([phase_durations,pulse_forms]).transpose()
stimulations = stimulations.rename(index = str, columns={0:"phase duration (us)",
                                                         1:"pulse form"})

##### loop over models
for ii,model in enumerate(models):
    
    ##### get data
    data = pd.read_csv("results/{}/Refractory_table {}.csv".format(model.display_name,model.display_name))
        
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","pulse form"])["absolute refractory period (us)"].astype(int))

    if ii == 0:
        ##### use model name as column header
        ARP_comparison_table = data.rename(index = str, columns={"absolute refractory period (us)":model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        ARP_comparison_table[model.display_name_plots] = data["absolute refractory period (us)"].tolist()

##### round for four significant digits
for ii in ARP_comparison_table.columns.values.tolist():
    ARP_comparison_table[ii] = ["%.4g" %ARP_comparison_table[ii][jj] for jj in range(ARP_comparison_table.shape[0])]
        
##### Add experimental data
ARP_comparison_table["\cite{Miller2001}"] = ["334", "-", "-", "-"]
ARP_comparison_table["\cite{Stypulkowski1984}"] = ["-", "300", "-", "-"]
ARP_comparison_table["\cite{Dynes1996}"] = ["-", "-", "500-700", "-"]
ARP_comparison_table["\cite{Brown1990}"] = ["-", "-", "-", "500"]

##### change column names / model names
ARP_comparison_table = ARP_comparison_table.rename(index = str, columns={"Rattay et al. (2001)":"Rattay model",
                                                                         "Briaire and Frijns (2005)":"Briaire-Frijns model",
                                                                         "Smit et al. (2010)":"Smit-Hanekom model"})

##### Transpose dataframe
ARP_comparison_table = ARP_comparison_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(ARP_comparison_table.columns)]):
    ARP_comparison_table = ARP_comparison_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### define caption and save table as tex
if save_tables:
    caption_top = "Absolute refractory period of ANF models measured with four stimuli (Unit: \SI{}{\micro\second}). Measurements from existing feline studies are also included (italicized)."
    caption_bottom = ""
    for ii,letter in enumerate(letters[:len(stimulations)]):
        if stimulations["pulse form"][ii] == "monophasic":
            caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                                            + "}{\micro\second} cathodic current pulses\\\\\n"
        else:
            caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                                + "}{\micro\second} cathodic first current pulses\\\\\n"
    italic_range = range(len(models),len(ARP_comparison_table))
    with open("{}/ARP_comparison_table.tex".format(paper_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(ARP_comparison_table, label = "tbl:ARP_comparison",
                                         caption_top = caption_top, caption_bottom = caption_bottom, italic = italic_range))

# =============================================================================
# Relative refractory table model comparison
# =============================================================================
##### define which data to show
phase_durations = [50, 100, 200]
pulse_forms = ["monophasic", "monophasic", "biphasic"]

##### create dataframe, that defines which data to show
stimulations = pd.DataFrame([phase_durations,pulse_forms]).transpose()
stimulations = stimulations.rename(index = str, columns={0:"phase duration (us)",
                                                         1:"pulse form"})

##### loop over models
for ii,model in enumerate(models):
    
    ##### get data
    data = pd.read_csv("results/{}/Refractory_table {}.csv".format(model.display_name,model.display_name))
        
    ##### just observe data, with the parameters of the stimulation dataframe
    data = pd.DataFrame(pd.merge(stimulations, data, on=["phase duration (us)","pulse form"])["relative refractory period (ms)"])

    if ii == 0:
        ##### use model name as column header
        RRP_comparison_table = data.rename(index = str, columns={"relative refractory period (ms)":model.display_name_plots})
        
    else:
        ##### add column with AP shape data of current model
        RRP_comparison_table[model.display_name_plots] = data["relative refractory period (ms)"].tolist()

##### round for three significant digits
for ii in RRP_comparison_table.columns.values.tolist():
    RRP_comparison_table[ii] = ["%.3g" %RRP_comparison_table[ii][jj] for jj in range(RRP_comparison_table.shape[0])]
    
##### Add experimental data
RRP_comparison_table["\cite{Stypulkowski1984}"] = ["3-4", "-", "-"]
RRP_comparison_table["\cite{Cartee2000}"] = ["4-5", "-", "-"]
RRP_comparison_table["\cite{Dynes1996}"] = ["-", "5", "-"]
RRP_comparison_table["\cite{Hartmann1984a}"] = ["-", "-", "5"]

##### change column names / model names
RRP_comparison_table = RRP_comparison_table.rename(index = str, columns={"Rattay et al. (2001)":"Rattay model",
                                                                         "Briaire and Frijns (2005)":"Briaire-Frijns model",
                                                                         "Smit et al. (2010)":"Smit-Hanekom model"})

##### Transpose dataframe
RRP_comparison_table = RRP_comparison_table.transpose()

##### Change column names
for ii,letter in enumerate(letters[:len(RRP_comparison_table.columns)]):
    RRP_comparison_table = RRP_comparison_table.rename(index = str, columns={"{}".format(ii):"Stim. {}".format(letter)})

##### define caption and save table as tex
if save_tables:
    caption_top = "Relative refractory period of ANF models measured with four stimuli (Unit: \SI{}{\milli\second}). Measurements from existing feline studies are also included (italicized)."
    caption_bottom = ""
    for ii,letter in enumerate(letters[:len(stimulations)]):
        if stimulations["pulse form"][ii] == "monophasic":
            caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                                            + "}{\micro\second} cathodic current pulses\\\\\n"
        else:
            caption_bottom = caption_bottom + "{}: {} ".format(letter,stimulations["pulse form"][ii]) + "\SI{" + "{}".format(stimulations["phase duration (us)"][ii])\
                                + "}{\micro\second} cathodic first current pulses\\\\\n"
    italic_range = range(len(models),len(RRP_comparison_table))
    with open("{}/RRP_comparison_table.tex".format(paper_table_path), "w") as tf:
        tf.write(ptol.dataframe_to_latex(RRP_comparison_table, label = "tbl:RRP_comparison",
                                         caption_top = caption_top, caption_bottom = caption_bottom, italic = italic_range))

# =============================================================================
# PSTHs
# =============================================================================
##### initialize list of dataframes to save psth data for each model
psth_data = [pd.DataFrame()]*len(models)

##### loop over models
for ii,model in enumerate(models):
    
    ##### get psth data of model
    psth_data[ii] = pd.read_csv("results/{}/PSTH_table_new {}.csv".format(model.display_name,model.display_name))
    
    #### add model information
    psth_data[ii]["model"] = model.display_name_plots

##### connect dataframes to one dataframe
psth_data = pd.concat(psth_data,ignore_index = True)

##### convert spike times to ms
psth_data["spike times (us)"] = np.ceil(list(psth_data["spike times (us)"]*1000)).astype(int)
psth_data = psth_data.rename(index = str, columns={"spike times (us)" : "spike times (ms)"})

##### add short name of models to tables
psth_data["short_name"] = ""
psth_data["short_name"][psth_data["model"] == "Briaire and Frijns (2005)"] = "BF"
psth_data["short_name"][psth_data["model"] == "Rattay et al. (2001)"] = "RA"
psth_data["short_name"][psth_data["model"] == "Smit et al. (2010)"] = "SH"

##### plot PSTH comparison
psth_plot = plot.psth_comparison(plot_name = "PSTH model comparison3",
                                 psth_data = psth_data.copy(),
                                 amplitudes = ['1.5*threshold'],
                                 pulse_rates = [400, 800, 2000, 5000],
                                 plot_style = "spikes_per_time_bin")

##### save plot
if save_plots:
    psth_plot.savefig("{}/psth_plot_comparison_thr.pdf".format(paper_image_path), bbox_inches='tight')

# =============================================================================
# Plot voltage course for all models
# =============================================================================
stim_amps = [0.06, 2.1, 0.1]
max_node = [7,15,16]
max_comp = [0,0,0]

##### initialize list to save voltage courses
voltage_courses =  [ [] for i in range(len(models)) ]

for ii, model in enumerate(models):
    
    ##### just save voltage values for a certain compartment range
    max_comp[ii] = np.where(model.structure == 2)[0][max_node[ii]]
    
    ##### set up the neuron
    neuron, model = model.set_up_model(dt = 5*us, model = model)
    
    ##### record the membrane voltage
    M = StateMonitor(neuron, 'v', record=True)
    
    ##### save initialization of the monitor(s)
    store('initialized')

    ##### define how the ANF is stimulated
    I_stim, runtime = stim.get_stimulus_current(model = model,
                                                dt = 5*us,
                                                pulse_form = "mono",
                                                stimulation_type = "intern",
                                                time_before = 0.1*ms,
                                                time_after = 1.5*ms,
                                                stimulated_compartment = 0, #np.where(model.structure == 2)[0][1],
                                                ##### monophasic stimulation
                                                amp_mono = stim_amps[ii]*nA,
                                                duration_mono = 100*us)
    
    ##### get TimedArray of stimulus currents
    stimulus = TimedArray(np.transpose(I_stim), dt = 5*us)
            
    ##### run simulation
    run(runtime)
    
    ##### save M.v in voltage_courses
    voltage_courses[ii] = M.v[:max_comp[ii],:]

##### Plot membrane potential of all compartments over time
voltage_course_comparison = plot.voltage_course_comparison_plot(plot_name = "Voltage courses all models",
                                                                model_names = ["rattay_01", "briaire_05", "smit_10"],
                                                                time_vector = M.t,
                                                                max_comp = max_comp,
                                                                voltage_courses = voltage_courses)

if save_plots:
    ##### save plot
    voltage_course_comparison.savefig("{}/voltage_course_comparison_plot.pdf".format(paper_image_path), bbox_inches='tight')

# =============================================================================
# Plot relative spread over jitter for different models and k_noise values
# =============================================================================
relative_spreads = pd.read_csv("results/Analyses/relative_spreads_k_noise_comparison.csv")
single_node_response_table = pd.read_csv("results/Analyses/single_node_response_table_k_noise_comparison.csv")

##### Combine relative spread and jitter information and exclude rows with na values
stochasticity_table = pd.merge(relative_spreads, single_node_response_table, on=["model","knoise ratio"]).dropna()

##### Exclude relative spreads bigger than 30% and jitters bigger than 200 us
stochasticity_table = stochasticity_table[(stochasticity_table["relative spread (%)"] < 30) & (stochasticity_table["jitter (us)"] < 200)]

##### confine dataset to human ANF models
stochasticity_table = stochasticity_table[stochasticity_table["model"].isin(["rattay_01", "briaire_05", "smit_10"])]

##### add short name of models to tables
stochasticity_table["short_name"] = ""
stochasticity_table["short_name"][stochasticity_table["model"] == "briaire_05"] = "BF"
stochasticity_table["short_name"][stochasticity_table["model"] == "rattay_01"] = "RA"
stochasticity_table["short_name"][stochasticity_table["model"] == "smit_10"] = "SH"

##### plot table
stochasticity_plot = plot.stochastic_properties_comparison(plot_name = "Comparison of stochastic properties",
                                                           stochasticity_table = stochasticity_table)

##### save plot
stochasticity_plot.savefig("{}/stochasticity_plot.pdf".format(paper_image_path), bbox_inches='tight')

# =============================================================================
# Thresholds for pulse train stimulation
# =============================================================================
##### load dataframes
pulse_train_thr_over_rate = pd.read_csv("results/Analyses/pulse_train_thr_over_rate_45us.csv")
pulse_train_thr_over_dur = pd.read_csv("results/Analyses/pulse_train_thr_over_dur_45us.csv")

##### confine datasets to human ANF models
pulse_train_thr_over_rate = pulse_train_thr_over_rate[pulse_train_thr_over_rate["model"].isin(["rattay_01", "briaire_05", "smit_10"])]
pulse_train_thr_over_dur = pulse_train_thr_over_dur[pulse_train_thr_over_dur["model"].isin(["rattay_01", "briaire_05", "smit_10"])]

##### add short name of models to tables
pulse_train_thr_over_rate["short_name"] = ""
pulse_train_thr_over_rate["short_name"][pulse_train_thr_over_rate["model"] == "briaire_05"] = "BF"
pulse_train_thr_over_rate["short_name"][pulse_train_thr_over_rate["model"] == "rattay_01"] = "RA"
pulse_train_thr_over_rate["short_name"][pulse_train_thr_over_rate["model"] == "smit_10"] = "SH"

pulse_train_thr_over_dur["short_name"] = ""
pulse_train_thr_over_dur["short_name"][pulse_train_thr_over_dur["model"] == "briaire_05"] = "BF"
pulse_train_thr_over_dur["short_name"][pulse_train_thr_over_dur["model"] == "rattay_01"] = "RA"
pulse_train_thr_over_dur["short_name"][pulse_train_thr_over_dur["model"] == "smit_10"] = "SH"

##### generate plot
thresholds_for_pulse_trains_plot = plot.thresholds_for_pulse_trains(plot_name = "Thresholds for pulse trains",
                                                                    pulse_train_thr_over_rate = pulse_train_thr_over_rate,
                                                                    pulse_train_thr_over_dur = pulse_train_thr_over_dur)

##### save plot
thresholds_for_pulse_trains_plot.savefig("{}/thresholds_for_pulse_trains.pdf".format(paper_image_path), bbox_inches='tight')

# =============================================================================
# Thresholds for sinusodial stimulation
# =============================================================================
##### load dataframes
sinus_thresholds = pd.read_csv("results/Analyses/sinus_thresholds.csv")

##### confine dataset to human ANF models
sinus_thresholds = sinus_thresholds[sinus_thresholds["model"].isin(["rattay_01", "briaire_05", "smit_10"])]

##### add short name of models to tables
sinus_thresholds["short_name"] = ""
sinus_thresholds["short_name"][sinus_thresholds["model"] == "briaire_05"] = "BF"
sinus_thresholds["short_name"][sinus_thresholds["model"] == "rattay_01"] = "RA"
sinus_thresholds["short_name"][sinus_thresholds["model"] == "smit_10"] = "SH"

##### generate plot
thresholds_for_sinus = plot.thresholds_for_sinus(plot_name = "Thresholds for pulse trains",
                                                 sinus_thresholds = sinus_thresholds)

##### save plot
thresholds_for_sinus.savefig("{}/thresholds_for_sinus.pdf".format(paper_image_path), bbox_inches='tight')
