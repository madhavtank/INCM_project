# Comparison of multi-compartment cable models of human auditory nerve fibers

The multi-compartment human ANF models by Rattay et all (2001), Briaire and Frijns (2005) and Smit et al (2010) were implemented in a single framework. Scripts for the measurement and comparison of model properties, including threshold, conduction velocity, action potential shape, latency, refractory properties, as well as stochastic and temporal behaviors, are provided. All scripts for model analysis support multiprocessing.

## Installation
All scripts were implemented in Python 3.4. and the following packages have to be installed:

* brian2
* thorns
* numpy
* scipy
* pandas
* matplotlib
* itertools

Brian is a simulator for spiking neural networks and provides for an efficient numerical solution of differential equations and the use physical units. To understand how it was used in the scripts, the reader is referred to the Brian2 documentation.

## Usage

### Plot action potential propagation of a model

Run the script **Simulations.py** to plot the action potential (AP) propagation along the fiber and over time for an ANF model. The time course of the ion currents and gating variables can also be plotted, so this script allows for a deeper analyses of the spiking behavior of a model.

### Measure ANF properties of a model

Run the script **Run_test_battery.py** to measure major ANF properties for one selected model. This includes:

* Chronaxie and Rheobase
* Relative spread of thresholds
* Conduction velocity
* AP height, rise and fall time
* Latency
* Absolute and relative refractory period

Furthermore the following figures can be plotted

* Strength-duration curve
* Visualization of relative spread of thresholds
* Single node response
* Refractory curve
* Post-Stimulus Time Histogram

All measured data (and data to reproduce plots) are saved as excel files. Most of the test will take quite along time when run on one core. The parameter "backend" can be set to *multiprocessing* to use multiple cores. This however does only work properly on Linux so far.

### Measure and analyze further ANF characteristics

Run the script **Model_analyses.py** to measure and analyze some further ANF characteristics. Different to the scripts mentioned so far, properties can be measured for several models simultaneously. The following characteristics can be analyzed.

* Thresholds for pulse-train stimulation plotted over pulse rate, train duration and number of pulses (three plots)
* Thresholds for sinusoidal stimulation for different frequencies and stimulus durations
* Latency dependency on stimulus amplitude and electrode distance
* Computational efficiency of the models
* Refractory periods for pulse trains (malfunctioning)
* Dependency of relative spread and jitter on the strength of the applied stochasticity
* Somatic delay measurement

Most of these results are visualized with a figure. Again all measured data are saved as excel files. Again the parameter "backend" can be set to *multiprocessing* to use multiple cores.