#!/usr/bin/env python2

# Searches the required firing rate of the inhibitory population to so
# as the ouput neuron fires at the same rate as the excitatory
# population.

from scipy.optimize import bisect
from nest import *
import nest.voltage_trace as plot
import matplotlib.pyplot as pyplot

simulation_duration = 100000.0 # milliseconds
n_excit_neurons = 16000
n_inhib_neurons = 4000
firing_freq_excit = 5.0 # Hertz
synaptic_current_excit = 45.0 # picoAmpere
synaptic_current_inhib = -45.0 # picoAmpere
min_firing_freq_inhib = 5.0 # Hertz
max_firing_freq_inhib = 25.0 # Hertz
synaptic_delay = 1.0 # millisecond
percent_accuracy_goal = 0.05

output_neuron = Create("iaf_neuron")
noise_generators = Create("poisson_generator", 2)
voltmeter = Create("voltmeter")
spike_detector = Create("spike_detector")

SetStatus([noise_generators[0]], [{"rate" : n_excit_neurons * firing_freq_excit}])
SetStatus(voltmeter, [{"interval" : 1000.0,
                       "withgid" : True}])

Connect(output_neuron, spike_detector)
Connect(voltmeter, output_neuron)
ConvergentConnect(noise_generators, output_neuron,
                 [synaptic_current_excit, synaptic_current_inhib], 
                 [synaptic_delay, synaptic_delay])

def output_rate(proposed_inhib_rate):
    rate = float(abs(n_inhib_neurons * proposed_inhib_rate))
    SetStatus([noise_generators[1]], [{"rate": rate}])
    SetStatus(spike_detector, [{"n_events": 0}])
    Simulate(simulation_duration)
    n_events = GetStatus(spike_detector, "n_events")[0]
    rate_output_neuron = n_events * 1000.0 / simulation_duration 
    print "proposed_inhib_rate = %.4f Hz," % proposed_inhib_rate,
    print "rate_output_neuron = %.3f Hz" % rate_output_neuron
    return rate_output_neuron

print "Desired target rate: %.2f Hz" % firing_freq_excit
r = bisect(lambda x: output_rate(x)-firing_freq_excit,
           min_firing_freq_inhib, max_firing_freq_inhib,
           rtol=percent_accuracy_goal) 
print "Resulting inhibitory rate: %.4f" % r

plot.from_device(voltmeter, timeunit="s")

pyplot.show()

