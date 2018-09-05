#!/usr/bin/env python

"""Python script to run cell model"""


"""
/* Copyright (c) 2015 EPFL-BBP, All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This work is licensed under a
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode or send a letter to
Creative Commons, 171 Second Street, Suite 300,
San Francisco, California, 94105, USA.
"""

"""
 * @file run.py
 * @brief Run simulation using pyneuron
 * @author Werner Van Geit @ BBP
 * @date 2015
"""

# pylint: disable=C0325, W0212, F0401, W0612, F0401

import os
import neuron
import numpy
import sys


def create_cell():
    """Create the cell model"""
    # Load morphology
    neuron.h.load_file("morphology.hoc")
    # Load biophysics
    neuron.h.load_file("biophysics.hoc")
    # Load main cell template
    neuron.h.load_file("template.hoc")

    # Instantiate the cell from the template

    print("Loading cell cADpyr232_L5_STPC_d16b0be14e")
    cell = neuron.h.cADpyr232_L5_STPC_d16b0be14e(0)
    return cell


def create_stimuli(cell, stim_start, stim_end, current_amplitude):
    """Create the stimuli"""

    print('Attaching stimulus electrodes')

    stimuli = []

    iclamp = neuron.h.IClamp(0.5, sec=cell.soma[0])
    iclamp.delay = stim_start
    iclamp.dur = stim_end - stim_start
    iclamp.amp = current_amplitude
    print('Setting up step current clamp: '
          'amp=%f nA, delay=%f ms, duration=%f ms' %
          (iclamp.amp, iclamp.delay, iclamp.dur))

    stimuli.append(iclamp)

    return stimuli


def create_recordings(cell):
    """Create the recordings"""
    print('Attaching recording electrodes')

    recordings = {}

    recordings['time'] = neuron.h.Vector()
    recordings['soma(0.5)'] = neuron.h.Vector()

    recordings['time'].record(neuron.h._ref_t, 0.1)
    recordings['soma(0.5)'].record(cell.soma[0](0.5)._ref_v, 0.1)

    return recordings


def run_RmpRiTau_step(
        stim_start,
        stim_end,
        current_amplitude,
        plot_traces=None):
    """Run """

    cell = create_cell()
    stimuli = create_stimuli(cell, stim_start, stim_end, current_amplitude)  # noqa
    recordings = create_recordings(cell)

    # Overriding default 30s simulation,
    neuron.h.tstop = stim_end + stim_start
    print(
        'Setting simulation time to %.6g ms for the step current' %
        neuron.h.tstop)

    print('Setting initial voltage to -70 mV')
    neuron.h.v_init = -70

    neuron.h.stdinit()
    neuron.h.dt = 1000
    neuron.h.t = -1e9
    for _ in range(10):
        neuron.h.fadvance()

    neuron.h.t = 0
    neuron.h.dt = 0.025
    neuron.h.frecord_init()

    neuron.h.continuerun(3000)

    time = numpy.array(recordings['time'])
    soma_voltage = numpy.array(recordings['soma(0.5)'])

    recordings_dir = 'python_recordings'

    soma_voltage_filename = os.path.join(
        recordings_dir,
        'soma_voltage_RmpRiTau_step.dat')
    numpy.savetxt(soma_voltage_filename, zip(time, soma_voltage))

    print('Soma voltage for RmpRiTau trace saved to: %s'
          % (soma_voltage_filename))

    if plot_traces:
        import pylab
        pylab.figure(facecolor='white')
        pylab.plot(recordings['time'], recordings['soma(0.5)'])
        pylab.xlabel('time (ms)')
        pylab.ylabel('Vm (mV)')
        pylab.gcf().canvas.set_window_title('RmpRiTau trace')

    return time, soma_voltage, stim_start, stim_end


def init_simulation():
    """Initialise simulation environment"""

    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    print('Loading constants')
    neuron.h.load_file('constants.hoc')


def analyse_RmpRiTau_trace(
        time,
        soma_voltage,
        stim_start,
        stim_end,
        current_amplitude):
    """Analyse the output of the RmpRiTau protocol"""

    # Import the eFeature Extraction Library
    import efel

    # Prepare the trace data
    trace = {}
    trace['T'] = time
    trace['V'] = soma_voltage
    trace['stim_start'] = [stim_start]
    trace['stim_end'] = [stim_end]

    # Calculate the necessary eFeatures
    efel_results = efel.getFeatureValues(
        [trace],
        ['voltage_base', 'steady_state_voltage_stimend',
         'decay_time_constant_after_stim'])

    voltage_base = efel_results[0]['voltage_base'][0]
    ss_voltage = efel_results[0]['steady_state_voltage_stimend'][0]
    dct = efel_results[0]['decay_time_constant_after_stim'][0]

    # Calculate input resistance
    input_resistance = float(ss_voltage - voltage_base) / current_amplitude

    rmpritau_dict = {}

    rmpritau_dict['Rmp'] = '%.6g' % voltage_base
    rmpritau_dict['Rmp_Units'] = 'mV'
    rmpritau_dict['Rin'] = '%.6g' % input_resistance
    rmpritau_dict['Rin_Units'] = 'MOhm'
    rmpritau_dict['Tau'] = '%.6g' % dct
    rmpritau_dict['Tau_Units'] = 'ms'

    print('Resting membrane potential is %s %s' %
          (rmpritau_dict['Rmp'], rmpritau_dict['Rmp_Units']))
    print('Input resistance is %s %s' %
          (rmpritau_dict['Rin'], rmpritau_dict['Rin_Units']))
    print('Time constant is %s %s' %
          (rmpritau_dict['Tau'], rmpritau_dict['Tau_Units']))

    import json

    with open('rmp_ri_tau.json', 'w') as rmpritau_json_file:
        json.dump(rmpritau_dict, rmpritau_json_file,
                        sort_keys=True,
                        indent=4,
                        separators=(',', ': '))


def main(plot_traces=False):
    """Main"""

    # Import matplotlib to plot the traces
    if plot_traces:
        import matplotlib
        matplotlib.rcParams['path.simplify'] = False

    init_simulation()

    current_amplitude = -0.01
    stim_start = 1000
    stim_end = 2000

    time, soma_voltage, stim_start, stim_end = run_RmpRiTau_step(
        stim_start, stim_end, current_amplitude, plot_traces=plot_traces)

    analyse_RmpRiTau_trace(
        time,
        soma_voltage,
        stim_start,
        stim_end,
        current_amplitude)

    if plot_traces:
        import pylab
        pylab.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main(plot_traces=True)
    elif len(sys.argv) == 2 and sys.argv[1] == '--no-plots':
        main(plot_traces=False)
    else:
        raise Exception(
            "Script only accepts one argument: --no-plots, not %s" %
            str(sys.argv))
