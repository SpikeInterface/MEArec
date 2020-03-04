import MEArec as mr
import time
from pathlib import Path

template_performance = False
recording_performance = True
template_folder = 'data/templates/'
verbose = False
n_jobs = None

if template_performance:
    # TEMPLATES GENERATION
    template_params = mr.get_default_templates_params()
    cell_models_folder = mr.get_default_cell_models_folder()

    # tetrode - n=30
    print('Tetrode - n=30 - min amp=30')
    prb = 'tetrode-mea-l'
    n = 30
    min_amp = 30
    template_params['probe'] = prb
    template_params['n'] = n
    template_params['min_amp'] = min_amp

    t_start = time.time()
    _ = mr.gen_templates(cell_models_folder=cell_models_folder, params=template_params, n_jobs=n_jobs, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'n', n, 'min amp', min_amp, ':', elapsed_time, 's')

    # tetrode - n=100
    print('Tetrode - n=100 - min amp=30')
    prb = 'tetrode-mea-l'
    n = 100
    min_amp = 30
    template_params['probe'] = prb
    template_params['n'] = n
    template_params['min_amp'] = min_amp

    t_start = time.time()
    _ = mr.gen_templates(cell_models_folder=cell_models_folder, params=template_params, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'n', n, 'min amp', min_amp, ':', elapsed_time, 's')

    # tetrode - n=100 - min_amp=0
    print('Tetrode - n=100 - min amp=0')
    prb = 'tetrode-mea-l'
    n = 100
    min_amp = 0
    template_params['probe'] = prb
    template_params['n'] = n
    template_params['min_amp'] = min_amp

    t_start = time.time()
    _ = mr.gen_templates(cell_models_folder=cell_models_folder, params=template_params, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'n', n, 'min amp', min_amp, ':', elapsed_time, 's')

    # Neuronexus - n=30
    print('Neuronexus-32 - n=100 - min amp=30')
    prb = 'Neuronexus-32'
    n = 100
    min_amp = 30
    template_params['probe'] = prb
    template_params['n'] = n
    template_params['min_amp'] = min_amp

    t_start = time.time()
    _ = mr.gen_templates(cell_models_folder=cell_models_folder, params=template_params, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'n', n, 'min amp', min_amp, ':', elapsed_time, 's')

    # Neuropixels-128 - n=30
    print('Neuropixels-128 - n=100 - min amp=30')
    prb = 'Neuropixels-128'
    n = 100
    min_amp = 30
    template_params['probe'] = prb
    template_params['n'] = n
    template_params['min_amp'] = min_amp

    t_start = time.time()
    _ = mr.gen_templates(cell_models_folder=cell_models_folder, params=template_params, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'n', n, 'min amp', min_amp, ':', elapsed_time, 's')

    # SqMEA-10-15 - n=30
    print('SqMEA-10-15 - n=100 - min amp=30')
    prb = 'SqMEA-10-15'
    n = 100
    min_amp = 30
    template_params['probe'] = prb
    template_params['n'] = n
    template_params['min_amp'] = min_amp

    t_start = time.time()
    _ = mr.gen_templates(cell_models_folder=cell_models_folder, params=template_params, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'n', n, 'min amp', min_amp, ':', elapsed_time, 's')

    # Neuronexus - n=30 - drifting
    print('Neuronexus-32 - n=100 - min amp=30 - drifting steps=50')
    prb = 'Neuronexus-32'
    n = 100
    min_amp = 30
    drifting = True
    drift_steps = 50
    template_params['probe'] = prb
    template_params['n'] = n
    template_params['min_amp'] = min_amp
    template_params['drifting'] = drifting
    template_params['drift_steps'] = drift_steps

    t_start = time.time()
    _ = mr.gen_templates(cell_models_folder=cell_models_folder, params=template_params, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'n', n, 'min amp', min_amp, 'drifting n_steps', drift_steps, ':', elapsed_time, 's')


if recording_performance:
    rec_params = mr.get_default_recordings_params()
    rec_params['recordings']['chunk_duration'] = 20
    n_jobs = 4

    # tetrode - d=10 - ncells=6
    print('Tetrode - d=10 - ncells=6')
    templates = template_folder + 'templates_30_tetrode.h5'
    prb = 'tetrode'
    duration = 10
    n_exc = 4
    n_inh = 2
    rec_params['spiketrains']['duration'] = duration
    rec_params['spiketrains']['n_exc'] = n_exc
    rec_params['spiketrains']['n_inh'] = n_inh

    t_start = time.time()
    _ = mr.gen_recordings(templates=templates, params=rec_params, n_jobs=n_jobs, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'ncells', n_exc + n_inh, 'duration', duration, ':', elapsed_time, 's')

    # tetrode - d=600 - ncells=6
    print('Tetrode - d=600 - ncells=6')
    templates = template_folder + 'templates_30_tetrode.h5'
    prb = 'tetrode'
    duration = 600
    n_exc = 4
    n_inh = 2
    rec_params['spiketrains']['duration'] = duration
    rec_params['spiketrains']['n_exc'] = n_exc
    rec_params['spiketrains']['n_inh'] = n_inh

    t_start = time.time()
    _ = mr.gen_recordings(templates=templates, params=rec_params, n_jobs=n_jobs, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'ncells', n_exc + n_inh, 'duration', duration, ':', elapsed_time, 's')

    # Neuronexus - d=30 - ncells=20
    print('Neuronexus - d=30 - ncells=20')
    templates = template_folder + 'templates_100_Neuronexus-32.h5'
    prb = 'neuronexus'
    duration = 30
    n_exc = 16
    n_inh = 4
    rec_params['spiketrains']['duration'] = duration
    rec_params['spiketrains']['n_exc'] = n_exc
    rec_params['spiketrains']['n_inh'] = n_inh

    t_start = time.time()
    _ = mr.gen_recordings(templates=templates, params=rec_params, n_jobs=n_jobs, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'ncells', n_exc + n_inh, 'duration', duration, ':', elapsed_time, 's')

    # Neuronexus - d=30 - ncells=20
    print('Neuronexus - d=30 - ncells=20 - bursting')
    templates = template_folder + 'templates_30_Neuronexus-32_drift.h5'
    prb = 'neuronexus'
    duration = 30
    n_exc = 16
    n_inh = 4
    rec_params['spiketrains']['duration'] = duration
    rec_params['spiketrains']['n_exc'] = n_exc
    rec_params['spiketrains']['n_inh'] = n_inh
    rec_params['recordings']['bursting'] = True
    rec_params['recordings']['shape_mod'] = True

    t_start = time.time()
    _ = mr.gen_recordings(templates=templates, params=rec_params, n_jobs=n_jobs, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'ncells', n_exc + n_inh, 'duration', duration, 'bursting:', elapsed_time, 's')

    # Neuropixels - d=30 - ncells=60
    print('Neuropixels - d=30 - ncells=60')
    templates = template_folder + 'templates_100_Neuropixels-128.h5'
    prb = 'neuropixels'
    duration = 30
    n_exc = 48
    n_inh = 12
    rec_params['spiketrains']['duration'] = duration
    rec_params['spiketrains']['n_exc'] = n_exc
    rec_params['spiketrains']['n_inh'] = n_inh
    rec_params['recordings']['bursting'] = False
    rec_params['recordings']['shape_mod'] = False

    t_start = time.time()
    _ = mr.gen_recordings(templates=templates, params=rec_params, n_jobs=n_jobs, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'ncells', n_exc + n_inh, 'duration', duration, ':', elapsed_time, 's')

    # SqMEA - d=30 - ncells=60
    print('SqMEA - d=30 - ncells=60')
    templates = template_folder + 'templates_100_SqMEA-10-15.h5'
    prb = 'sqmea'
    duration = 30
    n_exc = 40
    n_inh = 10
    rec_params['spiketrains']['duration'] = duration
    rec_params['spiketrains']['n_exc'] = n_exc
    rec_params['spiketrains']['n_inh'] = n_inh

    t_start = time.time()
    _ = mr.gen_recordings(templates=templates, params=rec_params, n_jobs=n_jobs, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'ncells', n_exc + n_inh, 'duration', duration, ':', elapsed_time, 's')

    # Neuronexus - d=60 - ncells=4 - drifting
    print('Neuronexus - d=30 - ncells=20 - drifting')
    templates = template_folder + 'templates_30_Neuronexus-32_drift.h5'
    prb = 'neuronexus'
    duration = 60
    n_exc = 3
    n_inh = 1
    rec_params['spiketrains']['duration'] = duration
    rec_params['spiketrains']['n_exc'] = n_exc
    rec_params['spiketrains']['n_inh'] = n_inh
    rec_params['recordings']['bursting'] = False
    rec_params['recordings']['shape_mod'] = False
    rec_params['recordings']['drifting'] = True

    t_start = time.time()
    _ = mr.gen_recordings(templates=templates, params=rec_params, n_jobs=n_jobs, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'ncells', n_exc + n_inh, 'duration', duration, 'drifring:', elapsed_time, 's')


    # Neuronexus - d=30 - ncells=20 - bursting+drifting
    print('Neuronexus - d=30 - ncells=20 - bursting - drifting')
    templates = template_folder + 'templates_30_Neuronexus-32_drift.h5'
    prb = 'neuronexus'
    duration = 60
    n_exc = 16
    n_inh = 4
    rec_params['spiketrains']['duration'] = duration
    rec_params['spiketrains']['n_exc'] = n_exc
    rec_params['spiketrains']['n_inh'] = n_inh
    rec_params['recordings']['bursting'] = False
    rec_params['recordings']['shape_mod'] = False
    rec_params['recordings']['drifting'] = True

    t_start = time.time()
    _ = mr.gen_recordings(templates=templates, params=rec_params, n_jobs=n_jobs, verbose=verbose)
    elapsed_time = time.time() - t_start
    print('Time', prb, 'ncells', n_exc + n_inh, 'duration', duration, 'drifring:', elapsed_time, 's')
