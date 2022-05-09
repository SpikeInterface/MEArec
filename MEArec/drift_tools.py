import numpy as np
import scipy.signal



def generate_drift_position_vector(
        drift_fs,
        duration,
        template_locations,
        t_start_drift=0.,
        t_end_drift=None,
        drift_mode_probe='rigid',
        drift_mode_speed='slow',
        non_rigid_gradient_mode='linear',
        preferred_dir=[0, 0, 1],
        slow_drift_velocity=5,
        slow_drift_waveform='triangluar',
        slow_drift_amplitude=None,
        fast_drift_period=10.,
        fast_drift_max_jump=20.,
        fast_drift_min_jump=5.,
    ):
    """
    Generate for each units a drift position vector.
    The output vector handle  of indexes for drift steps along time.
    
    Parameters
    ----------
    fs: float
        sampling frequency in Hz
    duration: float
        Duration drift vector in s
    template_locations: np.array 3d
        shape: (num_cells, drift_steps, 3)
        Template locations for every drift steps
    t_start_drift: float
        When the drift start in second
    drift_mode_probe: 'rigid' or 'non-rigid'
        Global drift or per cell.
    drift_mode_speed: 'slow', 'fast', 'slow+fast'
        drift speed
    non_rigid_gradient_mode: 'linear'
        Mode to generate non rigid gradient over units on velocity.
    preferred_dir: list of int
        Use for gradient when non rigid mode.
    slow_drift_velocity: float
        drift velocity in um/min.
    slow_drift_waveform: 'trianglar' or 'sine'
        waveform for slow drift
    fast_drift_period: float
        period between fast drift events
    fast_drift_max_jump: float
        maximum 'jump' in um for fast drifts
    fast_drift_min_jump: float
        minimum 'jump' in um for fast drifts

    Returns
    -------
    drift_list: list of dicts
        Each dict contains info for a drift signal:
            * "drift_times" (1d array): times for different drift vectors
            * "drift_fs" (float): sampling frequency
            * "drift_vector_um" (1d array): drift signals in um
            * "drift_vector_idxs" (1d array): drift signals in template idxs (centered on middle step)
            * "drift_mask" (1d array): array with gradient for each unit for the drift signal
               vector. For rigid motion, all values are 1.
    """

    num_cells, drift_steps, _ = template_locations.shape
     
    n_samples = int(drift_fs * duration)
    
    # TODO check on template_locs: they should be consistent!
    
    if t_end_drift is None:
        t_end_drift = duration
    
    assert t_start_drift < duration, f"'t_start_drift' must preceed 'duration'!"
    assert t_end_drift <= duration, f"'t_end_drift' must preceed 'duration'!"
    
    start_drift_index = int(t_start_drift * drift_fs)   
    end_drift_index = int(t_end_drift * drift_fs)    
    
    drift_vector_um = np.zeros(n_samples, dtype='float32')
    drift_times = np.arange(n_samples) / drift_fs        
    
    # velocity and jump are computed on bigger drift over cell
    loc0 = template_locations[:, 0, :]
    loc1 = template_locations[:, -1, :]
    dist = np.sum((loc0 - loc1)**2, axis=1) ** 0.5
    dist_boundary = np.min(dist)    
    
    if slow_drift_amplitude is None:
        slow_drift_amplitude = dist_boundary
        
    step = dist_boundary / drift_steps
    
    if 'slow' in drift_mode_speed:
        
        # compute half period for a regular drift speed
        half_period = slow_drift_amplitude / (slow_drift_velocity / 60)
        
        # triangle / sine frequency depends on the velocity
        freq = 1. / (2 * half_period)
        
        times = np.arange(end_drift_index - start_drift_index) / drift_fs

        if slow_drift_waveform == 'triangluar':
            triangle = np.abs(scipy.signal.sawtooth(2 * np.pi * freq * times + np.pi / 2 ))
            triangle *= slow_drift_amplitude
            triangle -= slow_drift_amplitude / 2.
            
            drift_vector_um[start_drift_index:end_drift_index] = triangle
            drift_vector_um[end_drift_index:] = triangle[-1]
        elif slow_drift_waveform == 'sine':
            sine = np.cos(2 * np.pi * freq * times + np.pi / 2)
            sine *= slow_drift_amplitude / 2.

            drift_vector_um[start_drift_index:end_drift_index] = sine
            drift_vector_um[end_drift_index:] = sine[-1]
        else:
            raise NotImplementedError('slow_drift_waveform')
        
    if 'fast' in drift_mode_speed:
        period_samples = int(fast_drift_period * drift_fs)
        n = int(np.round((end_drift_index - start_drift_index) / period_samples))
        
        t_idx = start_drift_index + period_samples
        for i in range(1, n):
            jump = np.random.rand() * (fast_drift_max_jump - fast_drift_min_jump) + fast_drift_min_jump
            if np.random.rand() > 0.5:
                jump = -jump

            # protect from boundaries
            if jump > 0:
                if (np.max(drift_vector_um[t_idx: t_idx + period_samples])+ jump) >= dist_boundary/2.:
                    jump = - jump
            
            # protect from boundaries
            if jump < 0:
                if (np.min(drift_vector_um[t_idx: t_idx + period_samples]) + jump) <= - dist_boundary/2.:
                    jump = - jump
                
            drift_vector_um[t_idx:] += jump
            t_idx += period_samples
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(drift_times, drift_vector)
    
    # Alessio : how to handle the clipping ??
    # fast + slow with similar period is impossible to avoid boundaries
    
    # avoid possible clipping
    drift_vector_um = np.clip(drift_vector_um, -dist_boundary/2, dist_boundary/2)
    
    # avoid boundary effect
    drift_vector_um *= 0.99999
    
    if drift_mode_probe == 'rigid':
        drift_mask = np.ones(num_cells)
        
    elif drift_mode_probe == 'non-rigid':
        assert non_rigid_gradient_mode in ('linear', )
        if non_rigid_gradient_mode == 'linear':
            # vector with shape (num_cells, ) and value between 0 and 1 which is a factor of the velocity
            # the upper units on the 'preferred_dir' vector get 0 the lower get 1
            preferred_dir = np.array(preferred_dir).reshape(-1, 1)
            locs = template_locations[:, drift_steps //2 , :]
            proj = np.dot(locs, preferred_dir)
            # print(proj[:8])
            non_rigid_gradient = (proj - np.min(proj)) / (np.max(proj) - np.min(proj))
            # print(non_rigid_gradient[:8])
            # print(template_locations[:, drift_steps //2 , :][:8])
            # raise Exception()
            non_rigid_gradient = 1 - non_rigid_gradient
            drift_mask = non_rigid_gradient.squeeze()
        else:
            raise NotImplementedError

        # drift_vectors = np.zeros((n_samples, num_cells), dtype='float32')
        # drift_vectors[:, :] = drift_vector.reshape(-1, 1)
        
        # drift_vectors *= non_rigid_gradient.reshape(1, -1)
    
    # shift to positive and to uint16
    drift_vector_idxs = drift_vector_um + slow_drift_amplitude / 2.
    drift_vector_idxs = np.floor(drift_vector_idxs / step).astype('uint16')
    
    drift_dict = {}
    drift_dict["drift_vector_um"] = drift_vector_um
    drift_dict["drift_vector_idxs"] = drift_vector_idxs
    drift_dict["drift_fs"] = drift_fs
    drift_dict["drift_times"] = drift_times
    drift_dict["drift_mask"] = drift_mask
    
    return [drift_dict]
    




def test_generate_drift_position_vector():
    fs = 2.
    duration = 600
    num_cells = 4
    drift_steps = 21
    template_locations = np.zeros((num_cells, drift_steps, 3))
    
    # one cells every 30 um
    template_locations[:, :, 2] = np.arange(num_cells).reshape(-1, 1) * 25.
    
    # drift step 1.5 um of Z
    template_locations[:, :, 2] += np.arange(drift_steps) * 1.5
    
    
    drift_dict = generate_drift_position_vector(
        drift_fs=fs,
        template_locations=template_locations,
        duration=duration,
        t_start_drift=10. ,
        # drift_mode_probe='rigid',
        drift_mode_probe='non-rigid',
        # drift_mode_speed='slow+fast',
        # drift_mode_speed='fast',
        drift_mode_speed='slow',

        non_rigid_gradient_mode='linear',
        preferred_dir=[0, 0, 1],
        
        
        # slow_drift_velocity=5.,
        # slow_drift_waveform='triangluar',
        slow_drift_velocity=17,
        slow_drift_waveform= 'sine',

        fast_drift_period=120.,
        fast_drift_max_jump=5,
        fast_drift_min_jump=1.,
        )
    
    drift_vectors = drift_dict["drift_vectors"]
    drift_fs = drift_dict["drift_fs"]
    times = np.arange(int(fs * duration)) / drift_fs
    
    print(drift_vectors.shape)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(times, drift_vectors)
    ax.axhline(drift_steps-1, ls='--', color='k')
    plt.ion()
    plt.show()

if __name__ == '__main__':
    test_generate_drift_position_vector()

