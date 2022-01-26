import numpy as np
import scipy.signal



def generate_drift_position_vector(
    fs=None,
    n_samples=None,
    template_locations=None,
    t_start_drift=0.,
    
    
    drift_mode_probe='rigid',
    drift_mode_speed='slow',
    
    non_rigid_non_rigid_gradient_mode='linear',
    preferred_dir=[0, 0, 1],

    slow_drift_velocity=5,
    slow_drift_waveform='triangluar',
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
        sampling frequency
    template_locations: np.array 3d
        shape: (num_cells, drift_steps, 3)
        Template locations for every drift steps
    drift_steps: int
        Number of drift step
    t_start_drift: float
        When the drift start in second
    drift_mode_probe: 'rigid' or 'non-rigid'
        Global drift or per cell.
    drift_mode_speed: 'slow', 'fast', 'slow+fast'
        drift speed
    non_rigid_non_rigid_gradient_mode: 'linear'
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
    drift_vectors: np.array 1d or 2d
        1d (n_samples, ) for rigid
        2d (n_samples, num_cells) for non rigid
    
    """

    num_cells, drift_steps, _ = template_locations.shape
    
    start_drift_index = int(t_start_drift * fs)
    
    
    
    # TODO check on template_locs
    
    
    assert start_drift_index < n_samples, f' samples ({n_samples}) must be < start drift ({start_drift_index})'
    
    drift_vector = np.zeros(n_samples, dtype='float32')
    
    #~ if drift_mode_probe == 'non-rigid':
        
    
    # velocity and jump are computed on bigger drift over cell
    loc0 = template_locations[:, 0, :]
    loc1 = template_locations[:, -1, :]
    dist = np.sum((loc0 - loc1)**2, axis=1) ** 0.5
    min_dist = np.min(dist)
    step = min_dist /drift_steps

    
    if 'slow' in drift_mode_speed:
        
        # compute half period for a regular drift speed
        half_period = min_dist / (slow_drift_velocity / 60)
        # print('half_period', half_period)
        
        # trianle / sin frequence depend on the velocity
        freq = 1. / (2 * half_period)
        
        times = np.arange(n_samples - start_drift_index) / fs
        
        
        if slow_drift_waveform == 'triangluar':
            triangle = np.abs(scipy.signal.sawtooth(2 * np.pi * freq * times + np.pi / 2 ))
            triangle *= min_dist
            triangle -= min_dist / 2.
            
            drift_vector[start_drift_index:] = triangle
        elif slow_drift_waveform == 'sine':
            sine = np.cos(2 * np.pi * freq * times + np.pi / 2)
            sine *= min_dist / 2.

            drift_vector[start_drift_index:] = sine
        else:
            raise NotIMplementedError('slow_drift_waveform')
        
    if 'fast' in drift_mode_speed:
        period = int(fast_drift_period * fs)
        n = int(np.round((n_samples - start_drift_index) / period))
        
        pos = start_drift_index + period
        for i in range(1, n):
            jump = np.random.rand() * (fast_drift_max_jump - fast_drift_min_jump) + fast_drift_min_jump
            if np.random.rand() > 0.5:
                jump = -jump

            # protect from boundaries
            if jump > 0:
                if (np.max(drift_vector[pos: pos + period])+ jump) >= min_dist/2.:
                    jump = - jump
            
            # protect from boundaries
            if jump < 0:
                if (np.min(drift_vector[pos: pos + period]) + jump) <= - min_dist/2.:
                    jump = - jump
                
            drift_vector[pos:] += jump
            pos += period
    
    # Alessio : how to handle the clipping ??
    # fast + slow with similar period is impossible to avoid boundaries
    
    # avoid possible clipping
    drift_vector = np.clip(drift_vector, -min_dist/2, min_dist/2)
    
    # avoid boundary effect
    drift_vector *= 0.99999

    if drift_mode_probe == 'rigid':
        drift_vectors = np.zeros(n_samples, dtype='float32')
        
    elif drift_mode_probe == 'non-rigid':
        assert non_rigid_non_rigid_gradient_mode in ('linear', )
        if non_rigid_non_rigid_gradient_mode == 'linear':
            # vector with shape (num_cells, ) and value between 0 and 1 which is a factor of the velocity
            # the upper units on the 'preferred_dir' vector get 0 the lower get 1
            preferred_dir = np.array(preferred_dir)
            locs = template_locations[:, drift_steps //2 , :]
            proj = np.dot(locs, preferred_dir)
            non_rigid_gradient = (proj - np.min(proj)) / (np.max(proj) - np.min(proj))
        
        drift_vectors = np.zeros((n_samples, num_cells), dtype='float32')
        drift_vectors[:, :] = drift_vector.reshape(-1, 1)
        
        drift_vectors *= non_rigid_gradient.reshape(1, -1)
    

    # shift to positive and to uint16
    drift_vectors += min_dist / 2.
    drift_vectors = np.floor(drift_vectors / step).astype('uint16')
    
    return drift_vectors
    




def test_generate_drift_position_vector():
    fs = 30000.
    num_cells=4
    drift_steps=21
    template_locations = np.zeros((num_cells, drift_steps, 3))
    
    # one cells every 30 um
    template_locations[:, :, 2] = np.arange(num_cells).reshape(-1, 1) * 25.
    
    # drift step 1.5 um of Z
    template_locations[:, :, 2] += np.arange(drift_steps) * 1.5
    
    
    drift_vectors = generate_drift_position_vector(fs=fs,
        template_locations=template_locations,
        n_samples=30000*600,
        t_start_drift=10. ,
        #~ drift_mode_probe='rigid',
        drift_mode_probe='non-rigid',
        #~ drift_mode_speed='slow+fast',
        #~ drift_mode_speed='fast',
        drift_mode_speed='slow',

        non_rigid_non_rigid_gradient_mode='linear',
        preferred_dir=[0, 0, 1],
        
        
        slow_drift_velocity=5.,
        slow_drift_waveform='triangluar',
        #~ slow_drift_velocity=17,
        #~ slow_drift_waveform= 'sine',

        fast_drift_period=120.,
        fast_drift_max_jump=8.,
        fast_drift_min_jump=1.,
        
        )
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    times = np.arange(drift_vectors.shape[0]) / fs
    ax.plot(times, drift_vectors)
    ax.axhline(drift_steps-1, ls='--', color='k')
    plt.show()

if __name__ == '__main__':
    test_generate_drift_position_vector()

