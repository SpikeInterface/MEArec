import numpy as np
import scipy.signal



def generate_drift_position_vector(
    fs=None,
    n_samples=None,
    template_locations=None,
    
    start_drift_index=0,
    
    drift_mode_probe='rigid',
    drift_mode_speed='slow',

    slow_drift_velocity=5,
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
    start_drift_index: int
        When the drift start
    drift_mode_probe: 'rigid' or 'non-rigid'
        Global drift or per cell.
    drift_mode_speed: 'slow', 'fast', 'slow+fast'
        drift speed
    
    slow_drift_velocity: float
        drift velocity in um/min.
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
    
    # check on template_locs
    
    if drift_mode_probe == 'rigid':
        drift_vectors = np.zeros(n_samples, dtype='uint16')
    elif drift_mode_probe == 'non-rigid':
        drift_vectors = np.zeros((n_samples, num_cells), dtype='uint16')
    
    # every cells start in the middle of drift vector line
    middle_step = drift_steps // 2
    drift_vectors[:] = middle_step
    
    if 'slow' in drift_mode_speed:
        times = np.arange(n_samples - start_drift_index) / fs
        
        # TODO make slow_drift_velocity to trangle frequency depending on the drift steps um
        freq = 0.5
        triangle = np.abs(scipy.signal.sawtooth(2 * np.pi * freq * times + np.pi / 2))
        triangle *= (drift_steps - 1)
        drift_vectors[start_drift_index:] = triangle
        
    if 'fast' in drift_mode_speed:
        raise NotImplementedError
    
    return drift_vectors
    




def test_generate_drift_position_vector():
    fs = 30000.
    num_cells=4
    drift_steps=20
    template_locations = np.zeros((num_cells, drift_steps, 3))
    
    drift_vectors = generate_drift_position_vector(fs=fs,
        template_locations=template_locations,
        n_samples=30000*20,
        start_drift_index=30000,
        drift_mode_probe='rigid',
        drift_mode_speed='slow',
        )
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    times = np.arange(drift_vectors.shape[0]) / fs
    ax.plot(times, drift_vectors)
    plt.show()

if __name__ == '__main__':
    test_generate_drift_position_vector()
