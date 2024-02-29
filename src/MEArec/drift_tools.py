import numpy as np
import scipy.signal


def generate_drift_dict_from_params(
    drift_fs,
    duration,
    template_locations,
    external_drift_vector_um=None,
    external_drift_times=None,
    t_start_drift=None,
    t_end_drift=None,
    drift_mode_probe="rigid",
    drift_mode_speed="slow",
    non_rigid_gradient_mode="linear",
    non_rigid_linear_direction=1,
    non_rigid_linear_min_factor=0.5,
    non_rigid_step_depth_boundary=None,
    non_rigid_step_factors=None,
    preferred_dir=[0, 0, 1],
    slow_drift_velocity=5,
    slow_drift_waveform="triangluar",
    slow_drift_amplitude=None,
    fast_drift_period=10.0,
    fast_drift_max_jump=20.0,
    fast_drift_min_jump=5.0,
    external_drift_factors=None,
):
    """
    Generate for each units a drift position vector.
    The output vector handle  of indexes for drift steps along time.

    Parameters
    ----------
    drift_fs: float
        Sampling frequency in Hz
    duration: float
        Duration drift vector in s
    template_locations: np.array 3d
        Shape: (num_cells, drift_steps, 3)
        Template locations for every drift steps
    external_drift_vector_um : 1d array or None
        External drift signal to apply. It needs to be consistent with drift_fs and duration (or external_drift_times)
    external_drift_times : 1d array or None
        External drift times to use for external_drift_signal_um.
    t_start_drift: float
        When the drift start in second
    drift_mode_probe: 'rigid' or 'non-rigid'
        Global drift or per cell.
    drift_mode_speed: str 'slow', 'fast', 'slow+fast'
        Drift mode speed
    non_rigid_gradient_mode: 'linear' | 'step'
        Mode to generate non rigid gradient over units on velocity.
        If 'linear', drift factors (0.5-1) are assigned to each cells linearly with depth (deeper cells have more drift).
    non_rigid_linear_direction : int
        Direction of linear gradient.
        Only used if 'drift_mode_probe' is 'non-rigid' and 'non_rigid_gradient_mode' is 'linear'
        If positive, deeper cells drift more than surface cells.
        If negative, deeper cells drift more than surface cells.
        Zero is not allowed, by default 1
    non_rigid_linear_min_factor: float (default 0.5)
        When non_rigid_gradient_mode='linear' this control the minimum factor. The max is always 1.
    non_rigid_step_depth_boundary : float or None
        Depth boundary in the preferred_dir dimension to apply the drift step,
        by default is half of the template locations extensions.
        Only used if 'drift_mode_probe' is 'non-rigid' and 'non_rigid_gradient_mode' is 'step'
    non_rigid_step_factors : None or tuple/list of two elements
        Factors to apply to cells above (non_rigid_step_factors[0]) and below (non_rigid_step_factors[1])
        step depth boundary. Default is (1, 0).
        Only used if 'drift_mode_probe' is 'non-rigid' and 'non_rigid_gradient_mode' is 'step'
    preferred_dir: list of int
        Use for gradient when non rigid mode.
    slow_drift_velocity: float
        Slow drift velocity in um/min.
    slow_drift_waveform: 'trianglar' or 'sine'
        Waveform shape for slow drift
    fast_drift_period: float
        Period between fast drift events
    fast_drift_max_jump: float
        Maximum 'jump' in um for fast drifts
    fast_drift_min_jump: float
        Minimum 'jump' in um for fast drifts
    external_drift_factors: None
        If templates are selected externally (and template_ids) is used in the
        generate_recordings() function, this vector specifies the factor (0-1)
        for each cell.

    Returns
    -------
    drift_list: list of dicts
        Each dict contains info for a drift signal:
            * "drift_times" (1d array): times for different drift vectors. (Only for external times)
            * "drift_fs" (float): sampling frequency
            * "drift_vector_um" (1d array): drift signals in um
            * "drift_vector_idxs" (1d array): drift signals in template idxs (with respect to middle step)
            * "drift_factor" (1d array): array with gradient for each unit for the drift signal
               vector. For rigid motion, all values are 1.
    """

    num_cells, drift_steps, _ = template_locations.shape
    n_samples = int(drift_fs * duration)

    drift_distances = np.linalg.norm(template_locations[:, -1] - template_locations[:, 0])
    mean_distance = np.mean(drift_distances)
    assert np.max(drift_distances - mean_distance) < 0.05 * mean_distance, (
        "Drift distances are not constant across templates. You can control this by setting "
        "'max_drift' and 'min_drift' to the same value and 'drift_zlim = [desired_drift, desired_drift]' "
        "in the template parameters."
    )
    # velocity and jump are computed on bigger drift over cell
    loc0 = template_locations[:, 0, :]
    loc1 = template_locations[:, -1, :]
    dist = np.sum((loc0 - loc1) ** 2, axis=1) ** 0.5
    dist_boundary = np.min(dist)
    step = dist_boundary / drift_steps

    if external_drift_vector_um is None:
        assert drift_mode_speed in ["slow", "fast"]
        t_start_drift = 0.0 if t_start_drift is None else t_start_drift
        t_end_drift = duration if t_end_drift is None else t_end_drift
        assert t_start_drift < duration, f"'t_start_drift' must preceed 'duration'!"
        assert t_end_drift <= duration, f"'t_end_drift' must preceed 'duration'!"
        start_drift_index = int(t_start_drift * drift_fs)
        end_drift_index = int(t_end_drift * drift_fs)

        drift_vector_um = np.zeros(n_samples, dtype="float32")
        # drift times is only used for external drift that are not defined over the entire recording
        drift_times = None

        if drift_mode_speed == "slow":
            if slow_drift_amplitude is None:
                slow_drift_amplitude = dist_boundary

            # compute half period for a regular drift speed
            half_period = slow_drift_amplitude / (slow_drift_velocity / 60)

            # triangle / sine frequency depends on the velocity
            freq = 1.0 / (2 * half_period)

            drift_times = np.arange(end_drift_index - start_drift_index) / drift_fs

            if slow_drift_waveform == "triangluar":
                triangle = np.abs(scipy.signal.sawtooth(2 * np.pi * freq * drift_times + np.pi / 2))
                triangle *= slow_drift_amplitude
                triangle -= slow_drift_amplitude / 2.0

                drift_vector_um[start_drift_index:end_drift_index] = triangle
                drift_vector_um[end_drift_index:] = triangle[-1]
            elif slow_drift_waveform == "sine":
                sine = np.cos(2 * np.pi * freq * drift_times + np.pi / 2)
                sine *= slow_drift_amplitude / 2.0

                drift_vector_um[start_drift_index:end_drift_index] = sine
                drift_vector_um[end_drift_index:] = sine[-1]
            else:
                raise NotImplementedError("slow_drift_waveform")

        else:  # 'fast' in drift_mode_speed:
            period_samples = int(fast_drift_period * drift_fs)
            n = int(np.round((end_drift_index - start_drift_index) / period_samples))

            t_idx = start_drift_index + period_samples
            for i in range(1, n):
                jump = np.random.rand() * (fast_drift_max_jump - fast_drift_min_jump) + fast_drift_min_jump
                if np.random.rand() > 0.5:
                    jump = -jump

                # protect from boundaries
                if jump > 0:
                    if (np.max(drift_vector_um[t_idx : t_idx + period_samples]) + jump) >= dist_boundary / 2.0:
                        jump = -jump

                # protect from boundaries
                if jump < 0:
                    if (np.min(drift_vector_um[t_idx : t_idx + period_samples]) + jump) <= -dist_boundary / 2.0:
                        jump = -jump

                drift_vector_um[t_idx:] += jump
                t_idx += period_samples
    else:
        if external_drift_times is None:
            assert (
                len(external_drift_vector_um) == n_samples
            ), f"'external_drift_times' must have a length of {n_samples} (duration * drift_fs)."
        else:
            assert all(np.diff(external_drift_times) > 0), "'external_drift_times' should be monotonically increasing"
            assert external_drift_times[-1] <= duration, "'external_drift_times' cannot exceed duration"
            assert len(external_drift_times) == len(
                external_drift_vector_um
            ), "'external_drift_times' and 'external_drift_vector_um' must have the same length"
            drift_vector_um = external_drift_vector_um
            drift_vector_um *= 0.99999
            drift_times = external_drift_times

    if external_drift_factors is None:
        if drift_mode_probe == "rigid":
            drift_factors = np.ones(num_cells)
        elif drift_mode_probe == "non-rigid":
            assert non_rigid_gradient_mode in ("linear", "step")
            # project locations over preferred direction
            preferred_dir = np.array(preferred_dir).reshape(-1, 1)
            locs = template_locations[:, drift_steps // 2, :]
            proj = np.dot(locs, preferred_dir).squeeze()
            if non_rigid_gradient_mode == "linear":
                assert 0 <= non_rigid_linear_min_factor < 1, "non_rigid_linear_min_factor must between 0 and 1"
                assert non_rigid_linear_direction in (1, -1, 1.0, -1.0), "non_rigid_linear_direction must be 1 or -1"
                # vector with shape (num_cells, ) and value between 0.5 and 1 which is a factor of the velocity
                # the upper units on the 'preferred_dir' vector get 0.5 the lower get 1
                print(f"Linear gradient with Min: {non_rigid_linear_min_factor} Dir: {non_rigid_linear_direction}")
                drift_factors = (proj - np.min(proj)) / (np.max(proj) - np.min(proj))
                if non_rigid_linear_direction > 0:
                    drift_factors = 1 - drift_factors
                f = non_rigid_linear_min_factor
                drift_factors = drift_factors * (1 - f) + f
            elif non_rigid_gradient_mode == "step":
                if non_rigid_step_depth_boundary is None:
                    non_rigid_step_depth_boundary = np.mean(proj)
                else:
                    assert (
                        np.min(proj) < non_rigid_step_depth_boundary < np.max(proj)
                    ), f"'non_rigid_step_depth_boundary' needs to be in between {np.min(proj)} and {np.max(proj)}"
                if non_rigid_step_factors is None:
                    non_rigid_step_factors = (1, 0)
                else:
                    assert len(non_rigid_step_factors) == 2, "'non_rigid_step_factors' needs two values"
                    assert (
                        0 <= non_rigid_step_factors[0] <= 1
                    ), "'non_rigid_step_factors' needs to be in the [0, 1] range"
                    assert (
                        0 <= non_rigid_step_factors[1] <= 1
                    ), "'non_rigid_step_factors' needs to be in the [0, 1] range"
                drift_factors = non_rigid_step_factors[0] * np.ones(num_cells)
                drift_factors[proj < non_rigid_step_depth_boundary] = non_rigid_step_factors[1]
    else:
        assert (
            len(external_drift_factors) == num_cells
        ), f"'external_drift_factors' must be defined for all {num_cells} cells"
        assert all(0 <= d <= 1 for d in external_drift_factors), "'external_drift_factors' must be in the [0, 1] range"
        drift_factors = external_drift_factors

    # avoid possible clipping
    drift_vector_um = np.clip(drift_vector_um, -dist_boundary / 2, dist_boundary / 2)

    # avoid boundary effect
    drift_vector_um *= 0.99999

    # shift with respect to mid point in int16
    drift_vector_idxs = (np.floor(drift_vector_um / step)).astype("int16")

    drift_dict = {}
    drift_dict["drift_vector_um"] = drift_vector_um
    drift_dict["drift_vector_idxs"] = drift_vector_idxs
    drift_dict["drift_fs"] = drift_fs
    drift_dict["drift_times"] = drift_times
    drift_dict["drift_factors"] = drift_factors
    drift_dict["drift_steps"] = drift_steps

    return drift_dict


def test_generate_drift_position_vector():
    fs = 2.0
    duration = 600
    num_cells = 4
    drift_steps = 21
    template_locations = np.zeros((num_cells, drift_steps, 3))

    # one cells every 30 um
    template_locations[:, :, 2] = np.arange(num_cells).reshape(-1, 1) * 25.0

    # drift step 1.5 um of Z
    template_locations[:, :, 2] += np.arange(drift_steps) * 1.5

    drift_dict = generate_drift_dict_from_params(
        drift_fs=fs,
        template_locations=template_locations,
        duration=duration,
        t_start_drift=10.0,
        # drift_mode_probe='rigid',
        drift_mode_probe="non-rigid",
        # drift_mode_speed='slow+fast',
        # drift_mode_speed='fast',
        drift_mode_speed="slow",
        non_rigid_gradient_mode="linear",
        preferred_dir=[0, 0, 1],
        # slow_drift_velocity=5.,
        # slow_drift_waveform='triangluar',
        slow_drift_velocity=17,
        slow_drift_waveform="sine",
        fast_drift_period=120.0,
        fast_drift_max_jump=5,
        fast_drift_min_jump=1.0,
    )

    drift_vectors = drift_dict["drift_vectors"]
    drift_fs = drift_dict["drift_fs"]
    times = np.arange(int(fs * duration)) / drift_fs

    print(drift_vectors.shape)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(times, drift_vectors)
    ax.axhline(drift_steps - 1, ls="--", color="k")
    plt.ion()
    plt.show()


if __name__ == "__main__":
    test_generate_drift_position_vector()
