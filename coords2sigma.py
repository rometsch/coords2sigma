import numpy as np


def surface_density_from_distribution(q1_coords, q2_coords, q1_grid_i, q2_grid_i, weights=None, polar=False):
    """Convert particle coordinates to a surface density field.
    
    Individuals masses of particles can be specified by passing them as the weights parameter.
    You'll need to rescale appropriately.
    
    Args:
        q1_coords (array): Particle coordinates in first dimension.
        q2_coords (array): Particle coordinates in second dimension.
        q1_grid_i (array): Interface locations in first dimension of the desired grid.
        q2_grid_i (array): Interface locations in second dimension of the desired grid.
        weights (array of same shape as coords): Wheights for the masses of particles.
        polar (bool): Whether or not it is a polar grid.
        
    Returns:
        2d array of shape = (len(q1_grid_i), len(q2_grid_i)): surface density of the particle distribution
        
    Raises:
        ValueError: if grid coordinates are non monotonic or different than linearly or logarithmically spaced.
    """

    # check whether phi is linspaced and mononically increasing
    delta_q2 = np.diff(q2_grid_i)
    if not all(delta_q2 > 0):
        raise ValueError(
            "second grid coordinates must be monotonically increasing")
    if np.std(np.abs(np.diff(q2_grid_i))) > 1e-10:
        raise ValueError("Only linearly spaced phi arrays are supported.")

    # check whether radius array is linearly or logarithmically spaced, monotonicity is automatically checked
    try:
        _ = infer_spacing_type(q1_grid_i)
    except ValueError:
        raise ValueError(
            "Only linearly or logarithmically spaced radii arrays are supported.")

    counts, _, _ = np.histogram2d(q1_coords, q2_coords, bins=(
        q1_grid_i, q2_grid_i), weights=weights)

    if polar:
        area = 0.5*delta_q2[0]*(q1_grid_i[1:]**2 - q1_grid_i[:-1]**2)
        Area = np.tile(np.expand_dims(area, axis=1), (1, len(q2_grid_i)-1))
    else:
        Yi, Xi = np.meshgrid(q2_grid_i, q1_grid_i)
        DXi = np.diff(Xi, axis=0)
        DYi = np.diff(Yi, axis=1)
        Area = DXi*DYi

    sigmadust = counts / Area

    return sigmadust


def infer_spacing_type(ri, thresh=1e-10):
    """Infer whether the radii are space linearly or logarithmicly.

    Compare to reconstructed arrays and take the one with lower deviation.
    The deviation is calculated as the sum over the absolute value of differences 
    devided by the median of the absolute values of the input radii. 

    Args:
        ri (np.array(double)): Radii array.
        thresh (double): Threshold for deviation.

    Return:
        (str, double): Type of the grid ("lin"/"log") and relative deviation.

    Raises:
        ValueError: If both deviations are above the threshold no decision is made.
    """
    r = ri
    r_lin = np.linspace(r[0], r[-1], len(r))
    delta_lin = np.sum(np.abs(r - r_lin))/np.median(np.abs(r))
    try:
        r_log = np.geomspace(r[0], r[-1], len(r))
        delta_log = np.sum(np.abs(r - r_log))/np.median(np.abs(r))
    except ValueError:
        delta_log = 1

    if delta_lin < thresh:
        return ("lin", delta_lin)

    elif delta_log < thresh:
        return ("log", delta_log)

    elif delta_lin > thresh and delta_log > thresh:
        raise ValueError(
            f"Can not determine spacing type. Log and lin deviations are above the {thresh} threshold.")

    else:
        raise ValueError(
            f"The radii array does not seem to be log or lin spaces. Please invesitate manually.")


def resample_grid(q1_i, q2_i, N, N_2=None, polar=False):
    """ Resample the coordinates of a regular grid.

    The grid is resampled to have squared cells if N_2 is not defined.
    Supported are linarly spaced and logarithmically spaced coordinates.

    For polar grids with linear radial spacing, the parameter N_2 has to be defined.

    Args:
        q1_i (np.array(double)): Interface coordinates in first direction.
        q2_i (np.array(double)): Interface coordinates in first direction.
        N (int): Number of grid cells in first direction.
        N_2 (int, optional): Number of grid cells in first direction.
        polar (bool): Specify whether it is a squred grid.

    Returns:
        (new_q1, new_q2) (np.array(double), np.array(double)): New corrdinate arrays.

    Raises:
        ValueError: For a polar grid with linear radial spacing, if N_2 is not defined.
    """

    L1 = q1_i[-1] - q1_i[0]
    L2 = q2_i[-1] - q2_i[0]

    spacing_type_1, _ = infer_spacing_type(q1_i)
    spacing_type_2, _ = infer_spacing_type(q2_i)

    # resample first direction
    if spacing_type_1 == "log":
        new_q1 = np.geomspace(q1_i[0], q1_i[-1], N)
    else:
        new_q1 = np.linspace(q1_i[0], q1_i[-1], N)

    if N_2 is None:
        if not polar:
            # cartesian
            N_2 = int(np.round(L2/L1*N))
        else:
            # for a polar grid with log spacing, cells have the same aspect ratio at all radii
            if spacing_type_1 == "log":
                deltar = np.diff(new_q1)
                deltar_over_r = deltar[0]/(0.5*(new_q1[1]+new_q1[0]))
                N_2 = int(np.round(2*np.pi/deltar_over_r))
            else:
                raise ValueError(
                    "Can not infer N_2 for a polar grid with linear radial spacing. Please set N_2 manually.")

    # resample second direction
    if spacing_type_2 == "log":
        new_q2 = np.geomspace(q2_i[0], q2_i[-1], N_2)
    else:
        new_q2 = np.linspace(q2_i[0], q2_i[-1], N_2)

    return new_q1, new_q2
