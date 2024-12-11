import numpy as np

#adapted from my week 11 make_tridagonal
def make_tridiagonal(N, b, d, a):
    """
    Makes a tridiagonal NxN matrix with the elements b, d, and a from bottom left to top right
    :param N: the size of the matrix to create
    :param b: the value of the elements immediately to the left of the main diagonal
    :param d: the values on the main diagonal
    :param a: the value of the elements immediately to the right of the main diagonal
    :return: the tridiagonal matrix
    """
    return d*np.eye(N)+b*np.eye(N, k=-1)+a*np.eye(N, k=1)

#copied from my week 11 submission
def make_initialcond(sigma_0, k_0, spatial_grid):
    """
    Creates the initial conditions as a wave packet with a form given in the assignment 10 description
    :param sigma_0: an adjustable parameter in the wave packet equation
    :param k_0: another adjustable parameter in the wave packet equation
    :param spatial_grid: the grid of x values to set the initial conditions of
    :return: an array of amplitudes for each position given by the spatial_grid parameter
    """
    return np.exp(-(spatial_grid**2/(2*sigma_0**2)))*np.cos(k_0*spatial_grid)

#copied from my week 11 submission
def spectral_radius(A):
    """
    Calculates the eigenvalue of the matrix A with greatest absolute magnitude
    :param A: the matrix to find the eigenvalue with the greatest absolute magnitude of
    :return: the eigenvalue with the greatest absolute magnitude
    """
    eig_vals = np.linalg.eig(A)[0]
    return eig_vals[np.where(np.abs(eig_vals) == np.max(np.abs(eig_vals)))][0]

#adapted from my week 11 advection1d function
def sch_eqn(nspace, ntime, tau, method="ftcs", length = 200, potential = [], wparam=[10, 0, 0.5]):
    """
        Solves the 1d advection equation
        :param method: the solution method to use
        :param nspace: the number of divisions in the spatial grid
        :param ntime: the number of time steps
        :param tau_rel: the time step expressed as a multiple of the critical time step
        :param params: a tuple of physical parameters for the system (L, c) where L is the length of the medium and c is the wavespeed in the medium
        :return: a tuple (a, x, t) where a is a 2d array of solutions for all x at each time, x is a 1d array of the points on the spatial grid, and t is a 1d array of the times which were solved for
        """
    # define variables
    sigma_0 = wparam[0]
    x_0 = wparam[1]
    k_0 = wparam[2]
    h = length / (nspace - 1)
    coeff = tau*1j

    x_grid = np.linspace(-length / 2, length / 2, nspace)
    t_grid = np.linspace(0, ntime * tau, ntime)

    # define amplification matrix
    A = np.zeros((nspace, nspace))
    method = method.lower()
    if method == "ftcs":
        A = make_tridiagonal(nspace, coeff, 1, -coeff)
        # periodic boundary conditions
        A[0, -1] = coeff
        A[-1, 0] = -coeff
    elif method == "crank":
        #A = make_tridiagonal(nspace, 1 / 2 + coeff, 0, 1 / 2 - coeff)
        # periodic boundary conditions
        #A[0, -1] = 1 / 2 + coeff
        #A[-1, 0] = 1 / 2 - coeff
    else:
        return

    # determine solution stability
    if spectral_radius(A) > 1:
        print("Warning: solution will be unstable")

    # initial conditions set by make_initialcond
    tt = make_initialcond(0.2, 35, x_grid)  # Initial cond. set by make_initialcond

    # loop over the desired number of time steps.
    ttplot = np.empty((nspace, ntime))
    for istep in range(ntime):  ## MAIN LOOP ##

        # compute new temperature using either method.
        tt = A.dot(tt.T)

        # record amplitude for plotting
        ttplot[:, istep] = np.copy(tt)  # record tt(i) for plotting

    return ttplot, x_grid, t_grid