import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#Copied from week 11 submission
class Animator:
    """
    I wrapped this in a class because it feels wrong otherwise, it just encapsulates all the animation variables and functions
    """
    def __init__(self, x_grid, title, x_label, y_label, y_bound=1):
        """
        constructor for the animator class
        :param x_grid: the x grid of the plot
        :param title: the title of the plot the class produces
        :param x_label: the x-axis label
        :param y_label: the y-axis label
        """
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], color='black')
        self.x_grid = x_grid
        self.lower_x_bound = np.min(x_grid)
        self.upper_x_bound = np.max(x_grid)
        self.lower_y_bound = -y_bound
        self.upper_y_bound = y_bound
        self.text = self.ax.text(self.lower_x_bound + 0.02*(self.upper_x_bound-self.lower_x_bound), self.upper_y_bound-0.04*(self.upper_y_bound-self.lower_y_bound), [])

    def init(self):
        """
        initializes the plot by setting its axes and title, called at the start of the animation
        :return: the initialized plot
        """
        self.ax.set_xlim(self.lower_x_bound, self.upper_x_bound)
        self.ax.set_ylim(self.lower_y_bound, self.upper_y_bound)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        return self.line,

    #data: (t, amplitude)
    def update(self, data):
        """
        draws each frame in the animation
        :param data: a tuple containing two elements, the time, and the amplitude of the wave at each point on the x-axis
        :return: the plot of the drawn frame
        """
        self.line.set_data(self.x_grid, data[1])
        self.text.set_text(f"t = {data[0]}s")
        return self.text,

    def animate(self, time, amplitude):
        """
        constructs the data array from time and theta arrays. Also creates the animation
        :param time: the time points of the simulation
        :param amplitude: a 2d array of the amplitudes of the wave at each time
        :return: the finished animation
        """
        return FuncAnimation(self.fig, self.update, frames=[(time[i], amplitude[:, i]) for i in range(len(time))], init_func=self.init, interval=1)

#adapted from my week 11 make_tridagonal
def make_H(N, b, d, a, V):
    """
    Makes a tridiagonal NxN matrix with the elements b, d, and a from bottom left to top right including wrap around for periodic boundary conditions
    :param N: the size of the matrix to create
    :param b: the value of the elements immediately to the left of the main diagonal (including [0,-1])
    :param d: the values on the main diagonal
    :param a: the value of the elements immediately to the right of the main diagonal (including [-1,0])
    :param V: the points in space to set the potential to 1
    :return: the matrix H (the periodic discrete hamiltonian)
    """
    H = d*np.eye(N)+b*np.eye(N, k=-1)+a*np.eye(N, k=1)
    H[0, -1] = b
    H[-1, 0] = a
    for i in V:
        H[i, i] += 1
    return H

#copied from my week 11 submission
def make_initialcond(sigma_0, x_0, k_0, spatial_grid):
    """
    Creates the initial conditions as a wave packet with a form given in the project 4 description
    :param sigma_0: the packet width of the wave packet
    :param x_0 the position at which the wave packet is centered on
    :param k_0: the average wave number of the wave packet
    :param spatial_grid: the grid of x values to set the initial conditions of
    :return: an array of amplitudes for each position given by the spatial_grid parameter
    """
    return (1/(np.sqrt(sigma_0*np.sqrt(np.pi))))*np.exp(1j*k_0*spatial_grid)*np.exp((-(spatial_grid-x_0)**2)/(2*sigma_0**2))

#copied from my week 11 submission
def spectral_radius(A):
    """
    Calculates the eigenvalue of the matrix A with greatest absolute magnitude
    :param A: the matrix to find the eigenvalue with the greatest absolute magnitude of
    :return: the eigenvalue with the greatest absolute magnitude
    """
    eig_vals = np.linalg.eig(A)[0]
    return np.max(np.abs(eig_vals))

#adapted from my week 11 advection1d function
def sch_eqn(nspace, ntime, tau, method="ftcs", length = 200, potential = [], wparam=[10, 0, 0.5]):
    """
        Solves the 1d schrodinger equation
        :param nspace: the number of points in the spatial grid
        :param ntime: the number of time steps
        :param tau: the time step to use in the solution
        :param method: the solution method to use
        :param length: the length of the spatial grid (will range from -length to length)
        :param potential: a 1D array giving the index values to set the potential to 1 at
        :param wparam: a tuple of parameters to set the initial condition (sigma_0, x_0, k_0)
        :return: a tuple (a, x, t, p) where a is a 2d array of solutions for all x at each time, x is a 1d array of the points on the spatial grid, t is a 1d array of the times which were solved for, and p is the total probability at each time step. Returns None if solution fails.
        """
    # define variables
    sigma_0 = wparam[0]
    x_0 = wparam[1]
    k_0 = wparam[2]
    h = length / (nspace - 1)

    #define the spacial and temporal grids
    x_grid = np.linspace(-length/2, length/2, nspace)
    t_grid = np.linspace(0, ntime * tau, ntime)

    # define amplification matrix
    A = np.zeros((nspace, nspace))
    #define the discrete hamiltonian matrix (see eqn 9.31 NM4P) with h bar = 1, m = 1/2
    H = make_H(nspace, -1/h**2, 2/h**2, -1/h**2, potential)

    if method == "ftcs":
        #see the matrix in eqn 9.32 NM4P with h bar = 1
        coeff = 1j*tau
        A = np.eye(nspace)-(coeff*H)
        # determine solution stability
        if spectral_radius(A)-1 > 1e-1:
            print("Warning: solution will be unstable")
            return None

    elif method == "crank":
        #see the matrix in eqn 9.40 NM4P with h bar = 1
        coeff = (1j*tau)/2
        A = np.matmul(np.linalg.inv(np.eye(nspace) + coeff*H), np.eye(nspace) - coeff*H)
    else:
        return None

    # initial conditions set by make_initialcond
    tt = make_initialcond(sigma_0, x_0, k_0, x_grid)  # Initial cond. set by make_initialcond

    # loop over the desired number of time steps.
    ttplot = np.empty((nspace, ntime), dtype=complex)
    ttplot[:, 0] = np.copy(tt)
    for istep in range(1, ntime):

        # compute the next time step using either method.
        tt = A.dot(tt.T)

        # record amplitude for plotting
        ttplot[:, istep] = np.copy(tt)  # record tt(i) for plotting

    total_prob = np.zeros((ntime), dtype=complex)
    total_prob[:] = np.sum(ttplot*np.conjugate(ttplot), axis=0)

    return ttplot, x_grid, t_grid, total_prob

def sch_plot(ttplot, x_grid, t_grid, t=0, graph="psi", file="", animate=False):
    """
    Plots solutions to the schrodinger equation
    :param ttplot: the 2d array of amplitudes at each time
    :param x_grid: the 1d array of x values for the domain
    :param t_grid: the 1d array of time values for the range
    :param t: the time to plot the amplitude at (ignored if animate is set to True)
    :param graph: the type of graph to produce (psi for the real part of psi, prob for the probability density of phi)
    :param file: the file name to save the plot or animation to (won't save if left blank)
    :param animate: whether to animate the function (t will be ignored if set to True)
    :return: None
    """

    #create animation
    if animate:

        #set parameters based on what is being animated
        if graph == "psi":
            ylabel = "$\\psi(x, t)$"
            title = "Animation of a Solution to Schrodinger's Equation"
            y_bound = np.max(np.real(ttplot))
        elif graph == "prob":
            ylabel = "$\\psi(x, t={t_grid[ind]})\\psi^*(x, t={t_grid[ind]})$"
            title = "Animation of the Probability Density of a Solution to Schrodinger's Equation"
            y_bound = np.max(np.real(ttplot*np.conjugate(ttplot)))
        else:
            return None

        #initialize animator
        animator = Animator(x_grid, title, "Position", ylabel, y_bound=y_bound)

        #create the animation
        if graph == "psi":
            animation = animator.animate(t_grid, np.real(ttplot))
        elif graph == "prob":
            animation = animator.animate(t_grid, np.real(ttplot*np.conjugate(ttplot)))

        #save the animation
        if file != "":
            if file.split(".")[-1] != "gif":
                file += ".gif"
            writer = PillowWriter(fps=20)
            animation.save(file, writer)

    else: #plot at specific time
        #find the index of the time to plot
        ind = np.searchsorted(t_grid, t)
        if ind >= len(t_grid) or abs(t_grid[ind-1] - t) < abs(t_grid[ind] - t):
            ind -= 1

        #create graph at time t
        fig, ax = plt.subplots()
        ax.set_xlabel("Position")
        if graph == "psi":
            ax.set_title(f"Solution of Schrodinger's Equation at t={t_grid[ind]}")
            ax.set_ylabel(f"$\\psi(x, t={t_grid[ind]})$")
            ax.plot(x_grid, np.real(ttplot[:, ind]))
        elif graph == "prob":
            ax.set_title(f"Probability Density for a Solution of Schrodinger's Equation at t={t_grid[ind]}")
            ax.set_ylabel(f"$\\psi(x, t={t_grid[ind]})\\psi^*(x, t={t_grid[ind]})$")
            ax.plot(x_grid, np.real(ttplot[:, ind]*np.conjugate(ttplot[:, ind])))

        #display and save plot
        fig.show()
        if file != "":
            if file.split(".")[-1] != "png":
                file += ".png"
            fig.savefig(file)

def test():
    """
    A function for testing the sch_eqn, and sch_plot
    """
    eqn = sch_eqn(200, 500, 0.1, length = 200, method="crank", potential=[0, -1])
    if eqn is not None:
        ttplot, x_grid, t_grid, prob = eqn

        #check that probability is conserved
        if (all([np.abs(i - prob[0]) < 1e-4 for i in prob])):
            print("Probability is conserved")
        else:
            print("Probability is not conserved")

        #test initial condition (should roughly recreate Fig 9.5 in NM4P)
        sch_plot(ttplot, x_grid, t_grid, 0, file="Schultz_Liam_Fig_9.5_crank.png")

        #test probability density (should look like one of the lines from Fig 9.6 in NM4P)
        sch_plot(ttplot, x_grid, t_grid, 0, graph="prob", file="Schultz_Liam_Fig_9.6_crank")

        #test psi animation to observe transmission and reflection of the wave
        sch_plot(ttplot, x_grid, t_grid, graph="psi", file="Schultz_Liam_Psi_Animation_crank.gif", animate=True)

        #same with probability density
        sch_plot(ttplot, x_grid, t_grid, graph="prob", file="Schultz_Liam_Prob_Animation_crank", animate=True)

#test()
#Code hosted at: https://github.com/liam-schultz/CompSims_Project4