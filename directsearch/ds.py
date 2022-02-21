"""
Implementation of a generic class of direct search methods.
This module contains the following Python functions

poll_directions: Generate a set of polling directions as in classical direct
search.
poll_set: Generate a sketched set of polling directions.
ds: Main code, used to run a direct-search method.

===============================================================================
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

###############################################################################
# Preliminary imports

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

# Math/NumPy imports
from math import sqrt
import numpy as np

# Local imports
from .sketcher import sketch_matrix, check_valid_sketch_method

###############################################################################
# Useful global variables parameter values

# Variables to be exported 
__all__ = ['ds', 'DEFAULT_PARAMS', 'EXIT_ALPHA_MIN_REACHED', 'EXIT_MAXFUN_REACHED']

# List of the available polling types
#   '2n': Coordinates vectors and their negatives
#   'n+1': Coordinate vectors and the sum of their negatives
#   'random2': A random unit vector and its negative
#   'random_ortho': Random set of orthogonal directions
VALID_POLL_TYPES = ['2n', 'np1', 'random2', 'random_ortho']

# Default choices for all parameters (to be consistent throughout)
DEFAULT_PARAMS = {}
DEFAULT_PARAMS['rho'] = None # Forcing function
DEFAULT_PARAMS['sketch_dim'] = None # Target dimension for sketching
DEFAULT_PARAMS['sketch_type'] = 'gaussian' # Sketching technique
DEFAULT_PARAMS['maxevals'] = None # Maximum number of function evaluations
DEFAULT_PARAMS['poll_type'] = '2n' # Polling direction type
DEFAULT_PARAMS['alpha0'] = None # Original stepsize value
DEFAULT_PARAMS['alpha_max'] = 1e3 # Maximum value for the stepsize
DEFAULT_PARAMS['alpha_min'] = 1e-6 # Minimum value for the stepsize
DEFAULT_PARAMS['gamma_inc'] = 2.0 # Increasing factor for the stepsize
DEFAULT_PARAMS['gamma_dec'] = 0.5 # Decreasing factor for the stepsize
DEFAULT_PARAMS['verbose'] = False # Display information about the method
DEFAULT_PARAMS['print_freq'] = None # How frequently to display information
DEFAULT_PARAMS['use_stochastic_three_points'] = False # Boolean for a specific method
DEFAULT_PARAMS['poll_scale_prob'] = 0.0 # Probabilty of direction scaling 
DEFAULT_PARAMS['poll_scale_factor'] = 1.0 # Scaling factor for direction norms
DEFAULT_PARAMS['rho_uses_normd'] = True # Forcing function based on direction norm

# Exit flags
EXIT_ALPHA_MIN_REACHED = 0  # alpha <= alpha_min
EXIT_MAXFUN_REACHED = 1     # budget reached

###############################################################################
def poll_directions(n, poll_type=DEFAULT_PARAMS['poll_type'], scale_prob=DEFAULT_PARAMS['poll_scale_prob'], scale_factor=DEFAULT_PARAMS['poll_scale_factor']):
    """
        Produces a matrix of polling directions.

        poll_directions(n) returns a matrix. Every column of that matrix is a 
        vector in R^n, to be used as a polling direction.

        Inputs:
            n: Problem dimension. All computed directions will be vectors in 
            R^n
            poll_type: Type of polling directions produced by the method.
                Accepted values: See VALID_POLL_TYPES
                Default: see DEFAULT_PARAMS['poll_type']
            scale_prob: Probability of scaling the polling directions.
                Default: See DEFAULT_PARAMS['poll_scale_prob']
            scale_factor: Factor used to scale the polling directions.
                Default: See DEFAULT_PARAMS['poll_scale_factor']

        Output:
            D: A matrix with n rows. Its columns represent the poll directions.
    """

    # Generate the directions according to poll_type (see description of
    # VALID_POLL_TYPEs).
    if poll_type == '2n': # +/- Identity
        I = np.eye(n)
        D = np.hstack([I, -I])
    elif poll_type == 'np1':  # Identity and -e
        I = np.eye(n)
        neg_e = -np.ones((n,1))  # need shape (n,1) not (n,) to work with np.append
        D = np.append(I, neg_e, axis=1)
    elif poll_type == 'random2':  # +/- random unit vector
        a = np.random.normal(size=(n,1))
        a = a / np.linalg.norm(a)
        D = np.hstack([a, -a])
    elif poll_type.startswith('random_ortho'):  # +/- random orthonormal directions
        ndirs = int(poll_type.replace('random_ortho', ''))
        A = np.random.normal(size=(n,ndirs))
        Q = np.linalg.qr(A, mode='reduced')[0]  # random orthonormal set
        D = np.hstack([Q, -Q])
    else:
        raise RuntimeError("Unknown poll type: %s" % poll_type)

    # Scale the directions with some probability
    if float(np.random.rand(1)) <= scale_prob:
        return scale_factor * D
        # return np.hstack([scale_factor * D, D])
        # a = np.random.normal(size=(n, 1))
        # a = scale_factor * a / np.linalg.norm(a)
        # D2 = np.hstack([a, -a])
        # return np.hstack([D2, D])
    else:
        return D
###############################################################################

###############################################################################
def poll_set(n, sketch_dim=DEFAULT_PARAMS['sketch_dim'], sketch_type=DEFAULT_PARAMS['sketch_type'], poll_type=DEFAULT_PARAMS['poll_type'], scale_prob=DEFAULT_PARAMS['poll_scale_prob'], scale_factor=DEFAULT_PARAMS['poll_scale_factor']):
    """
        Build a sketch of a polling set.

        poll_set(n) generates a sketched version of a polling set.

        Inputs:
            n: Problem dimension
            sketch_dim: Reduced dimension to generate polling directions in.
                Default: see DEFAULT_PARAMS['sketch_dim']
            sketch_type: Sketching technique to be used.
                Default: see DEFAULT_PARAMS['sketch_type']
            poll_type: Type of polling directions generated in the reduced spaces.
                Accepted values: See VALID_POLL_TYPES
                Default: see DEFAULT_PARAMS['poll_type']
            scale_prob: Probability of scaling the polling directions.
                Default: See DEFAULT_PARAMS['poll_scale_prob']
            scale_factor: Factor used to scale the polling directions.
                Default: See DEFAULT_PARAMS['poll_scale_factor']

        Output
            Dk: A sketched polling set built from combining a sketching matrix
            with a polling set in the reduced space. 
    """
    # TODO make into generator to save memory when n is large
    if sketch_dim is None:
        # No sketching - Simple call to poll_directions
        Dk = poll_directions(n, poll_type=poll_type, scale_prob=scale_prob, scale_factor=scale_factor)
    else:
        # Generate a sketching matrix and a polling set in the reduced space
        Pk = sketch_matrix(sketch_dim, n, sketch_method=sketch_type)
        # Note: Pk.T @ poll_directions doesn't work in Python 2.7, 
        # Using np.dot(...) for backwards compatibility
        Dk = np.dot(Pk.T, poll_directions(sketch_dim, poll_type=poll_type, scale_prob=scale_prob, scale_factor=scale_factor))
    return Dk
###############################################################################

###############################################################################
def ds(f, x0, rho=DEFAULT_PARAMS['rho'], sketch_dim=DEFAULT_PARAMS['sketch_dim'], sketch_type=DEFAULT_PARAMS['sketch_type'], maxevals=DEFAULT_PARAMS['maxevals'], poll_type=DEFAULT_PARAMS['poll_type'], alpha0=DEFAULT_PARAMS['alpha0'], alpha_max=DEFAULT_PARAMS['alpha_max'], alpha_min=DEFAULT_PARAMS['alpha_min'],
       gamma_inc=DEFAULT_PARAMS['gamma_inc'], gamma_dec=DEFAULT_PARAMS['gamma_dec'], verbose=DEFAULT_PARAMS['verbose'], print_freq=DEFAULT_PARAMS['print_freq'], use_stochastic_three_points=DEFAULT_PARAMS['use_stochastic_three_points'], poll_scale_prob=DEFAULT_PARAMS['poll_scale_prob'], poll_scale_factor=DEFAULT_PARAMS['poll_scale_factor'], rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    """
        A generic direct-search code based on reduced subspaces.

        ds(f,x0) attempts to minimize the function f starting at x0 using a 
        direct-search approach. The method is based on moves along certain 
        polling directions: these moves are controlled by means of an 
        adaptive stepsize.

        Inputs:
            f: Function handle for the objective to be minimized.
            x0: Initial point.
            rho: Choice of the forcing function.
                Default: see DEFAULT_PARAMS['rho']
            sketch_dim: Reduced dimension to generate polling directions in.
                Default: see DEFAULT_PARAMS['sketch_dim']
            sketch_type: Sketching technique to be used.
                Default: see DEFAULT_PARAMS['sketch_type']
            maxevals: Maximum number of calls to f performed by the algorithm.
                Default: see DEFAULT_PARAMS['maxevals']
            poll_type: Type of polling directions generated in the reduced spaces.
                Accepted values: See VALID_POLL_TYPES
                Default: see DEFAULT_PARAMS['poll_type']
            alpha0: Initial value for the stepsize parameter.
                Default: See DEFAULT_PARAMS['alpha0']
            alpha_max: Maximum value for the stepsize parameter.
                Default: See DEFAULT_PARAMS['alpha_max']
            alpha_min: Minimum value for the stepsize parameter.
                Default: See DEFAULT_PARAMS['alpha_min']
            gamma_inc: Increase factor for the stepsize update.
                Default: See DEFAULT_PARAMS['gamma_inc']
            gamma_dec: Decrease factor for the stepsize update.
                Default: See DEFAULT_PARAMS['gamma_dec']
            verbose: Boolean indicating whether information should be displayed 
            during an algorithmic run.
                Default: See DEFAULT_PARAMS['verbose']
            print_freq: Value indicating how frequently information should
            be displayed.
                Default: See DEFAULT_PARAMS['print_freq']
            use_stochastic_three_points: Boolean indicating whether the
            specific stochastic three points method should be used.
                Default: See DEFAULT_PARAMS['use_stochastic_three_points']
            poll_scale_prob: Probability of scaling the polling directions.
                Default: See DEFAULT_PARAMS['poll_scale_prob']
            poll_scale_factor: Factor used to scale the polling directions.
                Default: See DEFAULT_PARAMS['poll_scale_factor']
            rho_uses_normd: Boolean indicating whether the forcing function should 
            account for the norm of the direction.
                Default: See DEFAULT_PARAMS['rho_uses_normd']

        Outputs:
            x: Best solution found (vector of same dimension than x0).
            fx: Value of f at x.
            nf: Number of function evaluations that have been used.
            stopping flag: Indicator of the reason why the method stopped.
                EXIT_MAXFUN_REACHED: The maximum number of function evaluations 
                was reached.
                EXIT_ALPHA_MIN_REACHED: The stepsize reached the minimum 
                allowed value.

    """

    ###############
    # Initialization
    # Set some sensible defaults for: sufficient decrease threshold, # evaluations, initial step size
    # Set the forcing function
    rho_uses_normd = bool(rho_uses_normd)
    if rho is None:
        if rho_uses_normd:
            rho_to_use = lambda t, normd: min(1e-5, 1e-5 * (t * normd) ** 2)
        else:
            rho_to_use = lambda t: 1e-5 * t**2
    else:
        rho_to_use = rho
    
    # Force correct types
    x = np.array(x0, dtype=float)
    n = len(x)
    x = x.reshape((n,))
    alpha_max = float(alpha_max)
    alpha_min = float(alpha_min)
    gamma_inc = float(gamma_inc)
    gamma_dec = float(gamma_dec)
    poll_scale_prob = float(poll_scale_prob)
    poll_scale_factor = float(poll_scale_factor)
    verbose = bool(verbose)
    use_stochastic_three_points = bool(use_stochastic_three_points)
    
    # Compute the maximum number of evaluations according to the problem dimension
    if maxevals is None:
        maxevals = min(100 * (n + 1), 1000)
    maxevals = int(maxevals)
    
    # Set initial stepsize so that it satisfies the bounds
    # alpha_min <= alpha_0 <= alpha_max
    if alpha0 is None:
        alpha0 = 0.1 * max(np.max(np.abs(x0)), 1.0)
        alpha0 = max(min(alpha0, alpha_max), alpha_min) 
    alpha0 = float(alpha0)
   
    # Set frequence of information display 
    if print_freq is None:
        print_freq = max(int(maxevals // 20), 1)
    print_freq = int(print_freq)
    
    
    # Input checking
    assert callable(f), "Objective function should be callable"
    assert callable(rho_to_use), "Sufficient decrease function rho should be callable"
    assert maxevals > 0, "maxevals should be strictly positive"
    assert poll_type in VALID_POLL_TYPES, "Invalid poll_type '%s'" % (poll_type)
    assert alpha_max > 0.0, "alpha_max should be strictly positive"
    assert alpha0 > 0.0, "alpha0 should be strictly positive"
    assert alpha_min > 0.0, "alpha_max should be strictly positive"
    assert alpha_min <= alpha_max, "alpha_min should be <= alpha_max"
    assert alpha0 >= alpha_min, "alpha0 should be >= alpha_min"
    if not use_stochastic_three_points:  # STP never increases alpha above alpha0
        assert alpha0 <= alpha_max, "alpha0 should be <= alpha_max"
    assert gamma_inc >= 1.0, "gamma_inc should be at least 1"
    assert gamma_dec > 0.0, "gamma_dec should be strictly positive"
    assert gamma_dec < 1.0, "gamma_dec should be strictly < 1"
    assert poll_scale_prob >= 0.0, "poll_scale_prob should be non-negative"
    assert poll_scale_prob <= 1.0, "poll_scale_prob should be <= 1"
    assert poll_scale_factor > 0.0, "poll_scale_factor should be strictly positive"
    if verbose:
        assert print_freq > 0, "print_freq should be strictly positive"
    
    if use_stochastic_three_points:
        assert sketch_dim is None, "No sketch dimension needed for STP"
        assert poll_type == 'random2', "STP needs 'random2' poll type, got '%s'" % poll_type
    
    if sketch_dim is not None:
        assert check_valid_sketch_method(sketch_type), "Invalid sketch_type '%s'" % (sketch_type)
        assert sketch_dim > 0, "sketch_dim should be strictly positive"
        assert sketch_dim <= n, "sketch_dim should be <= problem dimension"

    ###################################
    # Start of the optimization process
    fx = f(x)
    nf = 1
    if nf >= maxevals:
        if verbose:
            print("Quit (max evals)")
        return x, fx, nf, EXIT_MAXFUN_REACHED

    ###############
    # Main loop
    alpha = alpha0
    k = -1
    if verbose:
        print("{0:^5}{1:^15}{2:^15}".format('k', 'f(xk)', 'alpha_k'))
    while nf < maxevals:
        k += 1
        if verbose and k % print_freq == 0:
            print("{0:^5}{1:^15.4e}{2:^15.2e}".format(k, fx, alpha))

        # Generate poll directions
        Dk = poll_set(n, sketch_dim=sketch_dim, poll_type=poll_type, sketch_type=sketch_type, scale_prob=poll_scale_prob, scale_factor=poll_scale_factor)

        # Start poll step
        ndirs = Dk.shape[1]

        # Perform a direct-search iteration according to the chosen approach
        if use_stochastic_three_points:
            # STP method - Simple decrease, fixed stepsize sequence
            alpha = alpha0 / sqrt(k + 1)
            for j in range(ndirs):
                dj = Dk[:,j]
                xnew = x + alpha * dj
                fnew = f(xnew)
                nf += 1

                # Check for simple decrease
                if fnew < fx:
                    x = xnew.copy()
                    fx = fnew

                # Quit on budget
                if nf >= maxevals:
                    if verbose:
                        print("{0:^5}{1:^15.4e}{2:^15.2e} - max evals reached".format(k, fx, alpha))
                    return x, fx, nf, EXIT_MAXFUN_REACHED

            # Quit on small stepsize
            if alpha < alpha_min:
                if verbose:
                    print("{0:^5}{1:^15.4e}{2:^15.2e} - small alpha reached".format(k, fx, alpha))
                break  # finish algorithm

            # end STP method
        else:
            # Regular direct search - Sufficient decrease, adaptive stepsize
            polling_successful = False
            for j in range(ndirs):
                dj = Dk[:,j]
                xnew = x + alpha * dj
                fnew = f(xnew)
                nf += 1
                # Compute the target improvement
                sufficient_decrease = (fnew < fx - rho_to_use(alpha, np.linalg.norm(dj))) if rho_uses_normd else (fnew < fx - rho_to_use(alpha))

                # Quit on budget (update to xnew if we just saw an improvement)
                if nf >= maxevals:
                    if sufficient_decrease:
                        x = xnew.copy()
                        fx = fnew
                    if verbose:
                        print("{0:^5}{1:^15.4e}{2:^15.2e} - max evals reached".format(k, fx, alpha))
                    return x, fx, nf, EXIT_MAXFUN_REACHED

                # If sufficient decrease, update xk and stop poll step
                if sufficient_decrease:
                    x = xnew.copy()
                    fx = fnew
                    # Found a better point=> Possibly increase the stepsize
                    alpha = min(gamma_inc * alpha, alpha_max)
                    polling_successful = True
                    break  # stop poll step, go to next iteration

            # If here, no decrease found
            if alpha < alpha_min:
                if verbose:
                    print("{0:^5}{1:^15.4e}{2:^15.2e} - small alpha reached".format(k, fx, alpha))
                break  # finish algorithm 
                # Note - Could return here 

            if not polling_successful:
                # No better found point=> Decrease the stepsize
                alpha = gamma_dec * alpha
            # End regular direct search method
        # End loop
        ###########

    return x, fx, nf, EXIT_ALPHA_MIN_REACHED
