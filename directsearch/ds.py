"""
A generic class of direct search methods

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

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

from math import sqrt
import numpy as np

from .sketcher import sketch_matrix, check_valid_sketch_method

__all__ = ['ds', 'DEFAULT_PARAMS', 'EXIT_ALPHA_MIN_REACHED', 'EXIT_MAXFUN_REACHED']

VALID_POLL_TYPES = ['2n', 'np1', 'random2', 'random_ortho']

# Default choices for all parameters (to be consistent throughout)
DEFAULT_PARAMS = {}
DEFAULT_PARAMS['rho'] = None
DEFAULT_PARAMS['sketch_dim'] = None
DEFAULT_PARAMS['sketch_type'] = 'gaussian'
DEFAULT_PARAMS['maxevals'] = None
DEFAULT_PARAMS['poll_type'] = '2n'
DEFAULT_PARAMS['alpha0'] = None
DEFAULT_PARAMS['alpha_max'] = 1e3
DEFAULT_PARAMS['alpha_min'] = 1e-6
DEFAULT_PARAMS['gamma_inc'] = 2.0
DEFAULT_PARAMS['gamma_dec'] = 0.5
DEFAULT_PARAMS['verbose'] = False
DEFAULT_PARAMS['print_freq'] = None
DEFAULT_PARAMS['use_stochastic_three_points'] = False
DEFAULT_PARAMS['poll_scale_prob'] = 0.0
DEFAULT_PARAMS['poll_scale_factor'] = 1.0
DEFAULT_PARAMS['rho_uses_normd'] = True

# Exit flags
EXIT_ALPHA_MIN_REACHED = 0  # alpha <= alpha_min
EXIT_MAXFUN_REACHED = 1     # budget reached


def poll_directions(n, poll_type=DEFAULT_PARAMS['poll_type'], scale_prob=DEFAULT_PARAMS['poll_scale_prob'], scale_factor=DEFAULT_PARAMS['poll_scale_factor']):
    # Build a PSS for R^n of different types. Output is a matrix where each column is a search direction
    # With some probability, increase the length of the directions by some factor
    if poll_type == '2n':  # +/-I
        I = np.eye(n)
        D = np.hstack([I, -I])
    elif poll_type == 'np1':  # I and -e
        I = np.eye(n)
        neg_e = -np.ones((n,1))  # need shape (n,1) not (n,) to work with np.append
        D = np.append(I, neg_e, axis=1)
    elif poll_type == 'random2':  # +/- random unit vector
        a = np.random.normal(size=(n,1))
        a = a / np.linalg.norm(a)
        D = np.hstack([a, -a])
    elif poll_type.startswith('random_ortho'):  # +/- random orthonormal directions, e.g. random_ortho5
        ndirs = int(poll_type.replace('random_ortho', ''))
        A = np.random.normal(size=(n,ndirs))
        Q = np.linalg.qr(A, mode='reduced')[0]  # random orthonormal set
        D = np.hstack([Q, -Q])
    else:
        raise RuntimeError("Unknown poll type: %s" % poll_type)

    if float(np.random.rand(1)) <= scale_prob:
        return scale_factor * D
        # return np.hstack([scale_factor * D, D])
        # a = np.random.normal(size=(n, 1))
        # a = scale_factor * a / np.linalg.norm(a)
        # D2 = np.hstack([a, -a])
        # return np.hstack([D2, D])
    else:
        return D


def poll_set(n, sketch_dim=DEFAULT_PARAMS['sketch_dim'], sketch_type=DEFAULT_PARAMS['sketch_type'], poll_type=DEFAULT_PARAMS['poll_type'], scale_prob=DEFAULT_PARAMS['poll_scale_prob'], scale_factor=DEFAULT_PARAMS['poll_scale_factor']):
    # Build sketched PSS. Ambient dimension is n, sketch dimension is sketch_dim
    # TODO make into generator to save memory when n is large
    if sketch_dim is None:
        Dk = poll_directions(n, poll_type=poll_type, scale_prob=scale_prob, scale_factor=scale_factor)
    else:
        Pk = sketch_matrix(sketch_dim, n, sketch_method=sketch_type)
        Dk = Pk.T @ poll_directions(sketch_dim, poll_type=poll_type, scale_prob=scale_prob, scale_factor=scale_factor)
    return Dk


def ds(f, x0, rho=DEFAULT_PARAMS['rho'], sketch_dim=DEFAULT_PARAMS['sketch_dim'], sketch_type=DEFAULT_PARAMS['sketch_type'], maxevals=DEFAULT_PARAMS['maxevals'], poll_type=DEFAULT_PARAMS['poll_type'], alpha0=DEFAULT_PARAMS['alpha0'], alpha_max=DEFAULT_PARAMS['alpha_max'], alpha_min=DEFAULT_PARAMS['alpha_min'],
       gamma_inc=DEFAULT_PARAMS['gamma_inc'], gamma_dec=DEFAULT_PARAMS['gamma_dec'], verbose=DEFAULT_PARAMS['verbose'], print_freq=DEFAULT_PARAMS['print_freq'], use_stochastic_three_points=DEFAULT_PARAMS['use_stochastic_three_points'], poll_scale_prob=DEFAULT_PARAMS['poll_scale_prob'], poll_scale_factor=DEFAULT_PARAMS['poll_scale_factor'], rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    # Set some sensible defaults for: sufficient decrease threshold, # evaluations, initial step size
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
    
    if maxevals is None:
        maxevals = min(100 * (n + 1), 1000)
    maxevals = int(maxevals)
    
    if alpha0 is None:
        alpha0 = 0.1 * max(np.max(np.abs(x0)), 1.0)
        alpha0 = max(min(alpha0, alpha_max), alpha_min)  # force alpha0 to be in [alpha_min, alpha_max]
    alpha0 = float(alpha0)
    
    if print_freq is None:
        print_freq = max(int(maxevals // 20), 1)
    print_freq = int(print_freq)
    
    alpha_max = float(alpha_max)
    alpha_min = float(alpha_min)
    gamma_inc = float(gamma_inc)
    gamma_dec = float(gamma_dec)
    poll_scale_prob = float(poll_scale_prob)
    poll_scale_factor = float(poll_scale_factor)
    verbose = bool(verbose)
    use_stochastic_three_points = bool(use_stochastic_three_points)
    
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

    # Begin main method
    fx = f(x)
    nf = 1
    if nf >= maxevals:
        if verbose:
            print("Quit (max evals)")
        return x, fx, nf, EXIT_MAXFUN_REACHED

    # Main loop
    alpha = alpha0
    k = -1
    if verbose:
        print(f"{'k':^5}{'f(xk)':^15}{'alpha_k':^15}")
    while nf < maxevals:
        k += 1
        if verbose and k % print_freq == 0:
            print(f"{k:^5}{fx:^15.4e}{alpha:^15.2e}")

        # Generate poll directions
        Dk = poll_set(n, sketch_dim=sketch_dim, poll_type=poll_type, sketch_type=sketch_type, scale_prob=poll_scale_prob, scale_factor=poll_scale_factor)

        # Start poll step
        ndirs = Dk.shape[1]

        if use_stochastic_three_points:
            # STP method
            alpha = alpha0 / sqrt(k + 1)
            for j in range(ndirs):
                dj = Dk[:,j]
                xnew = x + alpha * dj
                fnew = f(xnew)
                nf += 1

                if fnew < fx:
                    x = xnew.copy()
                    fx = fnew

                if nf >= maxevals:
                    if verbose:
                        print(f"{k:^5}{fx:^15.4e}{alpha:^15.2e} - max evals reached")
                    return x, fx, nf, EXIT_MAXFUN_REACHED

            if alpha < alpha_min:
                if verbose:
                    print(f"{k:^5}{fx:^15.4e}{alpha:^15.2e} - small alpha reached")
                break  # finish algorithm

            # end STP method
        else:
            # Regular direct search
            polling_successful = False
            for j in range(ndirs):
                dj = Dk[:,j]
                xnew = x + alpha * dj
                fnew = f(xnew)
                nf += 1
                sufficient_decrease = (fnew < fx - rho_to_use(alpha, np.linalg.norm(dj))) if rho_uses_normd else (fnew < fx - rho_to_use(alpha))

                # Quit on budget (update to xnew if we just saw an improvement)
                if nf >= maxevals:
                    if sufficient_decrease:
                        x = xnew.copy()
                        fx = fnew
                    if verbose:
                        print(f"{k:^5}{fx:^15.4e}{alpha:^15.2e} - max evals reached")
                    return x, fx, nf, EXIT_MAXFUN_REACHED

                # If sufficient decrease, update xk and stop poll step
                if sufficient_decrease:
                    x = xnew.copy()
                    fx = fnew
                    alpha = min(gamma_inc * alpha, alpha_max)
                    polling_successful = True
                    break  # stop poll step, go to next iteration

            # If here, no decrease found
            if alpha < alpha_min:
                if verbose:
                    print(f"{k:^5}{fx:^15.4e}{alpha:^15.2e} - small alpha reached")
                break  # finish algorithm

            if not polling_successful:
                alpha = gamma_dec * alpha
            # End direct search method
        # End loop

    return x, fx, nf, EXIT_ALPHA_MIN_REACHED
