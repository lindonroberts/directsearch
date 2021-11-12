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

from .sketcher import sketch_matrix

__all__ = ['ds']


OUTPUT_STRINGS = {'BUDGET':'Maximum evaluations reached', 'SMALL_ALPHA':'alpha_min reached'}


def poll_directions(n, poll_type='2n', scale_prob=0.0, scale_factor=1.0):
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


def poll_set(n, sketch_dim=None, sketch_type='gaussian', poll_type='2n', scale_prob=0.0, scale_factor=1.0):
    # Build sketched PSS. Ambient dimension is n, sketch dimension is sketch_dim
    # TODO make into generator to save memory when n is large
    if sketch_dim is None:
        Dk = poll_directions(n, poll_type=poll_type, scale_prob=scale_prob, scale_factor=scale_factor)
    else:
        Pk = sketch_matrix(sketch_dim, n, sketch_method=sketch_type)
        Dk = Pk.T @ poll_directions(sketch_dim, poll_type=poll_type, scale_prob=scale_prob, scale_factor=scale_factor)
    return Dk


def ds(f, x0, rho=None, sketch_dim=None, sketch_type='gaussian', maxevals=None, poll_type='2n', alpha0=None, alpha_max=1e3, alpha_min=1e-6,
       gamma_inc=2.0, gamma_dec=0.5, verbose=False, print_freq=None, use_stochastic_three_points=False, poll_scale_prob=0.0, poll_scale_factor=1.0,
       rho_uses_normd=True):
    # Set some sensible defaults for: sufficient decrease threshold, # evaluations, initial step size
    if rho is None:
        if rho_uses_normd:
            rho_to_use = lambda t, normd: min(1e-5, 1e-5 * (t * normd) ** 2)
        else:
            rho_to_use = lambda t: 1e-5 * t**2
    else:
        rho_to_use = rho

    if maxevals is None:
        maxevals = min(100 * (len(x0) + 1), 1000)
    if alpha0 is None:
        alpha0 = 0.1 * max(np.max(np.abs(x0)), 1.0)
    if print_freq is None:
        print_freq = max(int(maxevals // 20), 1)
    if use_stochastic_three_points:
        assert sketch_dim is None, "No sketch dimension needed for STP"
        assert poll_type == 'random2', "STP needs 'random2' poll type, got '%s'" % poll_type
    n = len(x0)
    x = x0.copy()
    fx = f(x0)
    nf = 1
    if nf >= maxevals:
        if verbose:
            print("Quit (max evals)")
        return x, fx, nf, OUTPUT_STRINGS['BUDGET']

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
                    return x, fx, nf, OUTPUT_STRINGS['BUDGET']

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
                    return x, fx, nf, OUTPUT_STRINGS['BUDGET']

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

    return x, fx, nf, OUTPUT_STRINGS['SMALL_ALPHA']

