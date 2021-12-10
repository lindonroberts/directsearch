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

from .ds import ds, DEFAULT_PARAMS, EXIT_ALPHA_MIN_REACHED, EXIT_MAXFUN_REACHED

__all__ = ['solve', 'solve_directsearch', 'solve_probabilistic_directsearch', 'solve_subspace_directsearch', 'solve_stp']


OUTPUT_STRINGS = {EXIT_MAXFUN_REACHED:'Maximum evaluations reached', EXIT_ALPHA_MIN_REACHED:'alpha_min reached'}


class OptimResults(object):
    # Structure to hold solution information
    def __init__(self, xmin, fmin, nf, exit_flag):
        self.x = xmin
        self.f = fmin
        self.nf = nf
        self.flag = exit_flag
        self.msg = OUTPUT_STRINGS[exit_flag]
        # Set standard names for exit flags
        self.EXIT_MAXFUN_REACHED = EXIT_MAXFUN_REACHED
        self.EXIT_ALPHA_MIN_REACHED = EXIT_ALPHA_MIN_REACHED

    def __str__(self):
        # Result of calling print(soln)
        output = "****** directsearch results ******\n"
        if len(self.x) < 100:
            output += "Solution xmin = %s\n" % str(self.x)
        else:
            output += "Not showing xmin because it is too long; check self.x\n"
        output += "Objective value f(xmin) = %.10g\n" % self.f
        output += "Needed %g objective evaluations\n" % (self.nf)
        output += "Exit flag = %g\n" % self.flag
        output += "%s\n" % self.msg
        output += "**********************************\n"
        return output


def solve(f, x0, rho=DEFAULT_PARAMS['rho'], sketch_dim=DEFAULT_PARAMS['sketch_dim'],
          sketch_type=DEFAULT_PARAMS['sketch_type'], maxevals=DEFAULT_PARAMS['maxevals'],
          poll_type=DEFAULT_PARAMS['poll_type'], alpha0=DEFAULT_PARAMS['alpha0'],
          alpha_max=DEFAULT_PARAMS['alpha_max'], alpha_min=DEFAULT_PARAMS['alpha_min'],
          gamma_inc=DEFAULT_PARAMS['gamma_inc'], gamma_dec=DEFAULT_PARAMS['gamma_dec'],
          verbose=DEFAULT_PARAMS['verbose'], print_freq=DEFAULT_PARAMS['print_freq'],
          use_stochastic_three_points=DEFAULT_PARAMS['use_stochastic_three_points'],
          rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    # Generic solve interface with all functionality
    xmin, fmin, nf, flag =  ds(f, x0, rho=rho, sketch_dim=sketch_dim, sketch_type=sketch_type, maxevals=maxevals,
                               poll_type=poll_type, alpha0=alpha0, alpha_max=alpha_max, alpha_min=alpha_min,
                               gamma_inc=gamma_inc, gamma_dec=gamma_dec, verbose=verbose, print_freq=print_freq,
                               use_stochastic_three_points=use_stochastic_three_points, rho_uses_normd=rho_uses_normd)
    return OptimResults(xmin, fmin, nf, flag)


def solve_directsearch(f, x0, rho=DEFAULT_PARAMS['rho'], maxevals=DEFAULT_PARAMS['maxevals'],
                       poll_type=DEFAULT_PARAMS['poll_type'], alpha0=DEFAULT_PARAMS['alpha0'],
                       alpha_max=DEFAULT_PARAMS['alpha_max'], alpha_min=DEFAULT_PARAMS['alpha_min'],
                       gamma_inc=DEFAULT_PARAMS['gamma_inc'], gamma_dec=DEFAULT_PARAMS['gamma_dec'],
                       verbose=DEFAULT_PARAMS['verbose'], print_freq=DEFAULT_PARAMS['print_freq'],
                       rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    # Classical directsearch and/or probabilistic direct search
    return solve(f, x0, rho=rho, sketch_dim=None, maxevals=maxevals, poll_type=poll_type, alpha0=alpha0,
                 alpha_max=alpha_max, alpha_min=alpha_min, gamma_inc=gamma_inc, gamma_dec=gamma_dec, verbose=verbose,
                 print_freq=print_freq, use_stochastic_three_points=False, rho_uses_normd=rho_uses_normd)


def solve_probabilistic_directsearch(f, x0, rho=DEFAULT_PARAMS['rho'], maxevals=DEFAULT_PARAMS['maxevals'],
                                     alpha0=DEFAULT_PARAMS['alpha0'], alpha_max=DEFAULT_PARAMS['alpha_max'],
                                     alpha_min=DEFAULT_PARAMS['alpha_min'], gamma_inc=DEFAULT_PARAMS['gamma_inc'],
                                     gamma_dec=DEFAULT_PARAMS['gamma_dec'], verbose=DEFAULT_PARAMS['verbose'],
                                     print_freq=DEFAULT_PARAMS['print_freq'],
                                     rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    # Probabilistic direct search
    return solve(f, x0, rho=rho, sketch_dim=None, maxevals=maxevals, poll_type='random2', alpha0=alpha0,
                 alpha_max=alpha_max, alpha_min=alpha_min, gamma_inc=gamma_inc, gamma_dec=gamma_dec, verbose=verbose,
                 print_freq=print_freq, use_stochastic_three_points=False, rho_uses_normd=rho_uses_normd)


def solve_subspace_directsearch(f, x0, rho=DEFAULT_PARAMS['rho'], sketch_dim=DEFAULT_PARAMS['sketch_dim'],
                                sketch_type=DEFAULT_PARAMS['sketch_type'], maxevals=DEFAULT_PARAMS['maxevals'],
                                poll_type=DEFAULT_PARAMS['poll_type'], alpha0=DEFAULT_PARAMS['alpha0'],
                                alpha_max=DEFAULT_PARAMS['alpha_max'], alpha_min=DEFAULT_PARAMS['alpha_min'],
                                gamma_inc=DEFAULT_PARAMS['gamma_inc'], gamma_dec=DEFAULT_PARAMS['gamma_dec'],
                                verbose=DEFAULT_PARAMS['verbose'], print_freq=DEFAULT_PARAMS['print_freq'],
                                rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    # Direct search using random subspaces
    return solve(f, x0, rho=rho, sketch_dim=sketch_dim, sketch_type=sketch_type, maxevals=maxevals,
                 poll_type=poll_type, alpha0=alpha0, alpha_max=alpha_max, alpha_min=alpha_min, gamma_inc=gamma_inc,
                 gamma_dec=gamma_dec, verbose=verbose, print_freq=print_freq, use_stochastic_three_points=False,
                 rho_uses_normd=rho_uses_normd)


def solve_stp(f, x0, maxevals=DEFAULT_PARAMS['maxevals'], alpha0=DEFAULT_PARAMS['alpha0'],
              alpha_min=DEFAULT_PARAMS['alpha_min'], verbose=DEFAULT_PARAMS['verbose'],
              print_freq=DEFAULT_PARAMS['print_freq']):
    # STP method with stepsize sequence alpha = alpha0/(k+1)
    return solve(f, x0, sketch_dim=None, maxevals=maxevals, poll_type='random2', alpha0=alpha0, alpha_min=alpha_min,
                 verbose=verbose, print_freq=print_freq, use_stochastic_three_points=True)
