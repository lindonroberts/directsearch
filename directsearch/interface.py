"""
An interface for several derivative-free algorihms of direct-search type.

This module defines:

- A class OptimResults for the output of a direct-search algorithmic run.

- A wrapper solve() function to apply a direct-search method to a given
optimization problem.

- A wrapper solve_directsearch() to apply regular direct-search techniques 
without sketching.

- A wrapper solve_probabilistic_directsearch() to apply direct search based 
on probabilistic descent without sketching.

- A wrapper solve_subspace_directsearch() to apply direct-search schemes based 
on polling directions in random subspaces.

- A wrapper solve_stp() to apply the stochastic three
points method, a particular direct-search technique.

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

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

# Import key elements from the ds module
from .ds import ds, DEFAULT_PARAMS, EXIT_ALPHA_MIN_REACHED, EXIT_MAXFUN_REACHED
from .lincons import ds_lincons

# Global variables
__all__ = ['solve', 'solve_directsearch', 'solve_probabilistic_directsearch', 'solve_subspace_directsearch', 'solve_stp']

OUTPUT_STRINGS = {EXIT_MAXFUN_REACHED:'Maximum evaluations reached', EXIT_ALPHA_MIN_REACHED:'alpha_min reached'}

# Optimization via direct-search methods
class OptimResults(object):
    """
        A class to encode the results of a direct-search procedure.

        Attributes
            x: Best solution found.
            f: Best function value found.
            nf: Number of function values used udring the optimization process.
            flag: Exit flag of the procedure.
            msg: String corresponding to the exit flag

        Methods
            __init__: Instantiate the class
            __str__: Plot the results in string format.
    """
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


###############################################################################
def solve(f, x0, A=None, b=None, rho=DEFAULT_PARAMS['rho'], sketch_dim=DEFAULT_PARAMS['sketch_dim'],
          sketch_type=DEFAULT_PARAMS['sketch_type'], maxevals=DEFAULT_PARAMS['maxevals'],
          poll_type=DEFAULT_PARAMS['poll_type'], alpha0=DEFAULT_PARAMS['alpha0'],
          alpha_max=DEFAULT_PARAMS['alpha_max'], alpha_min=DEFAULT_PARAMS['alpha_min'],
          gamma_inc=DEFAULT_PARAMS['gamma_inc'], gamma_dec=DEFAULT_PARAMS['gamma_dec'],
          verbose=DEFAULT_PARAMS['verbose'], print_freq=DEFAULT_PARAMS['print_freq'],
          use_stochastic_three_points=DEFAULT_PARAMS['use_stochastic_three_points'],
          rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    """
        Apply a direct-search method to an optimization problem.

            Opt = solve(f, x0)
        or
            Opt = solve(f, x0, A, b)
        attempts to minimize the function f starting at x0 (possibly subject to the linear inequality constraints
        A @ x <= b) using a direct-search method. The final information is output in the Opt structure.

        Sketching and choice of poll type is not available if linear inequality constraints are provided.

        Inputs:
            f: Function handle for the objective to be minimized.
            x0: Initial point. Must satisfy constraints A @ x0 <= b
            A: matrix defining linear inequality constraints A @ x <= b. Default: None
            b: right-hand side defining linear inequality constraints A @ x <= b. Default: None
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

        
        Output:
            Opt: An instance of the OptimResults class containing the final 
            solution value, the function value at that solution, the number of 
            function evaluations and a termination flag.
    """
    if A is None and b is None:
        xmin, fmin, nf, flag = ds(f, x0, rho=rho, sketch_dim=sketch_dim, sketch_type=sketch_type, maxevals=maxevals,
                                   poll_type=poll_type, alpha0=alpha0, alpha_max=alpha_max, alpha_min=alpha_min,
                                   gamma_inc=gamma_inc, gamma_dec=gamma_dec, verbose=verbose, print_freq=print_freq,
                                   use_stochastic_three_points=use_stochastic_three_points, rho_uses_normd=rho_uses_normd)
    else:
        xmin, fmin, nf, flag = ds_lincons(f, x0, A, b, rho=rho, maxevals=maxevals,
                                  alpha0=alpha0, alpha_max=alpha_max, alpha_min=alpha_min,
                                  gamma_inc=gamma_inc, gamma_dec=gamma_dec, verbose=verbose, print_freq=print_freq,
                                  rho_uses_normd=rho_uses_normd)
    return OptimResults(xmin, fmin, nf, flag)

###############################################################################
def solve_directsearch(f, x0, A=None, b=None, rho=DEFAULT_PARAMS['rho'], maxevals=DEFAULT_PARAMS['maxevals'],
                       poll_type=DEFAULT_PARAMS['poll_type'], alpha0=DEFAULT_PARAMS['alpha0'],
                       alpha_max=DEFAULT_PARAMS['alpha_max'], alpha_min=DEFAULT_PARAMS['alpha_min'],
                       gamma_inc=DEFAULT_PARAMS['gamma_inc'], gamma_dec=DEFAULT_PARAMS['gamma_dec'],
                       verbose=DEFAULT_PARAMS['verbose'], print_freq=DEFAULT_PARAMS['print_freq'],
                       rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    """
        A wrapper for deterministic and probabilistic direct search without 
        sketching, with optional linear constraints A @ x <= b.

            Opt=solve_directsearch(f, x0)
        or
            Opt=solve_directsearch(f, x0, A, b)
        applies a regular direct-search method with sufficient decrease and adaptive stepsize.

        Inputs:
            f: Function handle for the objective to be minimized.
            x0: Initial point. Must satisfy constraints A @ x0 <= b
            A: matrix defining linear inequality constraints A @ x <= b. Default: None
            b: right-hand side defining linear inequality constraints A @ x <= b. Default: None
            rho: Choice of the forcing function.
                Default: see DEFAULT_PARAMS['rho']
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
            rho_uses_normd: Boolean indicating whether the forcing function should 
            account for the norm of the direction.
                Default: See DEFAULT_PARAMS['rho_uses_normd']

        Output: See output of a call to the solve() function.
    """

    return solve(f, x0, A, b, rho=rho, sketch_dim=None, maxevals=maxevals, poll_type=poll_type, alpha0=alpha0,
                 alpha_max=alpha_max, alpha_min=alpha_min, gamma_inc=gamma_inc, gamma_dec=gamma_dec, verbose=verbose,
                 print_freq=print_freq, use_stochastic_three_points=False, rho_uses_normd=rho_uses_normd)

###############################################################################
def solve_probabilistic_directsearch(f, x0, rho=DEFAULT_PARAMS['rho'], maxevals=DEFAULT_PARAMS['maxevals'],
                                     alpha0=DEFAULT_PARAMS['alpha0'], alpha_max=DEFAULT_PARAMS['alpha_max'],
                                     alpha_min=DEFAULT_PARAMS['alpha_min'], gamma_inc=DEFAULT_PARAMS['gamma_inc'],
                                     gamma_dec=DEFAULT_PARAMS['gamma_dec'], verbose=DEFAULT_PARAMS['verbose'],
                                     print_freq=DEFAULT_PARAMS['print_freq'],
                                     rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    """
        A wrapper for probabilistic direct search methods (without sketching).

        Opt=solve_probabilistic_directsearch(f,x0) applies a direct-search 
        method based on sufficient decrease and adaptive stepsize. At 
        every iteration, the method polls two opposite random directions 
        uniformly distributed on the unit sphere.

        Inputs:
            f: Function handle for the objective to be minimized.
            x0: Initial point.
            rho: Choice of the forcing function.
                Default: see DEFAULT_PARAMS['rho']
            maxevals: Maximum number of calls to f performed by the algorithm.
                Default: see DEFAULT_PARAMS['maxevals']
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
            rho_uses_normd: Boolean indicating whether the forcing function should 
            account for the norm of the direction.
                Default: See DEFAULT_PARAMS['rho_uses_normd']

        Output: See output of the solve() function.
    """

    return solve(f, x0, rho=rho, sketch_dim=None, maxevals=maxevals, poll_type='random2', alpha0=alpha0,
                 alpha_max=alpha_max, alpha_min=alpha_min, gamma_inc=gamma_inc, gamma_dec=gamma_dec, verbose=verbose,
                 print_freq=print_freq, use_stochastic_three_points=False, rho_uses_normd=rho_uses_normd)

###############################################################################
def solve_subspace_directsearch(f, x0, rho=DEFAULT_PARAMS['rho'], sketch_dim=DEFAULT_PARAMS['sketch_dim'],
                                sketch_type=DEFAULT_PARAMS['sketch_type'], maxevals=DEFAULT_PARAMS['maxevals'],
                                poll_type=DEFAULT_PARAMS['poll_type'], alpha0=DEFAULT_PARAMS['alpha0'],
                                alpha_max=DEFAULT_PARAMS['alpha_max'], alpha_min=DEFAULT_PARAMS['alpha_min'],
                                gamma_inc=DEFAULT_PARAMS['gamma_inc'], gamma_dec=DEFAULT_PARAMS['gamma_dec'],
                                verbose=DEFAULT_PARAMS['verbose'], print_freq=DEFAULT_PARAMS['print_freq'],
                                rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd']):
    """
        A wrapper for direct search based on probabilistic descent in 
        reduced subspaces.

        Opt=solve_subspace_directsearch(f,x0) applies a direct-search 
        method based on sufficent decrease and adaptive stepsize. At 
        every iteration, the method polls opposite directions generated 
        in a random subspace.

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
            rho_uses_normd: Boolean indicating whether the forcing function should 
            account for the norm of the direction.
                Default: See DEFAULT_PARAMS['rho_uses_normd']

        Output: See output of the solve() function.
    """

    return solve(f, x0, rho=rho, sketch_dim=sketch_dim, sketch_type=sketch_type, maxevals=maxevals,
                 poll_type=poll_type, alpha0=alpha0, alpha_max=alpha_max, alpha_min=alpha_min, gamma_inc=gamma_inc,
                 gamma_dec=gamma_dec, verbose=verbose, print_freq=print_freq, use_stochastic_three_points=False,
                 rho_uses_normd=rho_uses_normd)

###############################################################################
def solve_stp(f, x0, maxevals=DEFAULT_PARAMS['maxevals'], alpha0=DEFAULT_PARAMS['alpha0'],
              alpha_min=DEFAULT_PARAMS['alpha_min'], verbose=DEFAULT_PARAMS['verbose'],
              print_freq=DEFAULT_PARAMS['print_freq']):
    """
        A wrapper for the stochastic three-point method.

        Opt=solve_stp(f,x0) applies the stochastic three-point method, a 
        direct-search method based on simple decrease and predefined stepsize 
        sequence (alpha0/(k+1)). At every iteration, the method polls Gaussian 
        directions.

        Inputs:
            f: Function handle for the objective to be minimized.
            x0: Initial point.
            maxevals: Maximum number of calls to f performed by the algorithm.
                Default: see DEFAULT_PARAMS['maxevals']
            alpha0: Initial value for the stepsize parameter.
                Default: See DEFAULT_PARAMS['alpha0']
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

        Output: See output of the solve() function.
    """

    return solve(f, x0, sketch_dim=None, maxevals=maxevals, poll_type='random2', alpha0=alpha0, alpha_min=alpha_min,
                 verbose=verbose, print_freq=print_freq, use_stochastic_three_points=True)
