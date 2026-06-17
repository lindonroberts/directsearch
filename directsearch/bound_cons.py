"""
Methods to handle bound constrained problems (no general linear constraints).

Based on lincons.py
"""
import numpy as np
# from scipy.linalg import null_space, qr
# from scipy.optimize import linprog, minimize, LinearConstraint, NonlinearConstraint, direct
from scipy.optimize import Bounds
import warnings

try:
    from .ds import DEFAULT_PARAMS, EXIT_MAXFUN_REACHED, EXIT_ALPHA_MIN_REACHED
except ImportError:
    from ds import DEFAULT_PARAMS, EXIT_MAXFUN_REACHED, EXIT_ALPHA_MIN_REACHED

def process_bounds(x0, bounds):
    """
    Take bounds object and format them as (xL, xU) numpy arrays
    """
    # Extract bounds into xL and xU
    xL = np.full(x0.shape, -np.inf, dtype=x0.dtype)
    xU = np.full(x0.shape, np.inf, dtype=x0.dtype)
    if bounds is not None:
        if isinstance(bounds, Bounds):
            if isinstance(bounds.lb, float):
                xL = np.full(x0.shape, bounds.lb, dtype=x0.dtype)
            else:
                xL = bounds.lb.astype(x0.dtype)
            if isinstance(bounds.ub, float):
                xU = np.full(x0.shape, bounds.ub, dtype=x0.dtype)
            else:
                xU = bounds.ub.astype(x0.dtype)
        elif isinstance(bounds, list):
            if len(bounds) != len(x0):
                raise ValueError("Length of 'bounds' inconsistent with 'x0'.")
            for i, (xl_i, xu_i) in enumerate(bounds):
                xL[i] = xl_i
                xU[i] = xu_i
        else:
            raise ValueError("If not None, 'bounds' must be of type Bounds or list.")
    return xL, xU

def get_poll_directions(xL, xU, x, alpha, include_negative_directions=True, include_negative_sum=False, verbose=False):
    """
    Given feasible region xL <= y <= xU, a feasible point x and radius alpha, return a useful set of
    feasible poll directions in B(x,alpha).

    D, Dneg = get_poll_directions(xL, xU, x, alpha)

    :param xL: length-n vector defining the bound constraints
    :param xU: length-n vector defining the bound constraints
    :param x: length-n vector for the current iterate
    :param alpha: radius of search region, alpha > 0
    :return: D, n*p matrix (some p) of vectors in B(0,alpha) such that all poll points x+D[:,i] are feasible
    :return: Dneg, n*p2 matrix (some p2) similar to D, but corresponding to negative of tangent directions. Possibly None
    """
    ZERO_THRESH = 10.0 * np.finfo(float).eps  # for measuring distance to boundary
    n = len(x)
    assert xL.shape == (n,), "xL has incompatible shape with x"
    assert xU.shape == (n,), "xU has incompatible shape with x"
    assert alpha > 0.0, "alpha must be strictly positive"
    assert np.all(x >= xL), "x must be >= xL"
    assert np.all(x <= xU), "x must be <= xU"

    if verbose and True:
        print("x =", x)
        print("xL =", xL)
        print("xU =", xU)
        print("alpha = %g" % alpha)

    # Vector defining nearly active bounds (i.e. where the bound is within distance alpha of x):
    near_xL = (x - alpha < xL)
    near_xU = (x + alpha > xU)
    if verbose:
        print("near xL =", near_xL)
        print("near xU =", near_xU)

    if np.all(near_xL) and np.all(near_xU):
        # Tangent cone is empty, use scaled normals only
        T = np.zeros((n, 0), dtype=float)
        for i in range(n):
            ei = np.zeros((n,1)); ei[i] = 1.0
            if near_xL[i]:
                alpha_i = min(alpha, x[i] - xL[i])
                T = np.hstack((T, -alpha_i*ei))
            if near_xU[i]:
                alpha_i = min(alpha, xU[i] - x[i])
                T = np.hstack((T, alpha_i * ei))
        return T, None, 2 * n

    # Tangent cone = +/- alpha*ei for inactive bounds
    T = np.zeros((n, 0), dtype=float)
    for i in range(n):
        ei = np.zeros((n,1)); ei[i] = 1.0
        if not near_xL[i]:
            T = np.hstack((T, -alpha*ei))
        if not near_xU[i]:
            T = np.hstack((T, alpha*ei))

    # Normal cone directions
    Tneg = np.zeros((n, 0), dtype=float)

    if include_negative_directions:
        for i in range(n):
            ei = np.zeros((n,1)); ei[i] = 1.0
            if near_xL[i]:
                alpha_i = min(alpha, x[i] - xL[i])
                Tneg = np.hstack((Tneg, -alpha_i*ei))
            if near_xU[i]:
                alpha_i = min(alpha, xU[i] - x[i])
                Tneg = np.hstack((Tneg, alpha_i * ei))

    if include_negative_sum:
        negsum = np.zeros((n,1), dtype=float)
        for i in range(n):
            ei = np.zeros((n,1)); ei[i] = 1.0
            if near_xL[i]:
                alpha_i = min(alpha, x[i] - xL[i])
                negsum += -alpha_i * ei
            if near_xU[i]:
                alpha_i = min(alpha, xU[i] - x[i])
                negsum += alpha_i * ei
        if np.linalg.norm(negsum) > alpha:
            negsum = (alpha / np.linalg.norm(negsum)) * negsum
        if np.linalg.norm(negsum) > ZERO_THRESH:
            # Check negsum not already in Tneg from normal directions (e.g. one active constraint)
            negsum_in_Tneg = False
            # print("negsum =", negsum)
            for j in range(Tneg.shape[1]):
                # print("Tneg[j] =", Tneg[:,j])
                # print("dist =", np.linalg.norm(negsum.reshape((n,)) - Tneg[:,j]))
                if np.linalg.norm(negsum.reshape((n,)) - Tneg[:,j]) < ZERO_THRESH:
                    negsum_in_Tneg = True
                    break

            if not negsum_in_Tneg:
                Tneg = np.hstack((Tneg, negsum))

    if Tneg.shape[1] == 0:
        Tneg = None

    return T, Tneg, int(near_xL.sum() + near_xU.sum())  # last return value is number of nearly active bounds


def ds_bounds(f, x0, bounds=None,
               rho=DEFAULT_PARAMS['rho'],
               maxevals=DEFAULT_PARAMS['maxevals'],
               alpha0=DEFAULT_PARAMS['alpha0'],
               alpha_max=DEFAULT_PARAMS['alpha_max'],
               alpha_min=DEFAULT_PARAMS['alpha_min'],
               gamma_inc=DEFAULT_PARAMS['gamma_inc'],
               gamma_dec=DEFAULT_PARAMS['gamma_dec'],
               verbose=DEFAULT_PARAMS['verbose'],
               print_freq=DEFAULT_PARAMS['print_freq'],
               rho_uses_normd=DEFAULT_PARAMS['rho_uses_normd'],
               poll_normal_cone=DEFAULT_PARAMS['poll_normal_cone'],
               poll_normal_cone_negsum=False
               ):
    """
        A generic direct-search code for bound constrained black-box optimization.

            x, fx, nf, flag = ds_bounds(f, x0, bounds)

        attempts to minimize the function f starting at x0, subject to the bounds xL <= x <= xU
        defined by the scipy.optimize.Bounds object, using a direct-search approach.
        The method is based on moves along certain polling directions: these moves are
        controlled by means of an adaptive stepsize.

        Inputs:
            f: Function handle for the objective to be minimized.
            x0: Initial point. Must satisfy constraints A @ x0 <= b
            bounds: scipy.optimize.Bounds object defining bounds xL <= x <= xU. Default: None
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
            account for the norm of the direction. Always set to False if poll_normal_cone=True.
                Default: See DEFAULT_PARAMS['rho_uses_normd']
            poll_normal_cone: Boolean indicating if the poll steps should include
                checking the normal cone (or just the tangent cone, if False)
                Default: See DEFAULT_PARAMS['poll_normal_cone']

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
    poll_normal_cone = bool(poll_normal_cone)
    poll_normal_cone_negsum = bool(poll_normal_cone_negsum)
    if rho is None:
        if rho_uses_normd and not poll_normal_cone:  # cannot use this when poll_normal_cone=True
            rho_to_use = lambda t, normd: min(1e-5, 1e-5 * (t * normd) ** 2)
        else:
            rho_to_use = lambda t: min(1e-5, 1e-5 * t ** 2)
    else:
        rho_to_use = rho

    # Force correct types
    x = np.array(x0, dtype=float)
    n = len(x)
    x = x.reshape((n,))
    xL, xU = process_bounds(x, bounds)
    alpha_max = float(alpha_max)
    alpha_min = float(alpha_min)
    gamma_inc = float(gamma_inc)
    gamma_dec = float(gamma_dec)
    verbose = bool(verbose)

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

    # Ensure x0 is feasible
    x = np.maximum(np.minimum(xU, x), xL)

    # Input checking
    assert callable(f), "Objective function should be callable"
    assert callable(rho_to_use), "Sufficient decrease function rho should be callable"
    assert maxevals > 0, "maxevals should be strictly positive"
    assert alpha_max > 0.0, "alpha_max should be strictly positive"
    assert alpha0 > 0.0, "alpha0 should be strictly positive"
    assert alpha_min > 0.0, "alpha_max should be strictly positive"
    assert alpha_min <= alpha_max, "alpha_min should be <= alpha_max"
    assert alpha0 >= alpha_min, "alpha0 should be >= alpha_min"
    assert gamma_inc >= 1.0, "gamma_inc should be at least 1"
    assert gamma_dec > 0.0, "gamma_dec should be strictly positive"
    assert gamma_dec < 1.0, "gamma_dec should be strictly < 1"
    if verbose:
        assert print_freq > 0, "print_freq should be strictly positive"

    ###################################
    # Start of the optimization process
    fx = f(x)
    nf = 1
    iteration_counts = {'successful': 0, 'successful_negative_direction': 0, 'unsuccessful': 0}
    if nf >= maxevals:
        if verbose:
            print("Quit (max evals)")
        return x, fx, nf, EXIT_MAXFUN_REACHED, iteration_counts

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

        # Generate poll directions adapted to linear constraints
        Dk, Dk_neg, m_active = get_poll_directions(xL, xU, x, alpha, include_negative_directions=poll_normal_cone,
                                                             include_negative_sum=poll_normal_cone_negsum, verbose=verbose)
        # print("gen type =", gen_type)
        # WARNING: poll directions Dk[:,i] are already scaled by alpha, don't multiply by alpha below
        # print("******")
        # print("At x =", x, ", alpha = %g" % alpha)
        # print("Poll directions =")
        # print(Dk)
        # print("******")

        # Start poll step
        ndirs1 = Dk.shape[1]
        ndirs2 = Dk_neg.shape[1] if Dk_neg is not None else 0
        ndirs = ndirs1 + ndirs2

        # Regular direct search - Sufficient decrease, adaptive stepsize
        polling_successful = False
        used_negative_direction = False
        for j in range(ndirs):
            dj = Dk[:, j] if j < ndirs1 else Dk_neg[:, j-ndirs1]
            xnew = x + dj  # WARNING: Dk is already scaled by alpha, so don't multiply by alpha here (different to ds.py)
            fnew = f(xnew)
            nf += 1
            # Compute the target improvement
            sufficient_decrease = (fnew < fx - rho_to_use(alpha, np.linalg.norm(dj))) if rho_uses_normd and not poll_normal_cone else (
                    fnew < fx - rho_to_use(alpha))

            # Quit on budget (update to xnew if we just saw an improvement)
            if nf >= maxevals:
                if sufficient_decrease:
                    x = xnew.copy()
                    fx = fnew
                if sufficient_decrease:
                    if used_negative_direction:
                        iteration_counts['successful_negative_direction'] += 1
                    else:
                        iteration_counts['successful'] += 1
                else:
                    iteration_counts['unsuccessful'] += 1
                if verbose:
                    print("{0:^5}{1:^15.4e}{2:^15.2e} - max evals reached".format(k, fx, alpha))
                return x, fx, nf, EXIT_MAXFUN_REACHED, iteration_counts

            # If sufficient decrease, update xk and stop poll step
            if sufficient_decrease:
                x = xnew.copy()
                fx = fnew
                # Found a better point=> Possibly increase the stepsize
                alpha = min(gamma_inc * alpha, alpha_max)
                polling_successful = True
                if j >= ndirs1:
                    used_negative_direction = True
                break  # stop poll step, go to next iteration

        # Determine iteration type
        if polling_successful:
            if used_negative_direction:
                iteration_counts['successful_negative_direction'] += 1
            else:
                iteration_counts['successful'] += 1
        else:
            iteration_counts['unsuccessful'] += 1

        # If here, no decrease found
        if alpha < alpha_min:
            if verbose:
                print("{0:^5}{1:^15.4e}{2:^15.2e} - small alpha reached".format(k, fx, alpha))
            break  # finish algorithm
            # Note - Could return here

        if not polling_successful:
            # No better found point=> Decrease the stepsize
            alpha = gamma_dec * alpha
        # End loop
        ###########

    return x, fx, nf, EXIT_ALPHA_MIN_REACHED, iteration_counts


def poll_test():
    xL = np.array([0.0, 0.0])
    xU = np.array([1.0, 1.0])

    print("*** Case 1 (interior) ***")
    x = np.array([0.5, 0.5])
    alpha = 0.1
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("")
    print("*** Case 2a (one side / lower) ***")
    x = np.array([0.5, 0.1])
    alpha = 0.2
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("")
    print("*** Case 2b (one side / upper) ***")
    x = np.array([0.9, 0.5])
    alpha = 0.2
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("")
    print("*** Case 3a (corner 1) ***")
    x = np.array([0.1, 0.1])
    alpha = 0.2
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("")
    print("*** Case 3b (corner 1 / upper) ***")
    x = np.array([0.9, 0.9])
    alpha = 0.2
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("")
    print("*** Case 3c (corner 1 / mixed) ***")
    x = np.array([0.9, 0.1])
    alpha = 0.2
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("")
    print("*** Case 4a (corner 2 / lower) ***")
    x = np.array([0.1, 0.1])
    alpha = 0.13  # negsum = (-0.1, -0.1) is outside B(x,alpha)
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("")
    print("*** Case 4b (corner 2 / upper) ***")
    x = np.array([0.9, 0.9])
    alpha = 0.13  # negsum = (-0.1, -0.1) is outside B(x,alpha)
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("")
    print("*** Case 4c (corner 2 / mixed) ***")
    x = np.array([0.1, 0.9])
    alpha = 0.13  # negsum = (-0.1, -0.1) is outside B(x,alpha)
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("*** Case 5 (everything) ***")
    x = np.array([0.5, 0.5])
    alpha = 1.0
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)

    print("*** Case 6 (everything) ***")
    x = np.array([0.5, 0.5])
    alpha = 0.6  # corners are not in B(x,alpha)
    for include_negative_directions in [False, True]:
        for include_negative_sum in [False, True]:
            D, Dneg, _ = get_poll_directions(xL, xU, x, alpha,
                                             include_negative_directions=include_negative_directions,
                                             include_negative_sum=include_negative_sum)
            print("Include neg dirns =", include_negative_directions, ", include neg sum =", include_negative_sum)
            print(" - D =")
            print(D)
            print(" - Dneg =")
            print(Dneg)
    return


def main():
    poll_test()
    print("Done")
    return

if __name__ == '__main__':
    main()