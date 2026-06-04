"""
Script demonstrating different examples of poll set generation
for linearly constrained problems

(previously in directsearch/lincons.py)
"""
import numpy as np
from directsearch.lincons import get_poll_directions


def poll_set_example():
    np.set_printoptions(precision=5, suppress=True)
    # 2d simplex
    A = np.array([[-1.0, 0.0],
                  [0.0, -1.0],
                  [1.0, 1.0]])
    b = np.array([0.0, 0.0, 1.0])

    print("*** 01 Simple case 1 ***")
    x = np.array([0.3, 0.3])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    print("*** 02 Simple case 2 ***")
    x = np.array([0.3, 0.3])
    alpha = 5.0
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    print("*** 03 Medium case 1 ***")
    x = np.array([0.01, 0.5])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    print("*** 04 Medium case 2 ***")
    x = np.array([0.01, 0.01])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    print("*** 05 Medium case 3 ***")
    x = np.array([0.49, 0.49])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    print("*** 07 Hard case 1 ***")
    x = np.array([0.01, 0.98])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    print("*** 08 Hard case 2 ***")
    x = np.array([0.01, 0.99])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    # 3d simplex
    A = np.array([[-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [1.0, 1.0, 1.0]])
    b = np.array([0.0, 0.0, 0.0, 1.0])

    print("*** 09 Hard case 1 (3d) ***")
    x = np.array([0.01, 0.01, 0.95])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    print("*** 10 Hard case 2 (3d) ***")
    x = np.array([0.02, 0.02, 0.96])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    # 3d pyramid
    print("*** 11 Pyramid (3d) ***")
    L = 2.0
    A = np.array([[0.0, 0.0, -1.0],  # z >= 0  <-->  -z <= 0
                  [1.0, 0.0, 0.0],  # x <= L
                  [-1.0, 0.0, 0.0],  # x >= -L  <-->  -x <= L
                  [0.0, 1.0, 0.0],  # y <= L
                  [0.0, -1.0, 0.0], # y >= -L  <-->  -y <= L
                  [1.0, 0.0, L],  # x + Lz <= L
                  [-1.0, 0.0, L], # -x + Lz <= L
                  [0.0, 1.0, L], # y + Lz <= L
                  [0.0, -1.0, L]]) # -y + Lz <= L
    b = np.array([0.0, L, L, L, L, L, L, L, L])

    x = np.array([0.0, 0.0, 0.98])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    # Fig 8.6a from Kolda et al (SIREV, 2003)
    print("*** 12 Kolda et al ***")
    A = np.array([[0.0, 1.0],  # x2 <= 1
                 [1.0, 1.0]])  # x1 + x2 <= 2
    b = np.array([1.0, 2.0])
    x = np.array([0.95, 0.99])
    alpha = 0.1
    D = get_poll_directions(A, b, x, alpha)
    print("D =")
    print(D)

    # Rank deficient case (caused errors in previous versions)
    print("*** 13 Rank deficient ***")
    A = np.array([[0.0, 1.0, 0.0],  # x2 <= 1
                  [0.0, -1.0, 0.0]])  # x2 >= 0
    b = np.array([1.0, 0.0])
    x = np.array([0.0, 0.8, 0.0])
    alpha = 1.0  # so both constraints are active
    Dpos, Dneg = get_poll_directions(A, b, x, alpha)
    print("Dpos =")
    print(Dpos)
    print("Dneg =")
    print(Dneg)
    return


def main():
    # simplex_example()
    # dd_example()
    poll_set_example()
    print("Done")
    return


if __name__ == '__main__':
    main()
