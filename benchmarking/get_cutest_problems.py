"""
Get the CUTEst problems used for benchmarking

Based on the ones used in
Gratton, Royer, Vicente & Zhang. Direct search based on probabilistic feasible descent for bound and linearly
constrained problems. Computational Optimization and Applications 72:3 (2019), pp. 525-559.
"""
import pycutest

### Raw data from paper  ###

# Bound constraints: (probname, dimension, # bound cons)
SET1 = [('ALLINIT', 4, 5),
('BQP1VAR', 1, 2),
('CAMEL6', 2, 4),
('CHEBYQAD', 10, 20),  # TODO typo in paper? Large-scale problems has same dimension but smaller dimension available (using smaller version for SET1)
('CHENHARK', 10, 10),
('CVXBQP1', 10, 20),
('DEGDIAG', 11, 11),
('DEGTRID', 11, 11),
# ('DEGTRID2', 11, 11),  # TODO removed as pycutest cannot set optional parameter to fix dimension
('EG1', 3, 4),
('EXPLIN', 12, 24),
('EXPLIN2', 12, 24),
('EXPQUAD', 12, 12),
('HARKERP2', 10, 10),
('HART6', 6, 12),
('HATFLDA', 4, 4),
('HATFLDB', 4, 5),
('HIMMELP1', 2, 4),
('HS1', 2, 1),
('HS25', 3, 6),
('HS2', 2, 1),
('HS38', 4, 8),
('HS3', 2, 1),
('HS3MOD', 2, 1),
('HS45', 5, 10),
('HS4', 2, 2),
('HS5', 2, 4),
('HS110', 10, 20),
('JNLBRNG1', 16, 28),
('JNLBRNG2', 16, 28),
('JNLBRNGA', 16, 28),
('JNLBRNGB', 16, 28),  # TODO added this since also in SET2 (same dimensions as JNLBRNGA)
('KOEBHELB', 3, 2),
('LINVERSE', 19, 10),
('LOGROS', 2, 2),
('MAXLIKA', 8, 16),
('MCCORMCK', 10, 20),
('MDHOLE', 2, 1),
('NCVXBQP1', 10, 20),
('NCVXBQP2', 10, 20),
('NCVXBQP3', 10, 20),
('NOBNDTOR', 16, 32),
('OBSTCLAE', 16, 32),
('OBSTCLBL', 16, 32),
('OSLBQP', 8, 11),
('PALMER1A', 6, 2),
('PALMER2B', 4, 2),
('PALMER3E', 8, 1),
('PALMER4A', 6, 2),
('PALMER5B', 9, 2),
('PFIT1LS', 3, 1),
('POWELLBC', 10, 20),  # TODO typo in paper? Large-scale problems has same dimension but smaller dimension available (using smaller version for SET1)
('PROBPENL', 10, 20),
('PSPDOC', 4, 1),
('QRTQUAD', 12, 12),
('S368', 8, 16),
('SCOND1LS', 12, 24),
('SIMBQP', 2, 2),
('SINEALI', 10, 20),  # TODO typo in paper? Large-scale problems has same dimension but smaller dimension available (using smaller version for SET1)
('SPECAN', 9, 18),
('TORSION1', 16, 32),
('TORSIONA', 16, 32),
('WEEDS', 3, 4),
('YFIT', 3, 1)]

# Bound constraints: (probname, dimension, # bound cons)
SET2 = [('CHEBYQAD', 20, 40),
('CHENHARK', 10, 10),
('CVXBQP1', 50, 100),
('DEGDIAG', 51, 51),
('DEGTRID', 51, 51),
# ('DEGTRID2', 51, 51),  # TODO removed as pycutest cannot set optional parameter to fix dimension
('EXPLIN', 12, 24),
('EXPLIN2', 12, 24),
('EXPQUAD', 12, 12),
('HARKERP2', 10, 10),
('HS110', 50, 100),
('JNLBRNG1', 16, 28),
('JNLBRNG2', 16, 28),
('JNLBRNGA', 16, 28),
('JNLBRNGB', 16, 28),
('LINVERSE', 19, 10),
('MCCORMCK', 50, 100),
('NCVXBQP1', 50, 100),
('NCVXBQP2', 50, 100),
('NCVXBQP3', 50, 100),
('NOBNDTOR', 16, 28),
('OBSTCLAE', 16, 32),
('OBSTCLBL', 16, 32),
('POWELLBC', 20, 40),
('PROBPENL', 50, 100),
('QRTQUAD', 12, 12),
('S368', 50, 100),
('SCOND1LS', 52, 104),
('SINEALI', 20, 40),
('TORSION1', 16, 32),
('TORSIONA', 16, 32)]

# Bound + linear eq constraints: (probname, dimension, # bound cons, # lin eq cons)
SET3 = [('AUG2D', 24, 0, 9),
('BOOTH', 2, 0, 2),
('BT3', 5, 0, 3),
('GENHS28', 10, 0, 8),
('HIMMELBA', 2, 0, 2),
('HS9', 2, 0, 1),
('HS28', 3, 0, 1),
('HS48', 5, 0, 2),
('HS49', 5, 0, 2),
('HS50', 5, 0, 3),
('HS51', 5, 0, 3),
('HS52', 5, 0, 3),
('ZANGWIL3', 3, 0, 3),
('CVXQP1', 10, 20, 5),
('CVXQP2', 10, 20, 2),
('DEGENLPA', 20, 40, 15),
('DEGENLPB', 20, 40, 15),
# ('DEGTRIDL', 11, 11, 1),  # TODO removed as pycutest cannot set optional parameter to fix dimension
('DUAL1', 85, 170, 1),
('DUAL2', 96, 192, 1),
('DUAL4', 75, 150, 1),
('EXTRASIM', 2, 1, 1),
('FCCU', 19, 19, 8),
# ('FERRISDC', 16, 24, 7),  # TODO removed as pycutest cannot set optional parameter to fix dimension
('GOULDQP1', 32, 64, 17),
('HONG', 4, 8, 1),
('HS41', 4, 8, 1),
('HS53', 5, 10, 3),
('HS54', 6, 12, 1),
('HS55', 6, 8, 6),
('HS62', 3, 6, 1),
('HS112', 10, 10, 3),
('LIN', 4, 8, 2),
('LOTSCHD', 12, 12, 7),
('NCVXQP1', 10, 20, 5),
('NCVXQP2', 10, 20, 5),
('NCVXQP3', 10, 20, 5),
('NCVXQP4', 10, 20, 2),
('NCVXQP5', 10, 20, 2),
('NCVXQP6', 10, 20, 2),
('ODFITS', 10, 10, 6),
('PORTFL1', 12, 24, 1),
('PORTFL2', 12, 24, 1),
('PORTFL3', 12, 24, 1),
('PORTFL4', 12, 24, 1),
('PORTFL6', 12, 24, 1),
('PORTSNQP', 10, 10, 2),
# ('PORTSQP', 10, 10, 1),  # TODO removed as pycutest cannot set optional parameter to fix dimension
('READING2', 9, 14, 4),
('SOSQP1', 20, 40, 11),
('SOSQP2', 20, 40, 11),
('STCQP1', 17, 34, 8),
('STCQP2', 17, 34, 8),
('STNQP1', 17, 34, 8),
('STNQP2', 17, 34, 8),
('SUPERSIM', 2, 1, 2),
('TAME', 2, 2, 1),
('TWOD', 31, 62, 10)]

# General linear constraints: (probname, dimension, # bound cons, # lin eq cons, # lin ineq cons)
SET4 = [('AVGASA', 8, 16, 0, 10),
('AVGASB', 8, 16, 0, 10),
('BIGGSC4', 4, 8, 0, 7),
('DUALC1', 9, 18, 1, 214),
('DUALC2', 7, 14, 1, 228),
('DUALC5', 8, 16, 1, 277),
('EQC', 9, 18, 0, 3),
('EXPFITA', 5, 0, 0, 22),
('EXPFITB', 5, 0, 0, 102),
('EXPFITC', 5, 0, 0, 502),
('HATFLDH', 4, 8, 0, 7),
('HS105', 8, 16, 0, 1),
('HS118', 15, 30, 0, 17),
('HS21', 2, 4, 0, 1),
('HS21MOD', 7, 8, 0, 1),
('HS24', 2, 2, 0, 3),
('HS268', 5, 0, 0, 5),
('HS35', 3, 3, 0, 1),
('HS35I', 3, 6, 0, 1),
('HS35MOD', 3, 4, 0, 1),
('HS36', 3, 6, 0, 1),
('HS37', 3, 6, 0, 2),
('HS44', 4, 4, 0, 6),
('HS44NEW', 4, 4, 0, 6),
('HS76', 4, 4, 0, 3),
('HS76I', 4, 8, 0, 3),
('HS86', 5, 5, 0, 10),
('HUBFIT', 2, 1, 0, 1),
('LSQFIT', 2, 1, 0, 1),
('OET1', 3, 0, 0, 6),
('OET3', 4, 0, 0, 6),
('PENTAGON', 6, 0, 0, 15),
('PT', 2, 0, 0, 501),
('QC', 9, 18, 0, 4),
('QCNEW', 9, 18, 0, 3),
('S268', 5, 0, 0, 5),
('SIMPLLPA', 2, 2, 0, 2),
('SIMPLLPB', 2, 2, 0, 3),
('SIPOW1', 2, 0, 0, 2000),
('SIPOW1M', 2, 0, 0, 2000),
('SIPOW2', 2, 0, 0, 2000),
('SIPOW2M', 2, 0, 0, 2000),
('SIPOW3', 4, 0, 0, 20),
('SIPOW4', 4, 0, 0, 2000),
('STANCMIN', 3, 3, 0, 2),
('TFI2', 3, 0, 0, 101),
('TFI3', 3, 0, 0, 101),
('ZECEVIC2', 2, 4, 0, 2)]


SET1_PARAMS = {}
SET1_PARAMS['CHEBYQAD'] = {'N': 10}  # TODO typo in paper? Large-scale problems also has N=20
SET1_PARAMS['CHENHARK'] = {'N': 10, 'NFREE': 5, 'NDEGEN': 2}
SET1_PARAMS['CVXBQP1'] = {'N': 10}
SET1_PARAMS['DEGDIAG'] = {'N': 10}
SET1_PARAMS['DEGTRID'] = {'N': 10}
# SET1_PARAMS['DEGTRID2'] = {'N': 10}  # TODO appears to be a valid option, but pycutest can't build it?
SET1_PARAMS['EXPLIN'] = {'N': 12, 'M': 6}
SET1_PARAMS['EXPLIN2'] = {'N': 12, 'M': 6}
SET1_PARAMS['EXPQUAD'] = {'N': 12, 'M': 6}
SET1_PARAMS['HARKERP2'] = {'N': 10}
SET1_PARAMS['HS110'] = {'N': 10}
SET1_PARAMS['JNLBRNG1'] = {'PT': 4, 'PY': 4}
SET1_PARAMS['JNLBRNG2'] = {'PT': 4, 'PY': 4}
SET1_PARAMS['JNLBRNGA'] = {'PT': 4, 'PY': 4}
SET1_PARAMS['JNLBRNGB'] = {'PT': 4, 'PY': 4}  # TODO manually added
SET1_PARAMS['LINVERSE'] = {'N': 10}
SET1_PARAMS['MCCORMCK'] = {'N': 10}
SET1_PARAMS['NCVXBQP1'] = {'N': 10}
SET1_PARAMS['NCVXBQP2'] = {'N': 10}
SET1_PARAMS['NCVXBQP3'] = {'N': 10}
SET1_PARAMS['NOBNDTOR'] = {'Q': 2}
SET1_PARAMS['OBSTCLAE'] = {'PX': 4, 'PY': 4}
SET1_PARAMS['OBSTCLBL'] = {'PX': 4, 'PY': 4}
SET1_PARAMS['POWELLBC'] = {'P': 5}  # TODO typo in paper? (P=10 used for large-scale problems)
SET1_PARAMS['PROBPENL'] = {'N': 10}
SET1_PARAMS['QRTQUAD'] = {'N': 12, 'M': 6}
SET1_PARAMS['S368'] = {'N': 8}
SET1_PARAMS['SCOND1LS'] = {'N': 10, 'LN': 9}
SET1_PARAMS['SINEALI'] = {'N': 10}  # TODO typo in paper? (N=20 used for large-scale problems)
SET1_PARAMS['SPECAN'] = {'K': 3}
SET1_PARAMS['TORSION1'] = {'Q': 2}
SET1_PARAMS['TORSIONA'] = {'Q': 2}

SET2_PARAMS = {}
SET2_PARAMS['CHEBYQAD'] = {'N': 20}
SET2_PARAMS['CHENHARK'] = {'N': 10, 'NFREE': 5, 'NDEGEN': 2}
SET2_PARAMS['CVXBQP1'] = {'N': 50}
SET2_PARAMS['DEGDIAG'] = {'N': 50}
SET2_PARAMS['DEGTRID'] = {'N': 50}
# SET2_PARAMS['DEGTRID2'] = {'N': 50}  # TODO appears to be a valid option, but pycutest can't build it?
SET2_PARAMS['EXPLIN'] = {'N': 12, 'M': 6}
SET2_PARAMS['EXPLIN2'] = {'N': 12, 'M': 6}
SET2_PARAMS['EXPQUAD'] = {'N': 12, 'M': 6}
SET2_PARAMS['HARKERP2'] = {'N': 10}
SET2_PARAMS['HS110'] = {'N': 50}
SET2_PARAMS['JNLBRNG1'] = {'PT': 4, 'PY': 4}
SET2_PARAMS['JNLBRNG2'] = {'PT': 4, 'PY': 4}
SET2_PARAMS['JNLBRNGA'] = {'PT': 4, 'PY': 4}
SET2_PARAMS['JNLBRNGB'] = {'PT': 4, 'PY': 4}  # TODO why not in SET1?
SET2_PARAMS['LINVERSE'] = {'N': 10}
SET2_PARAMS['MCCORMCK'] = {'N': 50}
SET2_PARAMS['NCVXBQP1'] = {'N': 50}
SET2_PARAMS['NCVXBQP2'] = {'N': 50}
SET2_PARAMS['NCVXBQP3'] = {'N': 50}
SET2_PARAMS['NOBNDTOR'] = {'Q': 2}
SET2_PARAMS['OBSTCLAE'] = {'PX': 4, 'PY': 4}
SET2_PARAMS['OBSTCLBL'] = {'PX': 4, 'PY': 4}
SET2_PARAMS['POWELLBC'] = {'P': 10}
SET2_PARAMS['PROBPENL'] = {'N': 50}
SET2_PARAMS['QRTQUAD'] = {'N': 12, 'M': 6}
SET2_PARAMS['S368'] = {'N': 50}
SET2_PARAMS['SCOND1LS'] = {'N': 50, 'LN': 45}
SET2_PARAMS['SINEALI'] = {'N': 20}
SET2_PARAMS['TORSION1'] = {'Q': 2}
SET2_PARAMS['TORSIONA'] = {'Q': 2}

SET3_PARAMS = {}
SET3_PARAMS['AUG2D'] = {'NX': 3, 'NY': 3}
SET3_PARAMS['CVXQP1'] = {'N': 10}
SET3_PARAMS['CVXQP2'] = {'N': 10}
# SET3_PARAMS['DEGTRIDL'] = {'N': 10}  # TODO pycutest crashes trying to set this
# SET3_PARAMS['FERRISDC'] = {'N': 4, 'K': 3}  # TODO pycutest crashes trying to set this
SET3_PARAMS['NCVXQP1'] = {'N': 10}
SET3_PARAMS['NCVXQP2'] = {'N': 10}
SET3_PARAMS['NCVXQP3'] = {'N': 10}
SET3_PARAMS['NCVXQP4'] = {'N': 10}
SET3_PARAMS['NCVXQP5'] = {'N': 10}
SET3_PARAMS['NCVXQP6'] = {'N': 10}
# SET3_PARAMS['PORTSQP'] = {'N': 10}  # TODO pycutest crashes trying to set this
SET3_PARAMS['READING2'] = {'N': 2}
SET3_PARAMS['SOSQP1'] = {'N': 10}
SET3_PARAMS['SOSQP2'] = {'N': 10}
SET3_PARAMS['STCQP1'] = {'P': 4}
SET3_PARAMS['STCQP2'] = {'P': 4}
SET3_PARAMS['STNQP1'] = {'P': 4}
SET3_PARAMS['STNQP2'] = {'P': 4}
SET3_PARAMS['TWOD'] = {'N': 2}

SET4_PARAMS = {}

def main():

    print("*** SET1 ***")
    for (probname, n, nbounds) in SET1:
        # print(probname, n, nbounds)  # SET1, SET2
        # pycutest.print_available_sif_params(probname)
        sifParams = SET1_PARAMS.get(probname, None)
        if sifParams is None:
            p = pycutest.import_problem(probname, drop_fixed_variables=False)
        else:
            p = pycutest.import_problem(probname, drop_fixed_variables=False, sifParams=sifParams)
        if p.n != n:
            print(probname, n, p)

    print("*** SET2 ***")
    for (probname, n, nbounds) in SET2:
        # print(probname, n, nbounds)  # SET1, SET2
        # pycutest.print_available_sif_params(probname)
        sifParams = SET2_PARAMS.get(probname, None)
        if sifParams is None:
            p = pycutest.import_problem(probname, drop_fixed_variables=False)
        else:
            p = pycutest.import_problem(probname, drop_fixed_variables=False, sifParams=sifParams)
        if p.n != n:
            print(probname, n, p)

    print("*** SET3 ***")
    for (probname, n, nbounds, nle) in SET3:
        # print(probname, n, nbounds, nle)  # SET3
        # pycutest.print_available_sif_params(probname)
        sifParams = SET3_PARAMS.get(probname, None)
        if sifParams is None:
            p = pycutest.import_problem(probname, drop_fixed_variables=False)
        else:
            p = pycutest.import_problem(probname, drop_fixed_variables=False, sifParams=sifParams)
        if p.n != n:
            print(probname, n, p)

    print("*** SET4 ***")
    for (probname, n, nbounds, nle, nli) in SET4:
        # print(probname, n, nbounds, nle, nli)  # SET4
        # pycutest.print_available_sif_params(probname)
        sifParams = SET4_PARAMS.get(probname, None)
        if sifParams is None:
            p = pycutest.import_problem(probname, drop_fixed_variables=False)
        else:
            p = pycutest.import_problem(probname, drop_fixed_variables=False, sifParams=sifParams)
        if p.n != n:
            print(probname, n, p)
    print("Done")
    return

if __name__ == '__main__':
    main()
