#!/usr/bin/env python

import sys, numpy as np
from yaff import *

def get_cv(ff):
    return [colvar.Volume()]

def adapt_structure(ff,cv):
    cv_object = get_cv(ff)[0]

    # Adapt stucture for cv
    scale = (cv/cv_object.get_cv_value(ff.system))**(1/3.)
    ff.update_pos(ff.system.pos * scale)
    ff.update_rvecs(ff.system.cell.rvecs * scale)

    cell_symmetrize(ff)

    try:
        assert np.round(cv,8) == np.round(cv_object.get_cv_value(ff.system),8)
    except AssertionError:
        print(cv, cv_object.get_cv_value(ff.system))
        sys.exit(1)

    # Short opt step for good initial configuration
    dof = FixedVolOrthoCellDOF(ff, gpos_rms=1e-8, dpos_rms=1e-6, grvecs_rms=1e-8, drvecs_rms=1e-6)
    opt = CGOptimizer(dof)
    opt.run(500)

    try:
        assert np.round(cv,8) == np.round(cv_object.get_cv_value(ff.system),8)
    except AssertionError:
        print(cv, cv_object.get_cv_value(ff.system))
        sys.exit(1)
