import numpy as np
from Global_Vars import Global_vars
from Model_A_MGA_GCY9_GELAN import Model_A_MGA_GCY9_GELAN


def objfun_seg(Soln):
    Feat = Global_vars.Feat
    Target = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, predict = Model_A_MGA_GCY9_GELAN(Feat, Target, sol)
            Fitn[i] = 1 / (Eval[0, 5])  # 1 / IoU
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, predict = Model_A_MGA_GCY9_GELAN(Feat, Target, sol)
        Fitn = 1 / (Eval[0, 5])  # 1 / IoU
        return Fitn