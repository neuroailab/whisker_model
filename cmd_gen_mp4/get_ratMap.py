import numpy as np
import os
import sys

class params_RatMap:
    def init(self):
        pass

def LOCAL_SetupDefaults():

    S = params_RatMap()
    # Calcualtion defaults
    S.Npts = 100
    S.TGL_PHI = 'proj'

    # Setup ellipsoid defaults
    S.E_C = [1.9128, -7.6549, -5.4439]
    S.E_R = [9.5304, 5.5393, 6.9745]
    S.E_OA = [106.5100, -2.5211, -19.5401]

    # Setup whisker transformation parameters and equations
    # (parameters must be None to use equations)

    S.EQ_BP_th = [15.2953, 0, -144.2220]
    S.BP_th = None

    S.EQ_BP_phi = [0, 18.2237, 34.7558]
    S.BP_phi = None

    S.EQ_W_s = [-7.9312,2.2224,52.1110]
    S.W_s = None

    S.EQ_W_a = [-0.02052,0,-0.2045]
    S.W_a = None

    S.EQ_W_th = [10.6475,0,37.3178]
    S.W_th = None

    S.EQ_W_psi = [18.5149,49.3499,-50.5406]
    S.W_psi = None

    S.EQ_W_zeta = [18.7700,-11.3485,-4.9844]
    S.W_zeta = None

    S.EQ_W_phi = [1.0988,-18.0334,50.6005]
    S.W_phi = None
    S.EQ_W_phiE = [0,-15.8761,47.3263]
    S.W_phiE = None

    return S

def LOCAL_SetupWhiskerNames(wselect):

    wname = ['A0','A1','A2','A3','A4',
              'B0','B1','B2','B3','B4','B5',
              'C0','C1','C2','C3','C4','C5','C6',
              'D0','D1','D2','D3','D4','D5','D6',
              'E1','E2','E3','E4','E5','E6']

    lwname  = ['L' + nam_per for nam_per in wname]
    rwname  = ['R' + nam_per for nam_per in wname]
    lwname.extend(rwname)
    wname   = lwname
    allow_list  = "LRABCDE0123456"
    for wpart in wselect:
        if wpart not in allow_list:
            continue
        wname   = [wname_tmp for wname_tmp in wname if wpart in wname_tmp]

    return wname

def LOCAL_UnpackEquations(S):
    ROW = np.zeros(len(S.wname))
    COL = np.zeros(len(S.wname))
    EYE = np.ones(len(S.wname))
    ltr = 'ABCDE'
    nm = '0123456'
    for name_indx, name_tmp in enumerate(S.wname):
        ROW[name_indx]  = ltr.index(name_tmp[1]) +1
        COL[name_indx]  = nm.index(name_tmp[2]) +1

    S.C_BP_th = S.EQ_BP_th[0]*COL + S.EQ_BP_th[1]*ROW + S.EQ_BP_th[2]
    S.C_BP_phi = S.EQ_BP_phi[0]*COL + S.EQ_BP_phi[1]*ROW + S.EQ_BP_phi[2]

    S.C_s = S.EQ_W_s[0]*COL + S.EQ_W_s[1]*ROW + S.EQ_W_s[2]
    S.C_a = np.exp(1/(S.EQ_W_a[0]*COL + S.EQ_W_a[1]*ROW + S.EQ_W_a[2]))

    #print(S.C_a)

    S.C_thetaP = S.EQ_W_th[0]*COL + S.EQ_W_th[1]*ROW + S.EQ_W_th[2]
    S.C_phiP = S.EQ_W_phi[0]*COL + S.EQ_W_phi[1]*ROW + S.EQ_W_phi[2]
    S.C_psiP = S.EQ_W_psi[1]*COL + S.EQ_W_psi[1]*ROW + S.EQ_W_psi[2]
    S.C_zetaP = S.EQ_W_zeta[0]*COL + S.EQ_W_zeta[1]*ROW + S.EQ_W_zeta[2]

    S.C_phiE = S.EQ_W_phiE[0]*COL + S.EQ_W_phiE[1]*ROW + S.EQ_W_phiE[2]

    return S

def LOCAL_CalculateParameters(S):

    d2r = np.pi/180
    SIDE    = [indx for indx, name_tmp in enumerate(S.wname) if name_tmp[0]=='R']
    SIDE_L  = [indx for indx, name_tmp in enumerate(S.wname) if name_tmp[0]=='L']
    S.C_zeta    = np.zeros(len(S.C_zetaP))
    S.C_zeta[SIDE] = S.C_zetaP[SIDE]+90;
    S.C_zeta[SIDE_L] = 90-S.C_zetaP[SIDE_L]

    S.C_theta = np.zeros(len(S.C_thetaP))

    for indx, thetaP_tmp in enumerate(S.C_thetaP):
        if indx in SIDE:
            if (thetaP_tmp >= 90) and (thetaP_tmp <= 200):
                S.C_theta[indx] = thetaP_tmp - 90
            elif (thetaP_tmp >= 0) and (thetaP_tmp <= 90):
                S.C_theta[indx] = thetaP_tmp + 270
        else:
            if (thetaP_tmp >= 90) and (thetaP_tmp <= 200):
                S.C_theta[indx] = 270 - thetaP_tmp
            elif (thetaP_tmp >= 0) and (thetaP_tmp <= 90):
                S.C_theta[indx] = 270 - thetaP_tmp

    #S.C_phi = S.C_phiE*(d2r)
    S.C_phi = np.zeros(len(S.C_phiE))
    for indx, theta_tmp in enumerate(S.C_theta):
        if (theta_tmp <= 45) and (theta_tmp >= 0):
            S.C_phi[indx] = np.arctan( np.tan(S.C_phiP[indx]*d2r) * np.cos(S.C_theta[indx]*d2r))
        if (theta_tmp <= 90) and (theta_tmp > 45):
            S.C_phi[indx] = np.arctan( np.tan((180 - S.C_psiP[indx])*d2r) * np.sin(S.C_theta[indx]*d2r))
        if (theta_tmp <= 135) and (theta_tmp > 90):
            S.C_phi[indx] = np.arctan( np.tan((180 - S.C_psiP[indx])*d2r) * np.sin((180 - S.C_theta[indx])*d2r))
        if (theta_tmp <= 225) and (theta_tmp > 135):
            S.C_phi[indx] = np.arctan( np.tan((S.C_phiP[indx])*d2r) * np.cos((180 - S.C_theta[indx])*d2r))
        if (theta_tmp <= 270) and (theta_tmp > 225) and (S.C_psiP[indx] <= 90) and (S.C_psiP[indx] >=0):
            S.C_phi[indx] = np.arctan( np.tan((S.C_psiP[indx])*d2r) * np.sin(abs(180 - S.C_theta[indx])*d2r))
        if (theta_tmp <= 270) and (theta_tmp > 225) and (S.C_psiP[indx] >= 270):
            S.C_phi[indx] = np.arctan( np.tan((S.C_psiP[indx] - 360)*d2r) * np.sin(abs(180 - S.C_theta[indx])*d2r))
        if (theta_tmp <= 315) and (theta_tmp > 270) and (S.C_psiP[indx] <= 90) and (S.C_psiP[indx] >=0):
            S.C_phi[indx] = np.arctan( np.tan((S.C_psiP[indx])*d2r) * np.sin((360 - S.C_theta[indx])*d2r))
        if (theta_tmp <= 315) and (theta_tmp > 270) and (S.C_psiP[indx] >= 270):
            S.C_phi[indx] = np.arctan( np.tan((S.C_psiP[indx] - 360)*d2r) * np.sin((360 - S.C_theta[indx])*d2r))
        if (theta_tmp <= 360) and (theta_tmp > 315):
            S.C_phi[indx] = np.arctan( np.tan((S.C_phiP[indx])*d2r) * np.cos((360 - S.C_theta[indx])*d2r))
    S.C_phi = S.C_phi*(-1)

    #print(S.C_phi)

    S.C_zeta = S.C_zeta*d2r
    S.C_theta = S.C_theta*d2r
    return S

def LOCAL_Calculate3DBasePoints(S):
    d2r = np.pi/180
    SIDE    = [indx for indx, name_tmp in enumerate(S.wname) if name_tmp[0]=='R']
    SIDE_L  = [indx for indx, name_tmp in enumerate(S.wname) if name_tmp[0]=='L']
    EYE = np.ones(len(S.wname))

    Rbp = np.sqrt(1/(
        (np.cos(S.C_BP_th*d2r))**2*(np.sin(S.C_BP_phi*d2r)**2)/(S.E_R[0]**2) +
        (np.sin(S.C_BP_th*d2r))**2*(np.sin(S.C_BP_phi*d2r)**2)/(S.E_R[1]**2) +
        (np.cos(S.C_BP_phi*d2r)**2)/(S.E_R[2]**2)))

    BP_x = Rbp*np.cos(S.C_BP_th*d2r)*np.sin(S.C_BP_phi*d2r)
    BP_y = Rbp*np.sin(S.C_BP_th*d2r)*np.sin(S.C_BP_phi*d2r)
    BP_z = Rbp*np.cos(S.C_BP_phi*d2r)

    c_x = np.cos(S.E_OA[2]*d2r);
    s_x = np.sin(S.E_OA[2]*d2r);
    c_y = np.cos(S.E_OA[1]*d2r);
    s_y = np.sin(S.E_OA[1]*d2r);
    c_z = np.cos(S.E_OA[0]*d2r);
    s_z = np.sin(S.E_OA[0]*d2r);
    A_list = [
        [c_y*c_z, c_z*s_x*s_y - c_x*s_z, s_x*s_z + c_x*c_z*s_y],
        [c_y*s_z, c_x*c_z + s_x*s_y*s_z, c_x*s_y*s_z - c_z*s_x],
        [-s_y,    c_y*s_x,               c_x*c_y]]
    A   = np.asarray(A_list)
    S.C_baseX = BP_x*A[0,0]+BP_y*A[0,1]+BP_z*A[0,2]
    S.C_baseY = BP_x*A[1,0]+BP_y*A[1,1]+BP_z*A[1,2]
    S.C_baseZ = BP_x*A[2,0]+BP_y*A[2,1]+BP_z*A[2,2]

    S.C_baseX = S.C_baseX + EYE*S.E_C[0]
    S.C_baseY = S.C_baseY + EYE*S.E_C[1]
    S.C_baseZ = S.C_baseZ + EYE*S.E_C[2]

    S.C_baseX[SIDE_L] = S.C_baseX[SIDE_L]*(-1);

    return S

def get_wholeS():
    S   = LOCAL_SetupDefaults()
    wselect = []
    S.wname = LOCAL_SetupWhiskerNames(wselect)
    S   = LOCAL_UnpackEquations(S)
    S   = LOCAL_CalculateParameters(S)
    S   = LOCAL_Calculate3DBasePoints(S)
    return S

if __name__=="__main__":
    S   = LOCAL_SetupDefaults()
    wselect = []
    S.wname = LOCAL_SetupWhiskerNames(wselect)
    S   = LOCAL_UnpackEquations(S)
    S   = LOCAL_CalculateParameters(S)
    S   = LOCAL_Calculate3DBasePoints(S)
