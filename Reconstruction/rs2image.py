####### Rapid-scan EPR Imaging code  ##########
# Yilin Shi, May 2019, shiyilin890@gmail.com
# choose 0: False, function skipped; 1: True, function performed
# Read Bruker data or csv; 2D data only
# Reference [1,2,3]

import os
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import csv
from tkinter import *
import math
import cmath
import struct
import numpy.matlib
from rs3func import *

if __name__ == '__main__':

    ##### Input parameters
    # adjust spectrum parameters (phase and first point) in step1 'rs1spectrum.py'
    # Bruker file converted to csv in step 1
    fn = 'ys188382'
    sample = '1mM CTPO, sf=5kHz, '  #in title
    tb = 100 * 10 ** -9 # time base
    pts = 8192          # known from data
    Gmax = 10           # Maximum gradient, G/cm
    gstep = 1           # gradient step, G/cm
    CF = 92             # center field
    sw = 70             # sweep width
    sf = 5020           # scan frequency,Hz
    fp = 9.28           # first point  (value must exist in (-1, 1)
    ph = -155           # phase
    h_max = 30          # final plot half field range
    zlim = [-1.5, 1.5]  # final plot spatial range
    fov = 5             # field of view
    Lw_min = 0.5        # minimal line width
    Ng = int(2 * Gmax / gstep + 1)  # number of projections/gradients

    # function options
    plotraw = 0               # plot raw data
    step2 = 1                 # I need to deconvolve projections (0: prj already deconvolved)
    plotsino = 0              # plot sinograph
    usebothch = 0             # use real and imaginary channels
    rawimg = 0                # plot raw image
    spatsl = 1                # plot spatial slice of the image
                              # for spectral slice:
    specsl =1                 # plot spectral slice of the image
    usebackcor = 1            # use backcor function
    backcorpar = 1            # I know what backcor parameters to use! (0: don't know)
    uselw = 1                 # calculate linewidth, input peak and baseline position below
    useSNR = 1                # calculate S/R
    # position of spectral and spatial slice pick on image and input at the bottom

    ########### calculation starts, usually don't require input
    # check if file exist
    if os.path.isfile(fn + 'r.csv') and os.path.isfile(fn + 'i.csv'):
        print(fn)
    else:
        print('File not exit!')
        exit()

    # import data in csv format
    reader = csv.reader(open(fn + 'r.csv', "r"), delimiter=",")
    x = list(reader)
    R = np.array(x).astype("int")
    reader = csv.reader(open(fn + 'i.csv', "r"), delimiter=",")
    x = list(reader)
    I = np.array(x).astype("int")

    # check if matrix dimension match
    pts1 = R.shape[0]
    Ng1 = R.shape[1]
    if not pts1 == pts:
        print('Input wrong! check number of points ')
        exit()
    elif not Ng1 == Ng:
        print('Input wrong! check Gmax, gstep ')
        exit()
    elif Ng1 == 1:
        print('Need 2D dataset, but 1D data is given')
        exit()

    t = np.arange(pts) * tb

    if plotraw:  # plot raw data
        n = 11
        rr = R[:, n - 1]
        ii = I[:, n - 1]
        plt.plot(t, ii, t, rr)
        plt.show()
        exit()

    bw = 10 ** 6  # bandwidth
    fwhm = 0.025  # post processing filter
    method = 'fast'
    # method='slower'

    Nh1 = round(2 * sw / fwhm)
    h_i = np.linspace(-sw / 2, +sw / 2, Nh1)

    AA = np.zeros((Nh1, Ng), dtype=complex)
    BB = np.zeros((Nh1, Ng), dtype=complex)

    # Step2: projection deconvolution
    # if you have deconvolved the data, you can skip.
    if step2:
        for k in range(Ng):
            rr = R[:, k]
            ii = I[:, k]
            rs = rr + 1j * ii

            if (0):  # if one wants channel balance
                Acorr = 1
                Phcorr = 0
                Ph_exp = cmath.exp(1j * Phcorr / 180 * math.pi)
                rs = Acorr * (rs * Ph_exp).real + 1j * rs.imag;

            h, A, B = sinDecoBG(sw, sf, tb, t, rs, ph, fp, method)
            h, Ax, Bx = sinDecoBG(sw, sf, tb, t, rs, (ph + 90), fp, method)

            A = A + 1j * Ax
            B = B + 1j * Bx
            Ai = np.interp(h_i, h, A)  # Xquery,x,y,
            cnt = len(Ai)
            if Ai[-1] == 0:
                check = Ai[0]
                for i in range(round(cnt / 2)):  # 37782
                    if Ai[i] == check:
                        Ai[i] = 0
            else:
                check = Ai[-1]
                for i in range(cnt - 1, round(cnt / 2), -1):  # 37782
                    if A[i] == check:
                        A[i] = 0

            Bi = np.interp(h_i, h, B)  # Xquery,x,y,
            cnt = len(Bi)
            if Bi[-1] == 0:
                check = Bi[0]
                for i in range(round(cnt / 2)):  # 37782
                    if Bi[i] == check:
                        Bi[i] = 0
            else:
                check = Bi[-1]
                for i in range(cnt - 1, round(cnt / 2), -1):  # 37782
                    if B[i] == check:
                        B[i] = 0
            AA[:, k] = Ai
            BB[:, k] = Bi

        AB = AA + BB
        np.savetxt("ABr.csv", AB.real, delimiter=",", fmt="%d")
        np.savetxt("ABi.csv", AB.imag, delimiter=",", fmt="%d")
        print('Deconvolved projections is saved as ABr, ABi.csv ')

    else:
        if os.path.isfile('ABr.csv') and os.path.isfile('ABi.csv'):
            print('Use saved deconvolved projections.')
        else:
            print('Deconvolved projections not exit!\n Please deconvolve projections first.')
            exit()
        reader = csv.reader(open('ABr.csv', "r"), delimiter=",")
        x = list(reader)
        ABR = np.array(x).astype("int")
        reader = csv.reader(open('ABi.csv', "r"), delimiter=",")
        x = list(reader)
        ABI = np.array(x).astype("int")
        AB = ABR+1j*ABI

    method1 = 'tikh_0'
    tol_tikh = 40
    Max_harm = 100
    h = h_i
    dz = Lw_min/Gmax
    Npoints = round(1.2*fov/dz)
    z = np.linspace(-fov/2,+fov/2, Npoints)
    g = np.linspace(-Gmax, +Gmax, Ng)
    Nh = len(h)
    Nz = len(z)

    if plotsino:  #plot sinogram
        sino = (AB.real).T
        imgplot = plt.imshow(sino, aspect='auto', extent=[h[0] + CF, h[-1] + CF, g[0], g[-1]])
        plt.ylabel('Gradient, G/cm')
        plt.xlabel('Magnetic Field, G')
        plt.title('Projections Stack Plot of ' + sample + fn)
        plt.show()
        exit()

    phnn = 0     # phase adjust if want
    CC = AB*cmath.exp(1j/180*math.pi*phnn)
    CC[0, :] = 0
    CC[-1, :] = 0
    RR = CC.real
    Rc = RR

    if usebothch:   #use real and imaginary channel
        II = CC.imag
        temp = II.conj().T
        hb = sp.signal.hilbert(temp)
        hb = hb.conj().T
        imH = hb.imag
        # plt.plot(h,imH[:,10],h,RR[:,10])
        # plt.plot(h,imH[10,:],h,RR[10,:])
        # plt.show()
        RI = RR-imH
        Rc = RI

    hc = h
    Nhc = len(hc)
    SW = hc[-1]-hc[0]
    TT = np.zeros((Ng, Nz, Nhc), dtype=complex)

    for n in range(Ng):
        v, A = fftM(hc, hc)
        w = np.fft.ifftshift(2*math.pi*v)
        W, Z = np.meshgrid(w, z)
        temp = np.multiply(Z,W)
        T = np.exp(1j*temp*g[n])
        TT[n, :, :] = T

    PR = np.fft.fft(Rc.conj().T)
    x_PH = np.zeros((Nz, Nhc), dtype=complex)

    # Regul operatotrs
    v0 = np.ones(Nz)
    v1 = np.ones(Nz-1)
    D0 = np.diag(v0)
    D1 = np.diag(v1, k=-1)+ np.diag(-v1, k=1)
    DD1 = np.matmul(D1.conj().T, D1)
    DD0 = np.matmul(D0.conj().T, D0)

    fH = 0
    if method1 == 'tikh_0':
        for m in range(fH, Max_harm):
            L = TT[:, :, m]
            b = PR[:, m]
            LL = np.matmul(L.conj().T, L)
            temp1 = LL + tol_tikh * DD0
            temp2 = L.conj().T
            xx1 = np.linalg.solve(temp1,temp2)
            xx = np.matmul(xx1, b)
            x_PH[:, m] = xx

    elif method1 == 'tikh_0':
        for m in range(fH, Max_harm):
            L = TT[:, :, m]
            b = PR[:, m]
            LL = np.matmul(L.conj().T, L)
            temp1 = LL + tol_tikh * DD1
            temp2 = L.conj().T
            xx1 = np.linalg.solve(temp1,temp2)
            xx = np.matmul(xx1, b)
            x_PH[:, m] = xx
    else:
        print('method unavailable.')
        exit()

    x_Ph = (np.fft.ifft(x_PH)).real
    np.savetxt("image.csv", x_Ph, delimiter=",")
    print('image matrix is generated: image.csv')

    if rawimg:    # raw image plot
        imgplot = plt.imshow(x_Ph,aspect='auto',extent=[hc[0],hc[-1],z[0],z[-1]])
        plt.xlabel('Magnetic Field, G')
        plt.ylabel('Position, cm')
        plt.title('VHF Rapid-scan of ' + sample + fn)
        plt.show()
        exit()

    #### cut image
    inx = np.abs(hc) < h_max
    jnx = ((z > zlim[0]) * (z < zlim[1]))
    h = hc[inx]
    zj = z[jnx]

    steph = abs(hc[1]-hc[0])
    p1 = np.where(abs(hc + h_max) < steph)[0][0]
    q1 = np.where(abs(hc - h_max) < steph)[0][0]
    stepz = abs(z[1]-z[0])
    p2 = np.where(abs(z - zlim[0]) < stepz)[0][0]
    q2 = np.where(abs(z - zlim[1]) < stepz)[0][0]
    I = x_Ph[p2:q2, p1:q1]   #z * h matrix
    h = h+CF

    plt.subplot(211)
    imgplot = plt.imshow(I,aspect='auto',extent=[h[0],h[-1],zj[0],zj[-1]])
    plt.xlabel('Magnetic Field, G')
    plt.ylabel('Position, cm')
    plt.title('VHF Rapid-scan of ' + sample + fn)
    # plt.show()
    # exit()

    ######### spectral slice(s)
    if specsl:
        z1 = -0.2731    #pick spatial position you want to cut the spectral slice
        z2 = 0.02101
        p1 = np.where(abs(zj - z1) < stepz)[0][0]
        p2 = np.where(abs(zj - z2) < stepz)[0][0]
        sl1 = I[p1, :]
        sl2 = I[p2, :]

        if usebackcor:        # use 'backcor' background correction
            AB = sl2
            if backcorpar:    # I known what parameters to use!
                BG = backcor(h, sl2, 4, 0.01, 'sh')
                AB = AB-BG

            else:     # try different parameters
                np.savetxt("temp/xdata.csv", h, delimiter=",")
                np.savetxt("temp/ydata.csv", AB, delimiter=",")
                window = Tk()
                window.title("backcor")
                window.geometry("300x250+900+200")
                start = mclass(window)
                window.mainloop()

                reader = csv.reader(open('temp/ybackcor.csv', "r"), delimiter=",")
                x = list(reader)
                AB1 = np.array(x).astype("float")
                AB = AB1[:,0]
            sl2=AB

        plt.subplot(212)
        # plt.plot(h, sl1, h, sl2, linewidth=1)  #plot 2 slices
        plt.plot(h, sl2, linewidth=1)
        plt.title('Spectral Slice ')
        plt.show()

        m = np.where(abs(h - 87) < steph)[0][0]
        n = np.where(abs(h - 95) < steph)[0][0]
        peak = sl2[m:n]
        ###### Calculate linewidth
        if uselw:
            swpeak = len(peak) * abs((h[-1] - h[0])) / len(sl2)
            pphh = LW(peak, swpeak)
            print('linewidth,mG: ', pphh)

        ###### Calculate SNR
        if useSNR:
            p = np.where(abs(h - 63) < steph)[0][0]
            q = np.where(abs(h - 70) < steph)[0][0]
            noise1 = np.std(sl2[p:q])  # p:q or q:p if h reverses
            p = np.where(abs(h - 113) < steph)[0][0]
            q = np.where(abs(h - 120) < steph)[0][0]
            noise2 = np.std(sl2[p:q])  # p:q or q:p if h reverses
            noise = (noise1 + noise2) / 2
            peak_amp = max(peak)       # signal
            SNR = peak_amp / noise
            print('S/N: ', SNR)

    ######### spatial slice
    if spatsl:
        field = 91.74                   # pick field on plot by cursor
        p = np.where(abs(h - field) < steph)[0][0]
        spatial = I[:, p]

        plt.subplot(212)
        plt.plot(zj, spatial, linewidth=1)
        plt.title('Spatial Slice ')
        plt.show()

