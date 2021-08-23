####### Rapid-scan EPR function code  ##########
# Yilin Shi, May 2019, shiyilin890@gmail.com
# choose 0: False, function skipped; 1: True, function performed
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

def fftM (t,a):
    if len(t) > 1:
        dt = t[1] - t[0]
    else:
        dt = t
    L = len(a)
    Wmax = 1/dt
    T = dt*L
    dw = 1/T
    Left = -Wmax / 2
    Right = Wmax / 2 -dw
    w = np.arange(Left, Right, dw)
    aa = np.fft.fft(a)
    A = np.fft.fftshift(aa)
    if len(w) < len(A):
        temp = w[1] - w[0]
        temp = w[-1] + temp
        w = np.append(w, temp)
    return w, A

# Interleaving data to one periodic cycle
def InterleavingCycles(tb,t,ts,rs,sf,Nfc,Ns,Wm,method):
    P = 1 / sf     #period
    inx = t > (t[-1] - P + tb)
    inc = t < Nfc * P
    K = np.arange(-Ns/2, Ns/2)
    rs_i = rs[inc]  #rs with Bfc full cyles
    M = len(rs_i)
    t_i = t[inc]    # time vector

    if method == 'slow':
        RS_i = np.zeros(Ns)
        for k in range(len(K)):
            tmp = cmath.exp(-1j*K[k]*Wm*t_i)
            RS_i[k] = rs_i * tmp
        aaa = np.fft.fftshift(RS_i)
        rs_x = np.fft.fft(aaa) * tb / ts[1] / Nfc / Ns
        print('method 1')
    elif method == 'fast':
        rs_i2 = np.append(rs_i, rs_i)
        v, RS = fftM(t_i, rs_i2)
        RS_i = np.interp(K/P, v, RS)    #Xquery,x,y,

        cnt = len(RS_i)
        if RS_i[-1] == 0:
            check = RS_i[0]
            for i in range(round(cnt/2)):
                if RS_i[i] == check:
                    RS_i[i] = 0
        else:
            check = RS_i[-1]
            for i in range(cnt-1, round(cnt/2),-1):
                if RS_i[i] == check:
                    RS_i[i] = 0

        aaa = np.fft.fftshift(RS_i)
        rs_x = np.fft.ifft(aaa)/M*Ns/2
        print("method2")
    else:
        print("Select appropriate case")
    return rs_x, RS_i, v, RS, rs_i, t_i, inc

def H_amplit(xi, yi, Fm):
    #Fm- scan frequency; Nh-number of harmonics
    t = 2 * math.pi * Fm * xi
    c1 = np.cos(t)
    s1 = np.sin(t)
    s2 = np.sin(2 * t)
    c2 = np.cos(2 * t)
    o = c1 * 0 + 1

    f = yi
    fs = (f + np.flipud(f)) / 2
    fa = (f - np.flipud(f)) / 2

    v1 = s1
    v2 = c2
    v3 = o

    a = np.matmul(v1, fs.conj().T)
    b = np.matmul(v2, fs.conj().T)
    c = np.matmul(v3, fs.conj().T)
    f = np.stack((a, b, c))

    a = np.array([np.matmul(v1, v1.conj().T), np.matmul(v1, v2.conj().T), np.matmul(v1, v3.conj().T)])
    b = np.array([np.matmul(v2, v1.conj().T), np.matmul(v2, v2.conj().T), np.matmul(v2, v3.conj().T)])
    c = np.array([np.matmul(v3, v1.conj().T), np.matmul(v3, v2.conj().T), np.matmul(v3, v3.conj().T)])
    T = np.stack((a, b, c))
    xs = np.matmul(np.linalg.inv(T), f)

    v1 = s2
    v2 = c1
    a = np.matmul(v1, fa.conj().T)
    b = np.matmul(v2, fa.conj().T)
    f = np.stack((a, b))

    a = np.array([np.matmul(v1, v1.conj().T), np.matmul(v1, v2.conj().T)])
    b = np.array([np.matmul(v2, v1.conj().T), np.matmul(v2, v2.conj().T)])
    T = np.stack((a, b))
    xa = np.matmul(np.linalg.inv(T),f)

                # sinx   sin2x   cosx   cos2x   const
    r = np.array([xs[0], xa[0], xa[1], xs[1], xs[2]])
    return r

#re-write sinDecoBG
def sinDecoBG (sw, sf, tb, t, rs, ph, fp, method):
    gamma = 1.7608e7
    g2f = gamma/(2*math.pi)    # = 2.8024e6
    Vmax = g2f*sw              # Max possible RS signal frequency
    Ns = 2*math.ceil(Vmax/sf)
    P = 1/sf
    ts = np.arange(Ns)*(P/Ns)
    Fmax = 1/(2*tb)            # Max frequency to be sampled without aliasimg
    ratio = Vmax/Fmax          # sampling ratio must be <1
    Wm = 2*math.pi*sf

    pts = rs.shape[0]

    # Error & Warning check
    Nc = round(1/(tb*sf))      # Points per period
    Nfc = math.floor(pts/Nc)   # Number of full cycles
    if Nfc == 0:
        print('ERROR: less than a full cycle')
    if ratio > 1:
        print('Warning: Sampling rate may not be sufficient')

    rs_i, RS_i, v, RS, rs_ii, t_i, inc=InterleavingCycles(tb, t, ts, rs, sf, Nfc, Ns, Wm, method)

    # Phase correction
    rs_ii = rs_i*cmath.exp(1j*(ph+0)/180*math.pi)

    # Position of 1st point correction
    shift = round(fp*Nc)   # Circular shift to find 1st point
    shift = shift % (len(rs_ii))
    shift1 = len(rs_ii)-shift
    aa = rs_ii[shift1::]
    bb = rs_ii[:shift1:]
    rs_iii = np.append(aa,bb)

    # Driving function
    rs = rs_iii
    t = ts
    tb = ts[1]   # new time base

    WF = -np.cos(2*math.pi*sf*t)
    W = gamma*sw/2*WF            # waverform
    dr = np.exp(-1j*np.cumsum(W)*tb) # driving function for 1 cycle

    # Separation Up from Down
    v, RS = fftM(t, rs)

    k = int(Ns/2)
    ind = v>0
    temp = np.fft.ifftshift(RS*ind)
    a = np.fft.ifft(temp)           # up scan
    jnd = v < 0
    temp = np.fft.ifftshift(RS*jnd)
    b = np.fft.ifft(temp)           # down scan
    aa = a[0:k]
    bb = b[k:Ns]
    bga = a[k:Ns]
    bgb = b[0:k]

    t1 = t[0:k]
    t2 = t[k:Ns]

    ## BG removal
    # 1st half
    x = 2*math.pi*sf*t1;
    c1 = np.cos(x);
    c2 = np.cos(2*x);
    s1 = np.sin(x);
    s2 = np.sin(2*x);

    # real and imaginary
    r = H_amplit(t2, bga.real, sf)
    #   sinx  sin2x  cosx  cos2x const
    # r=[xs(1) xa(1) xa(2) xs(2) xs(3)];
    bg = r[0]*s1+r[1]*s2+r[2]*c1+r[3]*c2+r[4]

    r = H_amplit(t2, bga.imag, sf)
    tmp = r[0]*s1+r[1]*s2+r[2]*c1+r[3]*c2+r[4]
    bg = bg+1j*tmp
    aaa = aa-bg

    # 2nd half
    x = 2*math.pi*sf*t2;
    c1 = np.cos(x);
    c2 = np.cos(2*x);
    s1 = np.sin(x);
    s2 = np.sin(2*x);

    # real and imaginary
    r = H_amplit(t1, bgb.real, sf)
    #   sinx  sin2x  cosx  cos2x const
    # r=[xs(1) xa(1) xa(2) xs(2) xs(3)];
    bg = r[0]*s1+r[1]*s2+r[2]*c1+r[3]*c2+r[4]

    r = H_amplit(t1, bgb.imag, sf)
    tmp = r[0]*s1+r[1]*s2+r[2]*c1+r[3]*c2+r[4]
    bg = bg+1j*tmp
    bbb = bb-bg

    ###########################################
    ## Deco for A
    drA = dr[0:k]
    aaaa = aaa*drA
    v, A = fftM(t,aaaa)
    v, D = fftM(t,drA)
    A = A/D

    ## Deco for B
    drB = dr[k:Ns]
    bbbb = bbb*drB
    v, B = fftM(t,bbbb)
    v, D = fftM(t,drB)
    B = B/D

    h = v*2*math.pi/gamma
    ind = np.abs(h)<sw/2
    A = A[ind]
    B = B[ind]
    h = h[ind]

    A = (np.flipud(A)).imag
    B = (np.flipud(B)).imag
    return h,A, B

def backcor(n, y, ord, s, fct):
    #Background estimation by minimizing a non-quadratic cost function; reference[1]
    # Rescaling
    N = len(n)
    i = np.argsort(n)
    n.sort()
    y = y[i]
    maxy = np.amax(y)
    dely = (maxy - np.amin(y)) / 2
    n = 2 * (n[:] - n[N - 1]) / (n[N - 1] - n[0]) + 1
    y = (y[:] - maxy) / dely + 1

    # Vandermonde matrix
    p = np.arange(ord + 1)
    T1 = np.matlib.repmat(n, ord + 1, 1)
    T2 = np.matlib.repmat(p, N, 1)
    TT = np.power(T1.T, T2)
    T3 = np.linalg.pinv(np.matmul(TT.T, TT))
    Tinv = np.matmul(T3, TT.T)

    # Initialisation (least-squares estimation)
    a = np.matmul(Tinv, y)
    z = np.matmul(TT, a)

    # Other variables
    alpha = 0.99 * 1 / 2  # Scale parameter alpha
    it = 0  # Iteration number

    # LEGEND
    while True:
        it = it + 1       # Iteration number
        zp = z            # Previous estimation
        res = y - z       # Residual

        # Estimate d
        if (fct == 'sh'):
            d = (res * (2 * alpha - 1)) * (abs(res) < s) + (-alpha * 2 * s - res) * (res <= -s) + (
                        alpha * 2 * s - res) * (res >= s)
        elif (fct == 'ah'):
            d = (res * (2 * alpha - 1)) * (res < s) + (alpha * 2 * s - res) * (res >= s)
        elif (fct == 'stq'):
            d = (res * (2 * alpha - 1)) * (abs(res) < s) - res * (abs(res) >= s)
        elif (fct == 'atq'):
            d = (res * (2 * alpha - 1)) * (res < s) - res * (res >= s)

        # Estimate z
        a = np.matmul(Tinv, y + d)
        z = np.matmul(TT, a)

        z1 = sum(np.power(z - zp, 2)) / sum(np.power(zp, 2))
        if z1 < 1e-9:
            break

    # Rescaling
    j = np.argsort(i)
    z = (z[j] - 1) * dely + maxy
    a[0] = a[0] - 1
    a = a * dely  # + maxy
    return z

def xForInterp(sweep,N):
    x = np.arange(N) / (N - 1)
    sw2 = sweep / 2
    x = -sw2 + x * sweep
    return x

def mygaussian(h,FWHM):
    Hpp = FWHM/1.1774     #sqrt(2*log(2))
    A = 0.7979/Hpp        #sqrt(2/pi)
    x = 2 * np.power(h / Hpp, 2)
    y = A * np.exp(-x)
    return y

def zeroline(spectrum, extent):
    L = len(spectrum)
    edge = round(extent * L)
    if edge == 0:
        edge=1

    Sleft = sum(spectrum[0:edge]) / edge
    Sright = sum(spectrum[L - edge:L+1]) / edge
    s = len(spectrum)

    LL = np.arange(1,L+1)
    tmp = Sleft + (Sright - Sleft) / L * LL
    if len(tmp) != s:
        tmp = tmp.T
    res = spectrum - tmp
    return res

def LW(spectrum, sweep):
    sp = zeroline(spectrum, 0.05)
    N = len(sp)
    n = N - 1
    x = np.arange(n+1)
    xn = (x - n / 2) / n

    M = 2000
    n = M - 1
    x = np.arange(n + 1)
    xm = (x - n / 2) / n

    if M > N:
        sp = np.interp(xm, xn, sp)   #don't need zero baseline

    mx = max(sp)
    inx = sp > (mx / 2)
    N = len(sp)
    linewidth = round(1000 * sum(inx) / N * sweep)
    return linewidth


class mclass:   #for backcor function
    def __init__(self,window):
        self.window = window

        self.l1 = Label(window, text="Background Correction      ", font=("Helvitca", 14))
        self.l2 = Label(window, text="Order     ")
        self.l3 = Label(window, text="Threshold      ")
        self.l4 = Label(window, text="Function      ")
        self.l5 = Label(window, text=" ")

        self.l1.grid(row=1, column=1, sticky=W, columnspan=5)
        self.l2.grid(row=2, column=1, sticky=E)
        self.l3.grid(row=3, column=1, sticky=E)
        self.l4.grid(row=4, column=1, sticky=E)
        self.l5.grid(row=5, column=1, sticky=E)

        self.e2 = Entry(window)
        self.e2.insert(0, 0.01)
        self.e2.grid(row=3, column=2, sticky=W)

        self.v1 = StringVar(window)
        self.v1.set("6")
        self.w1 = OptionMenu(window, self.v1, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
        self.w1.grid(row=2, column=2, sticky=W)

        self.v3 = StringVar(window)
        self.v3.set("ah")
        self.w = OptionMenu(window, self.v3, "sh", "ah", "stq", "atq")
        self.w.grid(row=4, column=2, sticky=W)

        self.b1 = Button(window, text='  Select and Plot  ', command=self.plotbg)
        self.b1.grid(row=6, column=1, sticky=E)
        # self.b2 = Button(window, text='  OK  ', command=window.destroy)  #possible close button
        # self.b2.grid(row=6, column=2, sticky=E)
        self.l6 = Label(window, text="When done, close window \nand background plot to proceed. ")
        self.l6.grid(row=7, column=1, sticky=W, columnspan=2)

    def plotbg(self):
        plt.clf()
        ord = int(self.v1.get())
        s = float(self.e2.get())
        fct = self.v3.get()
        reader = csv.reader(open('temp/xdata.csv', "r"), delimiter=",")
        x = list(reader)
        h1 = np.array(x).astype("float")
        h = h1[:, 0]
        reader = csv.reader(open('temp/ydata.csv', "r"), delimiter=",")
        x = list(reader)
        AB1 = np.array(x).astype("float")
        AB = AB1[:,0]

        BG = backcor(h, AB, ord, s, fct)
        np.savetxt("temp/ybackcor.csv", AB-BG, delimiter=",")

        plt.plot(h, AB, h,BG)
        plt.gca().legend(('Orginal', 'Background'))
        plt.title('Background Plot ')
        # plt.clf()
        plt.show()
        plt.close()    #must have this so this plot will close


class adjust:   #for first point and phase adjustment
    def __init__(self,window):
        self.window = window

        self.l1 = Label(window, text="Spectrum Parameters Adjustment", font=("Helvitca", 12))
        self.l2 = Label(window, text="Phase     ")
        self.l3 = Label(window, text="First Point      ")
        self.l4 = Label(window, text="(value exits in (-1, 1))      ")
        self.l5 = Label(window, text=" ")

        self.l1.grid(row=1, column=1, sticky=W, columnspan=5)
        self.l2.grid(row=2, column=1, sticky=E)
        self.l3.grid(row=3, column=1, sticky=E)
        self.l4.grid(row=4, column=1, sticky=E)
        self.l5.grid(row=5, column=1, sticky=E)

        f = open('temp/parvary.txt', "r")
        temp = f.read().splitlines()
        ph = float(temp[0])
        fp = float(temp[1])
        f.close

        self.e1 = Entry(window)
        self.e1.insert(0, ph)
        self.e1.grid(row=2, column=2, sticky=W)
        self.e2 = Entry(window)
        self.e2.insert(0, fp)
        self.e2.grid(row=3, column=2, sticky=W)

        self.b1 = Button(window, text='  Select and Plot  ', command=self.plotbg)
        self.b1.grid(row=6, column=1, sticky=E)
        self.l6 = Label(window, text="When done, close window \nand plot to proceed. ")
        self.l6.grid(row=7, column=1, sticky=W, columnspan=2)

    def plotbg(self):
        plt.clf()
        ph = float(self.e1.get())
        fp = float(self.e2.get())
        reader = csv.reader(open('temp/rsr.csv', "r"), delimiter=",")
        x = list(reader)
        y = np.array(x).astype("float")
        rsr = y[:, 0]
        reader = csv.reader(open('temp/rsi.csv', "r"), delimiter=",")
        x = list(reader)
        y = np.array(x).astype("float")
        rsi = y[:,0]
        rs = rsr + 1j * rsi
        reader = csv.reader(open('temp/t.csv', "r"), delimiter=",")
        x = list(reader)
        y = np.array(x).astype("float")
        t = y[:,0]

        f = open('temp/parfix.txt', "r")
        temp = f.read().splitlines()
        sw = float(temp[0])
        sf = float(temp[1])
        tb = float(temp[2])
        method = temp[3]
        f.close()

        if os.path.isfile('temp/parvary.txt'):
            os.remove('temp/parvary.txt')
        with open('temp/parvary.txt', 'a') as f:
            f.write(str(ph) + '\n')
            f.write(str(fp) + '\n')

        h, A, B = sinDecoBG(sw, sf, tb, t, rs, ph, fp, method)

        plt.plot(h, A, h, B)
        plt.title('Up and down scan ')
        # plt.clf()
        plt.show()
        plt.close()    #must have this so this plot will close




#reference
# [1] Deconvolution and image reconstruction from Dr. Mark Tseitlin's MATLAB code
# Mark Tseitlin, Joshua R. Biller, Hanan Elajaili, Valery Khramtsov,
# Ilirian Dhimitruka, Gareth R. Eaton, and Sandra S. Eaton
# New spectral-spatial imaging algorithm for full EPR spectra of multiline nitroxides
# and pH sensitive trityl radicals
# J Magn Reson. 2014 Aug; 245: 150–155.


# [2] Background correction code 'backcor':
# V. Mazet, C. Carteret, D. Brie, J. Idier, B. Humbert. Chemom. Intell. Lab. Syst. 76 (2), 2005.
# V. Mazet, D. Brie, J. Idier. Proceedings of EUSIPCO, pp. 305-308, 2004.
# V. Mazet. PhD Thesis, University Henri PoincarÃ© Nancy 1, 2005.
# 22-June-2004, Revised 19-June-2006, Revised 30-April-2010,
# Revised 12-November-2012 (thanks E.H.M. Ferreira!)
# Comments and questions to: vincent.mazet@unistra.fr.

# [3] Read Bruker data DTA and DSC in Python:
# https://github.com/mortenalbring/BES3Tconvert/blob/master/conv.py

