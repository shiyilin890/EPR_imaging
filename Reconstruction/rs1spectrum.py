####### Rapid-scan EPR spectrum code  ##########
# Yilin Shi, May 2019, shiyilin890@gmail.com
# choose 0: False, function skipped; 1: True, function performed
# Read Bruker data or csv; 1D or 2D data
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
    fn = 'ys188382'           # filename
    sample = '1mM CTPO, sf=5kHz, '   # in title
    tb = 100 * 10 ** -9       # time base, ns
    pts = 8192                # known from data
    CF = 92                   # center field
    sw = 70                   # sweep width
    sf = 5020                 # scan frequency,Hz
    fp = 9.28                 # first point, value must exist in (-1,1), ADJUST
    ph = -155                 # phase

    # function options
    use2d = 1                 # use 2D image data, also need Gmax, gstep, n below
    useBruker = 0             # use Bruker format data DTA and DSC
    plotraw = 0               # plot raw data
    adjustspectrum = 0        # I need to adjust parameters (first point and phase) (0: don't need)
    usefilter = 0             # use filter or not
    sH = 50                   # filter, [mG]
    usebackcor = 1            # use backcor function
    backcorpar = 0            # I know what backcor parameters to use! (0: don't know)
    uselw = 1                 # calculate linewidth, input peak and baseline position below
    useSNR = 1                # calculate S/R
                              # field range: [G], use cursor to pick values on the plot
    peakrange = [84, 96]      # includes the line, [G]
    noiserange1 = [60, 70]    # includes noise section 1, [G]
    noiserange2 = [116, 126]  # includes noise section 2, [G]

    if use2d:                 # use 2D image data
        Gmax = 10             # Maximum gradient, G/cm
        gstep = 1             # gradient step, G/cm
        Ng = int(2 * Gmax / gstep + 1)  # number of projections/gradients
        n = 11                # which projection to use
    else:                     # use 1D spectrum data
        Ng = 1
        n = 1

    ########### calculation starts, usually don't require input
    ###### import data
    if useBruker:              # use Bruker format data DTA and DSC
        # check if files exist
        if os.path.isfile(fn + '.DTA') and os.path.isfile(fn + '.DSC'):
            print('Bruker file ' + fn)
        elif os.path.isfile(fn + '.DTA') and (os.path.isfile(fn + '.DSC') == 0):
            print('Bruke data file \'DTA\' is present, but the parameter file \'DSC\' is missing.')
        else:
            print('Bruker file does not exit!')
            exit()
        # exit()

        Bruker = []
        fin = open(fn + '.DTA', 'rb')
        with open(fn + '.DTA', 'rb') as inh:
            indata = inh.read()
        for i in range(0, len(indata), 8):
            pos = struct.unpack('>d', indata[i:i + 8])
            Bruker.append(pos[0]);
        fin.close()

        if not len(Bruker) == pts*Ng*2:  # Imaginary and Real data, so X2
            print('Input wrong! check number of points, Gmax, Gstep ')
            exit()

        Bkre = np.zeros((pts, Ng), dtype=int)
        Bkim = np.zeros((pts, Ng), dtype=int)
        for i in range(Ng):
            for j in range(pts):
                Bkre[j][i] = Bruker[i * pts * 2 + j * 2]
                Bkim[j][i] = Bruker[i * pts * 2 + j * 2 + 1]
        np.savetxt(fn + "r.csv", Bkre, delimiter=",", fmt="%d")
        np.savetxt(fn + "i.csv", Bkim, delimiter=",", fmt="%d")
        # exit()

    # use csv format
    # check if file exist
    if os.path.isfile(fn + 'r.csv') and os.path.isfile(fn + 'i.csv'):
        print(fn)
    else:
        print('csv file does not exit!')
        exit()

    reader = csv.reader(open(fn + 'r.csv', "r"), delimiter=",")
    x = list(reader)
    R = np.array(x).astype("int")
    reader = csv.reader(open(fn + 'i.csv', "r"), delimiter=",")
    x = list(reader)
    I = np.array(x).astype("int")

    # check if input correct
    pts1 = R.shape[0]
    Ng1 = R.shape[1]
    if not pts1 == pts:
        print('Input wrong! Check number of points ')
        exit()
    elif not Ng1 == Ng:
        print('Input wrong! Check Gmax, gstep ')
        exit()

    t = np.arange(pts) * tb
    rr = R[:, n - 1]
    ii = I[:, n - 1]

    if plotraw:       # plot raw data
        plt.plot(t, ii, t, rr)
        plt.show()
        exit()

    bw = 10 ** 6  # bandwidth
    fwhm = 0.025  # post processing filter
    Acorr = 1
    Phcorr = 0
    Ph_exp = cmath.exp(1j * Phcorr / 180 * math.pi)
    rs = rr + 1j * ii
    rs = Acorr * (rs * Ph_exp).real + 1j * rs.imag
    method = 'fast'
    # method='slower'
    np.savetxt("temp/rsr.csv", rs.real, delimiter=",", fmt="%d")
    np.savetxt("temp/rsi.csv", rs.imag, delimiter=",", fmt="%d")
    np.savetxt("temp/t.csv", t, delimiter=",")

    ### save parameter file
    if os.path.isfile('temp/parfix.txt'):
        os.remove('temp/parfix.txt')
    with open ('temp/parfix.txt','a') as f:
        f.write(str(sw)+'\n')
        f.write(str(sf)+'\n')
        f.write(str(tb)+'\n')
        f.write(method+'\n')
        f.write(str(pts)+'\n')
        f.write(str(CF)+'\n')

    if adjustspectrum:
        window = Tk()
        window.title("Up and Down Scan Adjust")
        window.geometry("400x250+900+200")
        start = adjust(window)
        window.mainloop()

        f = open('temp/parvary.txt', "r")
        temp = f.read().splitlines()
        ph = float(temp[0])
        fp = float(temp[1])
        f.close()

    h, A, B = sinDecoBG(sw,sf,tb,t,rs,ph,fp,method)
    AB = A+B
    h = h+CF

    if usefilter:        # Filter
        x = xForInterp(h[-1] - h[0], len(AB))
        filterH = mygaussian(x, sH / 1000)
        AB = np.convolve(AB, filterH, 'same')

    if usebackcor:       # use 'backcor' background correction
        if backcorpar:   # I known what parameters to use!
            BG = backcor(h, AB, 6, 0.01, 'ah')
            AB = AB-BG

        else:            # try different parameters with GUI
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

    plt.subplot(211)
    plt.plot(h, AB, linewidth=1)
    plt.xlabel('Magnetic Field, G')
    plt.ylabel('Signal Amplitude, AU')
    plt.title('VHF rapid-scan of ' + sample + fn)

    plt.subplot(212)
    plt.plot(h, A, h, B, linewidth=1)
    plt.title('Up and Down Scans ')
    plt.show()
    # exit()

    step = abs(h[1] - h[0])
    m = np.where(abs(h - peakrange[0]) < step)[0][0]  # define peak
    n = np.where(abs(h - peakrange[1]) < step)[0][0]
    peak = AB[m:n]

    ### Calculate linewidth
    if uselw:
        swpeak = len(peak) * abs((h[-1] - h[0])) / len(AB)
        pphh = LW(peak,swpeak)
        print('linewidth,mG: ', pphh, '(lw at half height of absorption signal)')

    ### Calculate SNR
    if useSNR:
        p = np.where(abs(h-noiserange1[0])<step)[0][0]   # define baseline
        q = np.where(abs(h-noiserange1[1])<step)[0][0]
        noise1 = np.std(AB[p:q])     #p:q or q:p if h reverses
        p = np.where(abs(h-noiserange2[0])<step)[0][0]
        q = np.where(abs(h-noiserange2[1])<step)[0][0]
        noise2 = np.std(AB[p:q])     #p:q or q:p if h reverses
        noise = (noise1 + noise2) / 2
        peak_amp = max(peak)         #signal
        SNR = peak_amp / noise
        print('S/N: ', SNR)

