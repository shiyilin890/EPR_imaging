#python 2.7
import ttk
from Tkinter import *
from ttk import *
import tkFileDialog as fd
import XeprAPI  # Load the XeprAPI Module
import numpy as np
import math 
import time
import platform
import binascii
import serial
import socket
from function1 import *    #import functions
import tkMessageBox
import os
import sys
import gc

Xepr=XeprAPI.Xepr()
Xepr.XeprCmds.aqExpSelect(1,'Experiment')

root=Tk()
root.title('Rapid Scan Experiment Control')

grad_B0Offset=0
path='/home/xuser/Desktop/VHFSerial/temp/'

## 3 FUNCTIONS FOR RUN ##

def runRS():
    
    Xepr.XeprCmds.aqExpSelect(1,'Experiment')
    # Scan 1
    RSCD(float(e104.get()),float(e102.get()),90,int(e103.get()),float(e402.get()))
    MainMag(float(e101.get()),float(e401.get()))
    ZGrad(float(e105.get()),float(e403.get()))
    cur_exp=Xepr.XeprExperiment() # Get current experiment
    #ExperimentName = cur_exp.aqGetExpName() # Get experiment name

    #hidden=Xepr.XeprExperiment('AcqHidden')
    #Xepr.XeprCmds.aqExpDesc("AcqHidden")
    #hidden['NoOfPoints'].value=8192    #int(cur_exp['NbPoints'].value) # Get number of points
    #hidden['NoOfAverages'].value=4096
    #cur_exp.aqSetIntParValue(cur_exp,"NoOfPoints'",0,None,4096)
    tic1 = time.time()  ###$$$
    cur_exp.aqExpRunAndWait() # Equivalent to pressing 'play'
    toc1 = time.time()  ###$$$
    print (toc1-tic1)

    dset = Xepr.XeprDataset()
    ordinate=dset.O


    with open(path+'address.txt', 'r') as f:
        fold = f.read()   
    title3=e108.get()
    filename3=fold+'/'+ title3  #location
    Xepr.XeprCmds.vpSave('Current', 'Primary', title3, filename3)
    toc3 = time.time()  ###$$$
    print (toc3-tic1)

    if(abs(float(e105.get()))>9.9):
        ZGrad(0,1)

def runRSFSDD():
    Xepr.XeprCmds.aqExpSelect(1,'Experiment')
    ZGrad(float(e105.get()),float(e403.get()))

    stepnum = int (float(e201.get())/float(e202.get()) +1)
    FSarray=np.linspace(float(e101.get())-(float(e201.get())/2),float(e101.get())+(float(e201.get())/2), stepnum)
    q=0
    cur_exp=Xepr.XeprExperiment() # Get current experiment
    hidden=Xepr.XeprExperiment('AcqHidden')
    #ExperimentName=cur_exp.aqGetExpName() # Get experiment name
    xPts=hidden['NoOfPoints'].value    #int(cur_exp['NbPoints'].value) # Get number of points
    dset2D1=Xepr.XeprDataset(size=(int(xPts),int(len(FSarray))),xeprset = 'primary',iscomplex = True) # Create 2D dataset
    dset2D2=Xepr.XeprDataset(size=(int(xPts),int(len(FSarray))),xeprset = 'secondary',iscomplex = True)
    
    # Loop Through Field Values
    for i in FSarray:
        print i
        # Scan 1        
        MainMag(float(i),float(e401.get()))
        RSCD(float(e104.get()),float(e102.get()),90,int(e103.get()),float(e402.get()))
        time.sleep(1)
        cur_exp=Xepr.XeprExperiment() # Get current experiment
        ExperimentName=cur_exp.aqGetExpName() # Get experiment name
        cur_exp.aqExpRunAndWait() # Equivalent to pressing 'play'
        dset1D1=Xepr.XeprDataset()
        dset2D1.O[q] [:]=dset1D1.O[:]
        Xepr.XeprCmds.aqExpSelect(1,'Experiment')
        
        # Scan 2
        MainMag(-float(i),float(e401.get()))
        RSCD(float(e104.get()),float(e102.get()),270,int(e103.get()),float(e402.get()))
        time.sleep(1)
        cur_exp=Xepr.XeprExperiment() # Get current experiment
        ExperimentName=cur_exp.aqGetExpName() # Get experiment name
        cur_exp.aqExpRunAndWait() # Equivalent to pressing 'play'
        dset1D2=Xepr.XeprDataset()
        dset2D2.O[q] [:]=dset1D2.O[:]
        
        # Up the index counter 'q'
        q=q+1
    
    dset2D1.X=dset1D1.X
    dset2D1.Y=FSarray
    dset2D1.update(refresh=True)
    dset2D1.setXeprSet('primary')


    with open(path+'address.txt', 'r') as f:
        fold = f.read()
    title1=e106.get()
    filename1=fold+'/'+ title1  #location
    Xepr.XeprCmds.vpSave('Current', 'Primary', title1, filename1)
    
    dset2D2.X=dset1D2.X
    dset2D2.Y=FSarray
    dset2D2.update(refresh=True)
    dset2D2.setXeprSet('secondary')
    
    title2=e107.get()
    filename2=fold+'/'+ title2  #location
    Xepr.XeprCmds.vpSave('Current', 'Secondary', title2, filename2)
        
    Xepr.XeprCmds.prDiff('Current','Primary','All','Current',1,0,1)
    
    title3=e108.get()
    filename3=fold+'/'+ title3  #location
    Xepr.XeprCmds.vpSave('Current', 'Result', title3, filename3)

def runRSgrad():
    tic = time.time()  #####
    Xepr.XeprCmds.aqExpSelect(1,'Experiment')
    gstep=float(e302.get())
    gmax= float(e301.get())
    Garray=np.linspace(-gmax, gmax, 2*gmax/gstep+1)
    q=0
    cur_exp=Xepr.XeprExperiment() # Get current experiment
    hidden=Xepr.XeprExperiment('AcqHidden')
    #ExperimentName=cur_exp.aqGetExpName() # Get experiment name
    xPts=hidden['NoOfPoints'].value       # Get number of points
    dim_pt=int(xPts)
    dim_g =int(len(Garray))

###$$$Create 2D dataset
    #dset2D1=Xepr.XeprDataset(size=(dim_pt,dim_g),xeprset = 'primary',iscomplex = True) 
    #dset2D2=Xepr.XeprDataset(size=(dim_pt,dim_g),xeprset = 'secondary',iscomplex = True)
    
    hidden=Xepr.XeprExperiment('AcqHidden')
    if Avg.get()==1:
        hidden['NoOfAverages'].value=int((e305.get()))
        #hidden['NOnBoardAvgs'].value=256
    toc6 = time.time()  #####
    print (toc6-tic)

    specr=np.zeros(( dim_pt,dim_g ))
    speci=np.zeros(( dim_pt,dim_g ))
    #spec2r=np.zeros(( dim_pt,dim_g ))
    #spec2i=np.zeros(( dim_pt,dim_g ))

    # Loop Through Gradient Values
    for g in Garray:  # g is gradient value
        if Avg.get()==1:
	
	    if  ( abs(g+float(e303.get()))<0.001 ):
		hidden['NoOfAverages'].value=int((e306.get()))
                #hidden['NOnBoardAvgs'].value=256
	    elif( abs(g+float(e304.get()))<0.001 ):
		hidden['NoOfAverages'].value=int((e307.get()))
                #hidden['NOnBoardAvgs'].value=256
	    elif( abs(g-float(e304.get())-gstep)<0.001 ):
		hidden['NoOfAverages'].value=int((e306.get()))
                #hidden['NOnBoardAvgs'].value=256
	    elif( abs(g-float(e303.get())-gstep)<0.001 ):
		hidden['NoOfAverages'].value=int((e305.get()))
                #hidden['NOnBoardAvgs'].value=256

	    print str(g)+', avg.: '+str(hidden['NoOfAverages'].value)

        else:
            print g
        
        tic1=time.time()
        ### Scan 1        
        MainMag(float(e101.get())-(grad_B0Offset*(float(g)*float(g))),float(e401.get()))
        ZGrad(float(g),float(e403.get()))
        RSCD(float(e104.get()),float(e102.get()),90,int(e103.get()),float(e402.get()))
        #toc5 = time.time()   ######
        #print (toc5-tic1)

        #time.sleep(0.5)
        cur_exp=Xepr.XeprExperiment()         # Get current experiment
        #ExperimentName=cur_exp.aqGetExpName() # Get experiment name

	tic3=time.time()
        cur_exp.aqExpRunAndWait()             # Equivalent to pressing 'play'
        toc3 = time.time()  ######
        print (toc3-tic3)
   
	if  ( abs(g+10)<0.6 ) or (abs(g-10)<0.6):
	    ZGrad(0, 1)

	tic4 = time.time()
        dset1D1=Xepr.XeprDataset()
	#print (type (Xepr.XeprDataset()))
	#print (type (dset1D1))
	specr[:,q]=dset1D1.O[:].real
	speci[:,q]=(dset1D1.O[:]).imag
        #print (dset1D1.O[:])
        #dset2D1.O[q] [:]=dset1D1.O[:]  ###$$$

        #Xepr.XeprCmds.aqExpSelect(1,'Experiment') ###$$$

		
        toc1 = time.time()  #####
        print (toc1-tic4)    
	print (toc1-tic1)    

	'''
        tic2=time.time()
        ### Scan 2
        MainMag(-float(e101.get())+(grad_B0Offset*(float(g)*float(g))),float(e401.get()))
        ZGrad(-float(g),float(e403.get()))
        RSCD(float(e104.get()),float(e102.get()),270,int(e103.get()),float(e402.get()))
        cur_exp=Xepr.XeprExperiment()         # Get current experiment
        #ExperimentName=cur_exp.aqGetExpName() # Get experiment name
        cur_exp.aqExpRunAndWait()             # Equivalent to pressing 'play'
        dset1D2=Xepr.XeprDataset()
        #dset2D2.O[q] [:]=dset1D2.O[:]    ###$$$
	spec2r[:,q]= dset1D2.O[:].real
	spec2i[:,q]= (dset1D2.O[:] ).imag
	#print(dset1D2.O[:])
        toc2 = time.time()
        #print (toc2-tic2)
        print (toc2-tic1)
	'''
	gc.collect()
	del dset1D1
	#del dset1D2
        q=q+1   # index counter 'q'

    ZGrad(0, 1)
    tic4=time.time()   #####

    #dset2D1.X=dset1D1.X  ###$$$
    #dset2D1.Y=Garray     ###$$$
    #dset2D1.update(refresh=True)   ###$$$
    #dset2D1.setXeprSet('primary')  ###$$$
    
    with open(path+'address.txt', 'r') as f:
        fold = f.read()

    ''' ###$$$
    #title1=e106.get()
    #filename1=fold+'/'+ title1  #location
    #Xepr.XeprCmds.vpSave('Current', 'Primary', title1, filename1)
    
    #dset2D2.X=dset1D2.X
    #dset2D2.Y=Garray
    #dset2D2.update(refresh=True)
    #dset2D2.setXeprSet('secondary')
    
    #title2=e107.get()
    #filename2=fold+'/'+ title2  #location
    #Xepr.XeprCmds.vpSave('Current', 'Secondary', title2, filename2)
        
    #Xepr.XeprCmds.prDiff('Current','Primary','All','Current',1,0,1)
    ###$$$    
    '''

    title3=e108.get()
    filename3=fold+'/'+ title3  #location
    #Xepr.XeprCmds.vpSave('Current', 'Result', title3, filename3) ###$$$
    #time.sleep(1)
    #toc4 = time.time()  ###$$$
    #print (toc4-tic4)   ###$$$
    

    #specr=spec1r-spec2r
    #speci=spec1i-spec2i  
    #spec=specr+speci*1j

    #np.savetxt(path+"/ys189501r.csv",spec1r,delimiter=",",fmt="%d")
    #np.savetxt(path+"/ys189502r.csv",spec2r,delimiter=",",fmt="%d")
    np.savetxt(filename3+"r.csv",specr, delimiter=",",fmt="%d")
    #np.savetxt(path+"/ys189501i.csv",spec1i,delimiter=",",fmt="%d")
    #np.savetxt(path+"/ys189502i.csv",spec2i,delimiter=",",fmt="%d")
    np.savetxt(filename3+"i.csv",speci, delimiter=",",fmt="%d")
    #np.savetxt(path+"/ys189503.csv",spec, delimiter=",",fmt="%d")
    
    toc4 = time.time()  ###$$$
    print (toc4-tic4)   ###$$$
    toc = time.time()
    print (toc-tic)



#############################
## Experiment Settings Tab ##
#############################

nb=ttk.Notebook(root)
nb.grid(row=0, column=0, columnspan=50, rowspan=49, sticky='NESW')

frame1 = ttk.Frame(nb)   #experiment settings
frame2 = ttk.Frame(nb)
frame3 = ttk.Frame(nb)
frame4 = ttk.Frame(nb)
nb.add(frame1, text='  Experiment Settings  ')
nb.add(frame2, text='  Field Step Settings  ')
nb.add(frame3, text='  Z Gradient Settings  ')
nb.add(frame4, text='  Spectrometer Settings  ')

f = open(path+'parameters.txt', "r")
temp = f.read().splitlines()
i=0
#a=temp[0]
#print a


### frame1
eL101 = Label(frame1, text="Main Field (G)") #, font=("Helvitca", 16))
eL101.grid(row=1, column=1, sticky=E)
e101 = Entry(frame1)
e101.insert(0, temp[i])  #1: 91
i+=1
e101.grid(row=1, column=2,padx = 5,pady = 5)

eL102 = Label(frame1, text="Rapid Scan Width (G)")
eL102.grid(row=2, column=1, sticky=E)
e102 = Entry(frame1)
e102.insert(0, temp[i]) #2: 70
i+=1
e102.grid(row=2, column=2,padx = 5,pady = 5)

eL103 = Label(frame1, text="Cycles per Trigger")
eL103.grid(row=3, column=1, sticky=E)
e103=Spinbox(frame1, from_=1, to=10, increment=1)
e103.delete(0,"end")
e103.insert(0,temp[i])   #3: 4
i+=1
e103.grid(row=3, column=2,padx = 5,pady = 5)

eL104 = Label(frame1, text="Scan Frequency (Hz)")
eL104.grid(row=4, column=1, sticky=E)
e104 = Entry(frame1)
e104.insert(0, temp[i])  #4: 5070
i+=1
e104.grid(row=4, column=2,padx = 5,pady = 5)

eL105 = Label(frame1, text="Gradient (G/cm) ")
eL105.grid(row=5, column=1, sticky=E)
e105 = Entry(frame1)
e105.insert(0, temp[i])  #5: 0
i+=1
e105.grid(row=5, column=2,padx = 5,pady = 5)

eL106 = Label(frame1, text="Scan 1 name: ")
eL106.grid(row=1, column=3, sticky=E)
e106 = Entry(frame1)
e106.insert(0, temp[i])   #6: ys1
i+=1
e106.grid(row=1, column=4,padx = 5,pady = 5)

eL107 = Label(frame1, text="Scan 2 name: ")
eL107.grid(row=2, column=3, sticky=E)
e107 = Entry(frame1)
e107.insert(0, temp[i])  #7: ys2
i+=1
e107.grid(row=2, column=4,padx = 5,pady = 5)

eL108 = Label(frame1, text="Subtracted result name: ")
eL108.grid(row=3, column=3, sticky=E)
e108 = Entry(frame1)
e108.insert(0, temp[i])   #8: ys3
i+=1
e108.grid(row=3, column=4,padx = 5,pady = 5)


l11 = Label(frame1, text="Fold to store files: ")
l11.grid(row=4, column=3, sticky=E)

with open(path+'address.txt', 'r') as f:
    name=f.read()
l12 = Label(frame1, text = "Current fold is: ")
l12.grid(row=5, column=3, sticky=E)
l13 = Label(frame1, text = name)
l13.grid(row=6, column=3, columnspan=2)

def foldin():
    fold = fd.askdirectory()
    with open (path+'address.txt','w') as f:
        f.write(fold)

    with open (path+'address.txt', 'r') as f:
        name = f.read()
    l13['text'] = name

b11 = Button(frame1, text ="  Change  ", command=foldin)
b11.grid(row=4,column=4)

l14 = Label(frame1, text = " ")
l14.grid(row=7, column=1, columnspan=2, sticky=W)

### frame2
eL201 = Label(frame2, text="Field Step Scan Width (G)")
eL201.grid(row=2, column=1, sticky=E)
e201 = Entry(frame2)
e201.insert(0, temp[i])   #9: 40
i+=1
e201.grid(row=2, column=2,padx = 5,pady = 5)

eL202 = Label(frame2, text="Field Step Size (G)")
eL202.grid(row=3, column=1, sticky=E)
e202 = Entry(frame2)
e202.insert(0, temp[i])   #10: 5
i+=1
e202.grid(row=3, column=2,padx = 5,pady = 5)

eL203 = Label(frame2, text="  ")
eL203.grid(row=5, column=1, sticky=E)

### Use Field Step Toggle
def FStog():
    if FS.get()==1:
	for child in frame3.winfo_children():
	    child.configure(state='disabled')
    else:
	for child in frame3.winfo_children():
	    child.configure(state='normal')
        avgoff()

FS=IntVar()
FSDDCheck=Checkbutton(frame2, text='Field Step', variable=FS, onvalue=1, offvalue=0, command=FStog)
FSDDCheck.grid(row=6, column=1, sticky=E)


### frame 3
eL301 = Label(frame3, text="Maximum Gradient (G/cm)")
eL301.grid(row=2, column=1, sticky=E)
e301 = Entry(frame3)
e301.insert(0, temp[i])   #11: 10
i+=1
e301.grid(row=2, column=2,padx = 5,pady = 5)

eL302 = Label(frame3, text="Gradient Step Size (G/cm)")
eL302.grid(row=3, column=1, sticky=E)
#e302 = Entry(frame3)
e302=Spinbox(frame3, from_=0.1, to=10, increment=0.01)
e302.delete(0,"end")
e302.insert(0, temp[i])   #12: 0.2
i+=1
e302.grid(row=3, column=2,padx = 5,pady = 5)

l301 = Label(frame3, text="                  ")
l301.grid(row=4, column=1, sticky=E)
l302 = Label(frame3, text="           ")
l302.grid(row=8, column=1, sticky=E)

# Use Z Gradient Toggle
def Ztog():
    if ZG.get()==1:
        e105.configure(state='disabled')
	for child in frame2.winfo_children():
	    child.configure(state='disabled')
    else:
        e105.configure(state='enabled')
	for child in frame2.winfo_children():
	    child.configure(state='enabled')
ZG=IntVar()
ZGradCheck=Checkbutton(frame3, text='Z Gradient', variable=ZG, onvalue=1, offvalue=0, command=Ztog)
ZGradCheck.grid(row=5, column=1, sticky=E)


### Use Average Toggle
def avgon():
    l303.configure(state='normal')
    l304.configure(state='normal')
    l305.configure(state='normal')
    l306.configure(state='normal')
    l307.configure(state='normal')
    l308.configure(state='normal')
    eL303.configure(state='normal')
    e303.configure(state='normal')
    eL304.configure(state='normal')
    e304.configure(state='normal')
    eL305.configure(state='normal')
    e305.configure(state='normal')
    eL306.configure(state='normal')
    e306.configure(state='normal')
    eL307.configure(state='normal')
    e307.configure(state='normal')


def avgoff():
    l303.configure(state='disabled')
    l304.configure(state='disabled')
    l305.configure(state='disabled')
    l306.configure(state='disabled')
    l307.configure(state='disabled')
    l308.configure(state='disabled')
    eL303.configure(state='disabled')
    e303.configure(state='disabled')
    eL304.configure(state='disabled')
    e304.configure(state='disabled')
    eL305.configure(state='disabled')
    e305.configure(state='disabled')
    eL306.configure(state='disabled')
    e306.configure(state='disabled')
    eL307.configure(state='disabled')
    e307.configure(state='disabled')


def avgtog():
    if Avg.get()==1:
	l305['text']= e301.get()
        avgon()
    else:
        avgoff()

Avg=IntVar()
AvgCheck=Checkbutton(frame3, text='Variable Averages', variable=Avg, onvalue=1, offvalue=0, command=avgtog)
AvgCheck.grid(row=2, column=4, columnspan=4,sticky=E)

w = Canvas(frame3, width=300, height=150)
w.create_rectangle(10, 10, 300, 150, outline = 'white')
w.grid(row=3, column=3,columnspan=10, rowspan=5)


l303 = Label(frame3, text="     Gradient Range (absolute values, G/cm) ")
l303.grid(row=3, column=4, columnspan=7)

l304 = Label(frame3, text=" |---------------------|-------------------|------------------| ")
l304.grid(row=4, column=4, columnspan=7)

l305 = Label(frame3, text=" ", width=4)  #same as Gmax: 10
l305.grid(row=5, column=4,)

eL303 = Label(frame3, text="   ", width=5)
eL303.grid(row=5, column=5)
e303 = Entry(frame3, width=2)
e303.insert(0, temp[i])  #13: 6
i+=1
e303.grid(row=5, column=6)

eL304 = Label(frame3, text="   ", width=5)
eL304.grid(row=5, column=7)
e304 = Entry(frame3, width=2)
e304.insert(0, temp[i])  #14: 2
i+=1
e304.grid(row=5, column=8)

l306 = Label(frame3, text="   ", width=5)
l306.grid(row=5, column=9)
l307 = Label(frame3, text="0", width=2)
l307.grid(row=5, column=10)

eL305 = Label(frame3, text="Avg.: [", width=5)
eL305.grid(row=7, column=4, sticky=E)
e305 = Entry(frame3, width=5)
e305.insert(0, temp[i])  #15: 2048
i+=1
e305.grid(row=7, column=5)

eL306 = Label(frame3, text=")[", width=2)
eL306.grid(row=7, column=6)
e306 = Entry(frame3, width=5)
e306.insert(0, temp[i])  #16: 1024
i+=1
e306.grid(row=7, column=7)

eL307 = Label(frame3, text=")[", width=2)
eL307.grid(row=7, column=8)
e307 = Entry(frame3, width=5)
e307.insert(0, temp[i])  #17: 512
i+=1
e307.grid(row=7, column=9)
l308 = Label(frame3, text="]", width=2)
l308.grid(row=7, column=10)

avgoff()


### frame 4
eL401 = Label(frame4, text="Main Magnet Coil Constant (G/A)")
eL401.grid(row=1, column=1, sticky=E)
e401 = Entry(frame4)
e401.insert(0, temp[i])   #18: 4.74
i+=1
e401.grid(row=1, column=2,padx = 5,pady = 5)

eL402 = Label(frame4, text="Rapid Scan Coil Constant (G/A)")
eL402.grid(row=2, column=1, sticky=E)
e402 = Entry(frame4)
e402.insert(0, temp[i])  #19: 27.7
i+=1
e402.grid(row=2, column=2,padx = 5,pady = 5)

eL403 = Label(frame4, text="Z Gradient Coil Constant (G/A)")
eL403.grid(row=3, column=1, sticky=E)
e403 = Entry(frame4)
e403.insert(0, temp[i])    #20: 0.5
i+=1
e403.grid(row=3, column=2,padx = 5,pady = 5)


l401 = Label(frame4, text="  ")
l401.grid(row=4, column=1, sticky=E)
l402 = Label(frame4, text="  Check Serial Port (Rapid Scan Coil Driver)")
l402.grid(row=5, column=1, sticky=E)
l403 = Label(frame4, text="  Check...")
l403.grid(row=7, column=1, sticky=W)

def checkRSCD():
    a=checkcd()
    l403['text']=a

b401=Button(frame4, text='Check', command=checkRSCD)
b401.grid(row=6, column=1)


def CheckNRun():
    if os.path.isfile(path+'parameters.txt'):
        os.remove(path+'parameters.txt')
    with open (path+'parameters.txt','a') as f:
        f.write(e101.get()+'\n')
        f.write(e102.get()+'\n')
        f.write(e103.get()+'\n')
        f.write(e104.get()+'\n')
        f.write(e105.get()+'\n')
        f.write(e106.get()+'\n')
        f.write(e107.get()+'\n')
        f.write(e108.get()+'\n')
        f.write(e201.get()+'\n')
        f.write(e202.get()+'\n')
        f.write(e301.get()+'\n')
        f.write(e302.get()+'\n')
        f.write(e303.get()+'\n')
        f.write(e304.get()+'\n')
        f.write(e305.get()+'\n')
        f.write(e306.get()+'\n')
        f.write(e307.get()+'\n')
        f.write(e401.get()+'\n')
        f.write(e402.get()+'\n')
        f.write(e403.get()+'\n')

    if FS.get()==1 and ZG.get()==0:
        runRSFSDD()
    elif FS.get()==0 and ZG.get()==1:
	    runRSgrad()
#    elif FS.get()==1 and ZG.get()==1:
#	    tkMessageBox.showinfo('Warning','Cannot run field step and gradient step at the same time. \n Please select only one check-box.')
    else:
        runRS()


b1=Button(root, text='Send', command=CheckNRun)
b1.grid(row=50, column=1)

l1 = Label(root, text="  ")
l1.grid(row=50, column=2)

# 0 Buttons
def zero1():
    MainMag(0, 1)
b2 = Button(root, text =" Set Main Magnet to 0 G ", command=zero1)
b2.grid(row=50,column=3)

l2 = Label(root, text="  ")
l2.grid(row=50, column=4)

def zero2():
    RSCD(1000, 0, 90, 1, 1)
b3 = Button(root, text =" Set RSCD to 0 A ", command=zero2)
b3.grid(row=50,column=5)

l3 = Label(root, text="  ")
l3.grid(row=50, column=6)

def zero3():
    e105.delete(0,"end")
    e105.insert(0, 0)
    ZGrad(0, 1)
b4=Button(root, text='Set Z Gradient to 0 G/cm', command=zero3)
b4.grid(row=50, column=7, sticky=E)

l4 = Label(root, text="     ")
l4.grid(row=50, column=8)

b5=Button(root, text='Close', command=root.quit)
b5.grid(row=50, column=9)

l5 = Label(root, text="  ", font=("Helvitca", 8))
l5.grid(row=51, column=1)
	
root.mainloop()
