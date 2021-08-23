## MAIN MAGNET POWER SUPPLY MODULE ##
# Script for interfacing with the main magnet supply #
import numpy as NP
import math 
import time
# import platform
# import binascii
# import serial
# import socket
from tkinter.filedialog import askdirectory
from tkinter import *
from tkinter import ttk

if (0):

    w=Tk()
    w.title("Soil file converter: Job Done! ")
    w.geometry("500x250+200+250")

    with open('temp/address.txt', 'r') as f:
        name = f.read()
    # print(name)

    lb11 = Label(w, text = "Current fold to store files is: \n"+name)
    lb11.grid(row=1, column=1)

    def foldin():
        fold = askdirectory()
        with open ('temp/address.txt','w') as f:
            f.write(fold)

        with open('temp/address.txt', 'r') as f:
            name = f.read()
        lb11['text'] = "Current fold to store files is: \n"+ name

    def ff():

        def f(a):
            b=1+a
            print(b)
        f(1)

    b3 = Button(w, text="Choose folder", fg="black", command=ff)
    b3.grid(row=2)

    w.mainloop()

if (0):
    # filename1 = 'C:/Users/Yilin/Documents/EPR/2018.8.24VHFSerial/a1'
    fold=askdirectory()
    name='a1'
    filename1=fold+'/'+name

    x = len(filename1)
    x = range(x)
    for character in x:
        n = character + 1
        n = -n
        if filename1[n] == '/':
            break

    title1 = filename1[n + 1:]
    print(filename1)
    print(title1)

# from pandas import DataFrame
# import pandas as pd
# df = pd.read_csv('log.csv')
# # df.set_index("entry", inplace=True)
# # df.head()
# # a=df.loc['e101']
# a=df.loc[9,"value"]
# # b=int(a)+1
# print(a)
import os
if(1):
    if os.path.isfile('log.txt'):
        os.remove('log.txt')
    with open ('log.txt','a') as f:
        f.write(str(1)+'\n')
        f.write('2'+'\n')
        f.write('70'+'\n')
        f.write('5070'+'\n')
        f.write('ab2'+'\n')
        f.write('ab3'+'\n')
        f.write('ab1'+'\n')
        f.write('/home/xuser/xeprFiles/Data/Yilin'+'\n')
        f.write('2048'+'\n')
        f.write('1024'+'\n')
        f.write('25'+'\n')

    with open('log.txt', 'r') as f:
        a=f.read()
    print(a[0])