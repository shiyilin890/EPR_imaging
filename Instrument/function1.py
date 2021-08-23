## MAIN MAGNET POWER SUPPLY MODULE ##
# Script for interfacing with the main magnet supply #
import numpy as NP
import math 
import time
import platform
import binascii
import serial
import socket

def MainMag(field,cc):   #center field, coil constant
	# Open and Configure Socket
	TCP_IP = '192.168.1.4' 
	TCP_PORT = 10001
	BUFFER_SIZE = 1024
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((TCP_IP, TCP_PORT))
	
	# Calculate Current
	Current = str(field/cc)
	
	# Send Parameters
	#s.send('MON\r') # Send command to turn on
	#data = s.recv(BUFFER_SIZE)
	s.send('UPMODE:NORMAL\r')
	data = s.recv(BUFFER_SIZE)
	s.send('TRIG:OFF\r') # Set Trigger Mode to off activation
	data = s.recv(BUFFER_SIZE)
	s.send('MWIR:'+str(Current)+'\r') # Send current setting
	data = s.recv(BUFFER_SIZE)
	s.close() # close socket
	
def MainMagWFCntrl(Field_Command):
	# Open and Configure Socket
	TCP_IP = '192.168.1.4' 
	TCP_PORT = 10001
	BUFFER_SIZE = 1024
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((TCP_IP, TCP_PORT))
	
	# Send Parameters
	#s.send('MON\r') # Command to turn power supply ON
	#data = s.recv(BUFFER_SIZE)
	s.send('UPMODE:WAVEFORM\r')
	data = s.recv(BUFFER_SIZE)
	s.send('TRIG:NEG\r') # Set Trigger Mode to Negative Activation
	data = s.recv(BUFFER_SIZE)
	s.send('WAVE:N_PERIODS:1\r') # Send number of repeat Waveforms
	data = s.recv(BUFFER_SIZE)
	s.send(Field_Command+'\r') # Send Waveform Points
	data = s.recv(BUFFER_SIZE)
	s.send('WAVE:TRIGGER:POINT\r') # Set to point-by-point trigger mode
	data = s.recv(BUFFER_SIZE)
	s.send('WAVE:KEEP_START\r') # Activate WF in Trigger mode
	data = s.recv(BUFFER_SIZE)
	s.close() # close socket

def ZGrad(grad, cc):  #gradient, coil constant
    # Open and Configure Socket
	TCP_IP = '192.168.1.5'
	TCP_PORT = 10001
	BUFFER_SIZE = 1024
	g = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	g.connect((TCP_IP, TCP_PORT))
	I = str(grad / cc)   #current

	# Send Parameters
	MESSAGE1 = ('MON\r')  # Command to turn power supply ON
	MESSAGE2 = ('MWIR:' + str(I) + '\r')  # Defines current value to be sent
	g.send(MESSAGE1)  # Send command to turn on
	data = g.recv(BUFFER_SIZE)
	g.send(MESSAGE2)  # Send current setting
	data = g.recv(BUFFER_SIZE)
	g.close()  # close socket


def RSCD(sf, sw, trigger, cpt, cc):
	#scan frequency, sweep width, trigger, cycles per trigger, coil constant
	# Open and configure serial port
	rscd = serial.Serial()
	rscd.baudrate = 9600
	rscd.port = '/dev/ttyS2'
	rscd.bytesize = serial.EIGHTBITS
	rscd.parity = serial.PARITY_NONE
	rscd.stopbits = serial.STOPBITS_TWO
	rscd.open()
	run = 1

	swApp = 6  # Maximum Amps peak to peak
	curr = float(sw) / float(cc)
	sw_level = 4096 * ((curr / swApp))  # 12 bits used to define sweep width (current)
	upper_sweepwidth = int(sw_level / 256)
	lower_sweepwidth = int(sw_level - (upper_sweepwidth * 256))
	trigger_delay = int(256 * trigger / 360)  # 8 bits used to define trigger delay 0->360 degrees

	# Scan Freq Range - 3 bits are used (8 states) to define sf range. Loop to define 3 bits. See user manual ( Table 2, pg 17 ) Translated from Richard Quine's Code
	if sf == 10000:  # Will have to manually send byte 6 and 7 if frequency is external
		sf_range = 0
		byte6 = 0
		byte7 = 0

	if sf > 35000:
		div = 1
		sf_range = 2
	elif sf > 8000:
		div = 4
		sf_range = 3
	elif sf > 2000:
		div = 16
		sf_range = 4
	elif sf > 800:
		div = 40
		sf_range = 5
	else:
		div = 160
		sf_range = 6
	# Scan Frequency 16 bit over each range. Translated from Richard Quine's code

	DDSres = 2 ** 28
	DDSref = 61.44 * (10 ** 6)
	ADLSB = DDSref * 2 ** 10 / DDSres
	DDSoffset = ((2 ** (24)) / float(DDSres))
	DDSout = sf * 128 * div
	DDSzero = DDSref * DDSoffset
	DDSfreqadd = DDSout - DDSzero
	ADnum = DDSfreqadd / ADLSB
	ADnumU = ADnum / 256
	upper_scanfreq_fine = int(ADnumU)
	Z = 256 * upper_scanfreq_fine
	Z1 = ADnum - Z
	lower_scanfreq_fine = int(Z1)

	# Write data to coil driver

	byte1 = run
	byte2 = lower_sweepwidth
	byte3 = upper_sweepwidth
	byte4 = trigger_delay
	byte5 = cpt
	byte6 = sf_range
	byte7 = lower_scanfreq_fine
	byte8 = upper_scanfreq_fine

	rscd.write(chr(byte1))
	time.sleep(.01)
	rscd.write(chr(byte2))
	time.sleep(.01)
	rscd.write(chr(byte3))
	time.sleep(.01)
	rscd.write(chr(byte4))
	time.sleep(.01)
	rscd.write(chr(byte5))
	time.sleep(.01)
	rscd.write(chr(byte6))
	time.sleep(.01)
	rscd.write(chr(byte7))
	time.sleep(.01)
	rscd.write(chr(byte8))
	time.sleep(.01)

	rscd.close()
# rscd.is_open

def checkcd():
	# Open and configure serial port
	rscd=serial.Serial()
	rscd.baudrate=9600
	rscd.port = '/dev/ttyS2'
	rscd.bytesize=serial.EIGHTBITS
	rscd.parity=serial.PARITY_NONE
	rscd.stopbits=serial.STOPBITS_TWO
	rscd.open()

        a="Check..."
	if rscd.isOpen():
	    print "Serial Port is Open and Ready"
            a=  "  Serial Port is Open and Ready"
        
        return a



