import sys
from helper import Helper

if len(sys.argv) != 2:
	print('Usage: main.py [model: nb/lr/dt]')
	quit()

if sys.argv[1] == 'nb' or sys.argv[1] == 'lr' or sys.argv[1] == 'dt':
	model = sys.argv[1]
else:
	print('Please input valid ml model! nb/lr/dt')
	quit()

if model == 'nb':

elif model == 'lr':

elif model == 'dt':


