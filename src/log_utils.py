"""
Author: Jim Clauwaert
Created in the scope of my PhD
"""
import json
import numpy as np
import datetime as dt
import pandas as pd
import random

class logger_class(object):
	def __init__(self, args, timestamp, start):
		self.events = []
		self.args = args
		self.ts = timestamp
		self.start = start
		self.register_job()
	
	def log_event(self, event, verbose=True):
		self.events.append(event)
		if verbose:
			print("...CHECK\n{}".format(event), end="")
	
	def register_job(self):
		output = "\n\nSTARTED {}_{}\n\targuments: {}".format(self.args.name, self.ts, self.args)
		with open("../logs/_log.txt", 'a') as f:  
			f.write(output)
		f.close()
		self.log_event(output)
		
	def wrap_job(self):
		with open("../logs/_log.txt", 'a') as f:
			f.write('\n...FINISHED')
		f.close()
	
	def __str__(self):
		"""logger class"""
		
