"""
Author: Bokyung Kim
Affiliation: Duke University
Project: NSF_ASCENT (Umass/UNC/Duke)
Description: Estimation of long short-term memory (LSTM) or reservoir computing (RC) layer
"""

import configparser as cp
import numpy as np

class lstm:
	def __init__ (self, m1, in1):
		"""
		m1	- the number of hidden neurons
		in1	- the number of inputs
		"""

		self.config = cp.ConfigParser()
		self.config.read(config_path)

		self.num_h = m1
		self.num_in = in1

		# memristor size (provided by Prof. Xia)
		self.cell_area = 300*1e-9

		# the nuber of cells, area, and cycles
		if int(self.config.get("Spatiotemporal data processing", "method"): # LSTM
			self.cells = (self.num_in + self.num_h) * 4 * self.num_h
			self.area = self.cell_area * self.cells
			self.cycles = 1
			self.ops = 1
		else: # RC
			self.cells = self.num_h
			self.area = self.cell_area * self.cells
			self.cycles = 1
			self.ops = 1

	def update_state (self):
		"""
		Nothing to do here when updating state
		"""
		return

	def apply_latch (self):
		"""
		return the calculated estimation
		"""
		return self.cells, self.area, self.cycles, self.ops
