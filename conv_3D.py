"""
Author: Bokyung Kim (bk174@duke.edu)
Affiliation: Duke Univeristy
Project: NSF_ASCENT (Umass/UNC/Duke)
Decription: Estimation of 3D RRAM architecture for convolutional layers
"""

import configparser as cp
import numpy as np

class conv_3D:
    def __init__(self, config_path, k_w, k_h, in_channels, out_channels, o_w, o_h, stride):
        """
        k_w, o_w     - kernel, output width
        k_h, o_h     - kernel, output height
        k_num   - number of kernel channels 
        """
        self.config = cp.ConfigParser()
        self.config.read(config_path)

        self.k_w = k_w
        self.k_h = k_h
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.o_w = o_w
        self.o_h = o_h
        self.stride = stride
        
        # memristor size (provided by Prof.Xia)
        self.cell_area = 300*1e-9

        # the number of cells, projected area and cycles
        if int(self.config.get("conv_3D", "method")): # Duke
            self.layers = self.k_w
            self.cells = self.k_w*self.k_h*(self.k_w+(self.o_w-1)*self.stride)*self.in_channels*self.out_channels
            self.area = self.cell_area * self.k_h * (self.k_w+(self.o_w-1)*self.stride) * self.in_channels * self.out_channels
            self.cycles = self.o_h
            self.ops = self.k_w * self.k_h * self.o_w * self.o_h * self.in_channels * self.out_channels * 2
        else: # Umass
            self.layers = self.k_w
            self.cells = self.k_w * (self.k_h+(self.o_h-1)*stride) * (self.k_w+(self.o_w-1)*stride) * self.in_channels * self.out_channels
            self.area = (self.k_h+(self.o_h-1)*stride) * (self.k_w+(self.o_w-1)*stride) * self.in_channels * self.out_channels
            self.cycles = 1
            self.ops = self.k_w * self.k_h * self.o_w * self.o_h * self.in_channels * self.out_channels * 2

    def update_state (self):
        """
        Nothing to do here when updating state
        """
        return

    def apply_latch (self):
        """
        return the calculated estimation
        """
        return self.layers, self.cells, self.area, self.cycles, self.ops

