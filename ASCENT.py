"""
Author: Bokyung Kim (bk174@duke.edu)
Affiliation: Duke University
Project: NSF_ASCENT (Umass/UNC/Duke)
Description: integration of all modules, including 3D conv architecture and so on (later)
"""

from conv_3D import conv_3D
import math
import numpy as np
import configparser as cp
import csv

class ASCENT:
    def __init__(self, config_path):
        """
        - Compute cycle-accurate latency, energy (later)
        - Compute total area
        - Implement Finite State Machine for control flow
        """
        
        self.config = cp.ConfigParser()
        self.config.read(config_path)

        # Registers for FSM
        self.start = False
        self.state = 0
        self.layer = 0
        self.cells = 0
        self.area = 0
        self.cycle = 0 
        self.ops = 0

        # Default layer stats
        self.kernel_width = 3
        self.kernel_height = 3
        self.in_channels = 3
        self.out_channels = 256
        self.output_width = 256
        self.output_kernel = 256
        self.stride = 1

        # Instantiate conv_3D
        self.conv = conv_3D(config_path, self.kernel_width, self.kernel_height, self.in_channels, self.out_channels, self.output_width, self.output_kernel, self.stride)

        # Summary variables 
        self.total_layers = []
        self.total_cells = []
        self.total_area =  []
        self.total_cycle = []
        self.total_ops = []
        self.total_latency = []
        self.total_power = []

    def read_config(self, path):
        """ input filter/IFM/OFM dimensions"""
        layer_name = []
        kernel_width = []
        kernel_height = []
        in_channels = []
        out_channels = []
        output_width = []
        output_height = []
        stride = []

        f = open(path, "r")
        next(f) # skip header line
        for line in f:
            line = line.strip()
            line = line.split(',')
            # support only Conv-type layers. "WA" is 1x1 pointwise Conv
            if len(line) >= 7 and ("Conv" in line[0]):
                print(line)
                layer_name.append(line[0])
                k_h = int(line[3])
                k_w = int(line[4])
                kernel_height.append(int(line[3]))
                kernel_width.append(int(line[4]))
                 #kernel_size.append(kernel_height*kernel_width)
                # assuming padded inputs
                padding_height = (k_h // 2) * 2
                padding_width = (k_w // 2) * 2
                input_height = int(line[1])
                input_width = int(line[2])
                 #in_obj_size.append(input_height*input_width)
                s = int(line[7])
                output_height.append((input_height-padding_height)/s)
                output_width.append((input_width-padding_width)/s)
                 #out_obj_size.append(output_height*output_width)
                in_channels.append(int(line[5]))
                out_channels.append(int(line[6]))
                stride.append(s)
            else:
                print("Skipping: {}".format(line[0]))
        f.close()
        return layer_name, kernel_width, kernel_height, in_channels, out_channels, output_width, output_height, stride 
    
    def load_layer (self, kernel_width, kernel_height, in_channels, out_channels, output_width, output_height, stride):
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_width = output_width
        self.output_height = output_height
        self.stride = stride
        return 

    def update_state (self, start=False):
        """
        Finite State Machine (FSM)
        State:
        0 - wait
        1 - 3D_conv architecture
        (2 - reservoir computing (later))
        (3 - trainable 1T1R (later))
        2 - store results in memory: temporarily
        """
        self.start = start

        # 0
        if self.state == 0:
            if self.start:
                self.done = False
                self.state = 1
            else: 
                self.cycle = 0
        #1
        elif self.state == 1:
            self.state = 2
        elif self.state == 2:
            self.state = 0
        return 
    
    def apply_latch (self):
        # to return energy of the given cycle 

        # 0
        if self.state == 0:
            self.done = True
            return 
        elif self.state == 1:
            self.layer = max(self.conv.layers, self.layer)
            self.cells = self.conv.cells
            self.area = self.conv.area
            self.cycle = self.conv.cycles
            self.ops = self.conv.ops
            return 
        elif self.state == 2:
            self.compute_stats ()

    def compute_stats(self):
        self.total_layers.append(self.total_layers)
        self.total_cells.append(self.cells)
        self.total_area.append(self.area)
        self.total_cycle.append(self.cycle)
        self.total_ops.append(self.ops)

        cycle_latency = 10e-9 # need to confirm 
        self.total_latency.append(self.cycle * cycle_latency)
        #self.total_power.append(self.
        return

    def summary(self):
        print("----- Total Summary -----")
        print("Conv architecture layers {}".format(max(self.total_layers)))
        print("Conv required total cells {}".format(sum(self.total_cells)))
        print("Conv required total area {}".format(sum(self.total_area)))
        print("Conv required total cycles {}".format(sum(self.total_cycle)))
        print("Conv required total ops {}".format(sum(self.total_ops)))
        print("Conv required total latency {}".format(sum(self.total_latency)))
        return

    def main():
       
        if __name__ == "__main__":
            main()

