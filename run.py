"""
Author: Bokyung Kim (bk174@duke.edu)
Affiliation: Duke University
Project: NSF_ASCENT (Umass/UNC/Duke)
Description: runs the integrated system, which is described in ASCENT.py
"""

from ASCENT import ASCENT
import configparser as cp
import os
import argparse

parser = argparse.ArgumentParser(description="NSF_ASCENT: 3D_conv, reservoir, and trainable 1T1R system")
parser.add_argument("--config", type=str, default="default.cfg", help="Simulation configuration file, loaded from sim_cfgs/")
args = parser.parse_args()

def main():
    config = cp.ConfigParser()
    cwd = os.getcwd() # obtain currently working directory
    config_path = os.path.join(cwd, "sim_cfgs", args.config)
    config.read(config_path)

    sys = ASCENT(config_path)

    model_cfg = config.get("simulation", "model_cfg")
    model = os.path.join(cwd, "model_cfgs/CIFAR10", model_cfg)

    layer_name, kernel_width, kernel_height, in_channels, out_channels, output_width, output_height, stride = sys.read_config(model)

    for layer_idx in range(len(layer_name)):
        print()
        print("Processing layer: {}".format(layer_name[layer_idx]))

        # configure the system with current layer dimensions
        sys.load_layer(kernel_width[layer_idx], kernel_height[layer_idx], in_channels[layer_idx], out_channels[layer_idx], output_width[layer_idx], output_height[layer_idx], stride[layer_idx])

        # update and apply FSM state until 'done' signal is reached

        sys.update_state(True)
        cycle = 0
        while not sys.done:
            sys.apply_latch()
            sys.update_state()

    print()
    sys.summary()

if __name__ == "__main__":
    main()

