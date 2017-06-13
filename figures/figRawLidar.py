#! /usr/bin/python3

# Copyright © 2017 Sami Mezhoud <sami.mezhoud@ulb.ac.be>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See the COPYING file for more details.

""" ...

Usage:
    figRawLidar.py <FIGURE_DIR>

Arguments:
    <FIGURE_DIR>          Directory for saved figures.

Options:
    -h, --help
"""

import matplotlib.pyplot as plt
from docopt import docopt
import pandas as pd
import numpy as np

FIGURE_DIR = docopt(__doc__)['<FIGURE_DIR>']
PLOT_RED = 'tomato'
PLOT_BLUE = 'cornflowerblue'
PLOT_GREEN = 'forestgreen'
PLOT_ORANGE = 'darkorange'
PLOT_YELLOW = 'gold'
PLOT_COLORS = [PLOT_BLUE, PLOT_ORANGE, PLOT_GREEN, PLOT_RED, PLOT_YELLOW]
PLOT_AXIS = [-12, 12, -12, 12]

FILE = '/home/sami/work/memo/data/processed/manSlam03_clean.pickle'


def main():
    data = pd.read_pickle(FILE)

    # Different plot
    data['color'] = np.where(data['time'] >= 27, 'forestgreen', 'cornflowerblue')
    data["color"][(data["time"] >= 42)] = "tomato"

    plt.scatter(data.lidarX, data.lidarY, c=data.color,
                s=1, marker='.', label="LiDAR data")

    plt.plot(data.droneX, data.droneY, c='dimgray',
             linewidth=1, label="Trajectory")

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis(PLOT_AXIS)
    # plt.legend()
    # plt.show()
    plt.savefig("{}/res_rawLidar.png".format(FIGURE_DIR), bbox_inches='tight')
    plt.savefig("{}/res_rawLidar.pdf".format(FIGURE_DIR), bbox_inches='tight')


if __name__ == '__main__':
    main()
