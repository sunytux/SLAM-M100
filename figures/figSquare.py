#! /usr/bin/python3

# Copyright © 2017 Sami Mezhoud <sami.mezhoud@ulb.ac.be>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See the COPYING file for more details.

""" ...

Usage:
    figSquare.py <FIGURE_DIR>

Arguments:
    <FIGURE_DIR>          Directory for saved figures.

Options:
    -h, --help
"""

import matplotlib.pyplot as plt
from docopt import docopt
import pandas as pd

FIGURE_DIR = docopt(__doc__)['<FIGURE_DIR>']
PLOT_COLORS = ['cornflowerblue', 'darkorange', 'forestgreen', 'tomato', 'gold']
PLOT_AXIS = [-0.5, 5.5, -2.2, 2.2]
PLOT_AXIS = ['equals']

FILE = '/home/sami/work/memo/data/processed/square02.pickle'


def main():
    d = pd.read_pickle(FILE)
    x = [0, 3, 3, 0, 0]
    y = [0, 0, 3, 3, 0]

    plt.plot(d.droneX, d.droneY, color=PLOT_COLORS[0],  label="Drone trajectory")
    plt.plot(x, y, '--', color=PLOT_COLORS[1], label="Reference")

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend()
    # plt.show()
    plt.savefig("{}/droneSquareTest.png".format(FIGURE_DIR), bbox_inches='tight')
    plt.savefig("{}/droneSquareTest.pdf".format(FIGURE_DIR), bbox_inches='tight')


if __name__ == '__main__':
    main()
