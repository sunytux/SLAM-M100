#! /usr/bin/python3

# Copyright © 2017 Sami Mezhoud <sami.mezhoud@ulb.ac.be>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See the COPYING file for more details.

""" ...

Usage:
    figTrajectoryCorrection.py <FIGURE_DIR>

Arguments:
    <FIGURE_DIR>          Directory for saved figures.

Options:
    -h, --help
"""

import matplotlib.pyplot as plt
from docopt import docopt
import pandas as pd

FIGURE_DIR = docopt(__doc__)['<FIGURE_DIR>']
PLOT_RED = 'tomato'
PLOT_BLUE = 'cornflowerblue'
PLOT_GREEN = 'forestgreen'
PLOT_ORANGE = 'darkorange'
PLOT_YELLOW = 'gold'
PLOT_COLORS = [PLOT_BLUE, PLOT_ORANGE, PLOT_GREEN, PLOT_RED, PLOT_YELLOW]
PLOT_AXIS = [-12, 12, -12, 12]

DATA_DIR = FIGURE_DIR + '/data'

LAPS2 = [
        (0, 47),
        (44, 999),
]
LAPS3 = [
        (0, 27),
        (27, 42),
        (42, 57),
]

LAPS4 = [
        (0, 33),
        (33, 54),
        (54, 69),
]

LAPS = LAPS3


def main():

    data = pd.read_pickle(DATA_DIR + "/origData.pickle")
    correctedData = pd.read_pickle(DATA_DIR + "/correctedData.pickle")

    for i in range(len(LAPS)):
        mi, ma = LAPS[i]
        d = data[(data.time >= mi) & (data.time <= ma)]
        dc = correctedData[(data.time >= mi) & (data.time <= ma)]

        plt.clf()

        plt.plot(d.droneX, d.droneY, color=PLOT_BLUE,
                 label='Original Trajectory')
        plt.plot(dc.droneX, dc.droneY, color=PLOT_ORANGE,
                 label='Corrected Tajectory')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend()
        plt.axis(PLOT_AXIS)

        plt.savefig("{}/pdf/lap{}.pdf".format(FIGURE_DIR, i+1),
                    bbox_inches='tight')
        plt.savefig("{}/png/lap{}.png".format(FIGURE_DIR, i+1),
                    bbox_inches='tight')


if __name__ == '__main__':
    main()
