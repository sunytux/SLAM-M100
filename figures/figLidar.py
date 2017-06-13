#! /usr/bin/python3

# Copyright © 2017 Sami Mezhoud <sami.mezhoud@ulb.ac.be>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See the COPYING file for more details.

""" ...

Usage:
    figLidar.py <FIGURE_DIR>

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


def main():
    l1 = pd.read_pickle("/home/sami/work/memo/data/processed/lidar1m.pickle")
    l2 = pd.read_pickle("/home/sami/work/memo/data/processed/lidar2m.pickle")
    l3 = pd.read_pickle("/home/sami/work/memo/data/processed/lidar3m.pickle")
    l4 = pd.read_pickle("/home/sami/work/memo/data/processed/lidar4m.pickle")
    l5 = pd.read_pickle("/home/sami/work/memo/data/processed/lidar5m.pickle")

    plt.scatter(0, 0, color='black', facecolor='black', marker=(3, 0, -90), s=400)

    for i, d in enumerate([l1, l2, l3, l4, l5]):
        d = d[(d.angle > 90) & (d.angle < 270) & (d.distance > 0.2)]

        label = "{:d} m".format(i+1)
        color = PLOT_COLORS[i]
        plt.scatter(d.lidarX, d.lidarY, color=color,  label=label,
                    s=30, marker='.',)
        if len(d) > 0:
            s = "dist: {:d}m, data: {:3d}, variance: {:1.2f}m"
            print(s.format(i+1, len(d), max(d.lidarX) - min(d.lidarX)))

    # plt.title("LiDAR measurements")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis(PLOT_AXIS)
    # plt.legend()
    # plt.show()
    plt.savefig("{}/lidarRangeTest.png".format(FIGURE_DIR), bbox_inches='tight')
    plt.savefig("{}/lidarRangeTest.pdf".format(FIGURE_DIR), bbox_inches='tight')


if __name__ == '__main__':
    main()
