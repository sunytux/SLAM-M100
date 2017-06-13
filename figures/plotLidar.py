#! /usr/bin/python3

# Copyright © 2017 Sami Mezhoud <sami.mezhoud@ulb.ac.be>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See the COPYING file for more details.

""" This script plots LiDAR measurements and drone trajectory both on the same
    graph. Data are provided by a pickle file produced by the `parser.py`
    script.

Usage:
    plotLidar.py <PICKEL_FILE>

Arguments:
    <PICKEL_FILE>         Pickle file containing processed data.

Options:
    -h, --help
"""

import matplotlib.pyplot as plt
from docopt import docopt
import numpy as np
import pandas as pd


def main():
    data = pd.read_pickle(docopt(__doc__)['<PICKEL_FILE>'])

    # Different plot
    data['color'] = np.where(data['time'] >= 45, 'tomato', 'cornflowerblue')

    plt.scatter(data.lidarX, data.lidarY, c=data.color,
                s=1, marker='.', label="LiDAR data")

    plt.plot(data.droneX, data.droneY, c='dimgray',
             linewidth=1, label="Trajectory")

    plt.title("LiDAR measurements")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
