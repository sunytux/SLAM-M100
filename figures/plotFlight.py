#! /usr/bin/python3

# Copyright © 2017 Sami Mezhoud <sami.mezhoud@ulb.ac.be>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See the COPYING file for more details.

""" This script plots multiple graph concerning the drone trajectory. Data are
    provided by a pickle file produced by the `parser.py` script.

Usage:
    plotFlight.py <PICKEL_FILE>

Arguments:
    <PICKEL_FILE>         Pickle file containing processed data.

Options:
    -h, --help
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from docopt import docopt
import pandas as pd


def main():
    data = pd.read_pickle(docopt(__doc__)['<PICKEL_FILE>'])

    plt.subplots_adjust(hspace=0.35)

    # Drone trajectory
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection='3d')
    #ax1.plot(data.droneX, data.droneY, data.droneZ)
    ax1.set_title("Drone trajectory")
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    ax1.axis('equal')

    # Trajectory view from above
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.plot(data.droneX, data.droneY)
    ax2.set_title("Trajectory view from above")
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.axis('equal')

    # Altitude
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.plot(data.time, data.droneZ)
    ax3.set_title("Altitude")
    ax3.set_xlabel('time [s]')
    ax3.set_ylabel('z [m]')

    plt.show()


if __name__ == '__main__':
    main()
