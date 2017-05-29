#! /usr/bin/python3

# (c) Copyright 2014-2016 HIPPEROS S.A.
# (c) Copyright 2010-2013 Universite Libre de Bruxelles
# (c) Copyright 2006-2013 MangoGem S.A.
#
# The license and distribution terms for this file may be
# found in the file LICENSE.txt in this distribution.

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
