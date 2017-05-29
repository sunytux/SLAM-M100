#! /usr/bin/python3

# (c) Copyright 2014-2016 HIPPEROS S.A.
# (c) Copyright 2010-2013 Universite Libre de Bruxelles
# (c) Copyright 2006-2013 MangoGem S.A.
#
# The license and distribution terms for this file may be
# found in the file LICENSE.txt in this distribution.

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
