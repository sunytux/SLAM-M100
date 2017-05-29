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
from docopt import docopt
import pandas as pd
import numpy as np
import slam
import parser
import math

FILE = docopt(__doc__)['<PICKEL_FILE>']


def main():

    # data = pd.read_pickle(FILE)
    data = pd.read_pickle("/home/sami/work/memo/data/processed/manSlam02_clean.pickle")
    data2 = pd.read_pickle("/home/sami/work/memo/data/processed/manSlam03_clean.pickle")


    data['color'] = np.where(data['time'] >= 45, 'green', 'cornflowerblue')


    data2['color'] = np.where(data2['time'] >= 27, 'yellow', 'cornflowerblue')

    data2['color'] = np.where(data2['time'] >= 42, 'green', data2['color'])

    plt.scatter(data.lidarX, data.lidarY, c=data.color, s=1, marker='.', label="LiDAR data")

    plt.scatter(data2.lidarX, data2.lidarY, c=data2.color, s=1, marker='.', label="LiDAR data")

    #plt.plot(data.droneX, data.droneY, c='dimgray',
    #         linewidth=1, label="Trajectory")

    plt.title("LiDAR measurements")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend()
    plt.show()


def main2():

    # RanSaC parameters
    D_RANSAC = 20  # Degree range
    S_RANSAC = 8   # Nb of samples
    X_RANSAC = 0.3  # 0.4  # Distance to wall in m
    N_RANSAC = 40  # Max trials
    # TODO consensus low
    C_RANSAC = 0.5  # Consensus = C *len(data)
    C_RANSAC_MIN = 30  # Minimum consensus

    PLOT_AXIS = [-12, 12, -12, 12]

    data = pd.read_pickle("/home/sami/work/memo/data/processed/manSlam03_clean.pickle")

    data = data[(data.time > 33.0) & (data.time < 34.5)]
    print(len(data))
    # TODO define better consensus low and clean code
    if C_RANSAC*len(data) >= C_RANSAC_MIN:
        Consensus = len(data) * C_RANSAC
    else:
        Consensus = C_RANSAC_MIN

    plt.scatter(data.lidarX, data.lidarY, color='gainsboro', s=2)
    localLandmarks, trial = slam.RanSaC(data.copy(), D_RANSAC, S_RANSAC,
                                   X_RANSAC, Consensus, N_RANSAC, plt=plt)

    """
    for m in localLandmarks:
        p1, p2 = m
        p = np.poly1d([p1, p2])
        xp = np.linspace(-7, 2)

        plt.plot(xp, p(xp), color='red')

        x,y = slam.landmarkCharacterisation(*m)
        plt.scatter(x, y, color='red', marker='o', s=100)
    """
    plt.axis(PLOT_AXIS)
    plt.show()


def main3():
    FILE = "/home/sami/work/memo/data/raw/manSlam01.CSV"


    PLOT_AXIS = [-12, 12, -12, 12]

    for i in range(0, 90, 15):
        pickleFile = "/tmp/memo/data/{:03d}.pickle".format(i)
        offset = i * math.pi/180
        
        data = parser.getData(FILE, offset)

        data = parser.cleanManSlam3(data)

        print('Saving to {}'.format(pickleFile))
        data.to_pickle(pickleFile)
        
        data['color'] = np.where(data['time'] >= 45, 'tomato', 'cornflowerblue')
        
        plt.clf()
        plt.scatter(data.lidarX, data.lidarY, c=data.color,
                    s=1, marker='.', label="LiDAR data")

        plt.plot(data.droneX, data.droneY, c='dimgray',
                 linewidth=1, label="Trajectory")

        plt.title("LiDAR measurements")
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis(PLOT_AXIS)
        plt.legend()
        print("Saving plot.")
        plt.savefig("/tmp/memo/data/{:03d}.png".format(i), bbox_inches='tight')


if __name__ == '__main__':
    main3()
