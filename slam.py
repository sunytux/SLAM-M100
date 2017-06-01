#! /usr/bin/python3 -u

# Copyright © 2017 Sami Mezhoud <sami.mezhoud@ulb.ac.be>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See the COPYING file for more details.

""" ...

Usage:
    slam.py [-v|-vv] [-f|] <PICKEL_FILE>
    slam.py [-v|-vv] [-f|] <PICKEL_FILE> -p <FIGURE_DIR>

Arguments:
    <PICKEL_FILE>         Pickle file containing processed data.
    <FIGURE_DIR>          Directory for saved figures.

Options:
    -p <FIGURE_DIR>       Directory where to save figures.
    -f                    Force overwrite.
    -v                    Verbose mode.
    -vv                   Very Verbose mode.
    -h, --help
"""

from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import copy
import os
import shutil
import warnings
import pickle

# RanSaC parameters
D_RANSAC = 20  # Degree range
S_RANSAC = 8  # Nb of samples
X_RANSAC = 0.4   # Distance to wall in m
N_RANSAC = 40  # Max trials
C_RANSAC = 0.6  # Consensus = C *len(data)
C_RANSAC_MIN = 50  # 50 # Minimum consensus

# Landmark validation parameters
D_VALIDATION = 4  # Distance max between 2 observations of a landmarks in m
N_VALIDATION = 2  # Nb of observation of a landmark need before data adjustment

# EKF parameters
TIME_WINDOW = 3  # Time window size in s
TIME_STEP = 3  # Sliding window step in s
ODOMETRY_NOISE = 5  # noise on the odometry process
RANGE_LIDAR_NOISE = 0.1  # distances noise on the LiDAR
BEARING_LIDAR_NOISE = 1 * math.pi/180  # angle noise on the LiDAR

# Figure parameters
PLOT_RED = 'tomato'
PLOT_BLUE = 'cornflowerblue'
PLOT_GREEN = 'forestgreen'
PLOT_ORANGE = 'darkorange'
PLOT_YELLOW = 'gold'
PLOT_LIGHT_GREY = 'gainsboro'
PLOT_COLORS = [PLOT_RED, PLOT_ORANGE, PLOT_YELLOW, 'navy'] * 10
PLOT_AXIS = [-12, 12, -12, 12]
PLOT_DRONE_HISTORY = 10  # seconds

IS_VERBOSE_MODE = docopt(__doc__)['-v'] >= 1
IS_HARD_VERBOSE_MODE = docopt(__doc__)['-v'] >= 2
IS_PLOTTING_MODE = docopt(__doc__)['-p'] is not None
FILE = docopt(__doc__)['<PICKEL_FILE>']
IS_OVERWRITE_MODE = docopt(__doc__)['-f'] >= 1
if IS_PLOTTING_MODE:
    FIGURE_DIR = docopt(__doc__)['-p']

# Ignore the warnings related to polyfit
warnings.filterwarnings('ignore')


def main():

    # If in plotting mode, assure that the directory to save figure exist
    if IS_PLOTTING_MODE:
        if os.path.isdir(FIGURE_DIR):
            if not IS_OVERWRITE_MODE:
                i = input("Figure directory already exist. Erase ?[y/n] ")
            if IS_OVERWRITE_MODE or i == "y":
                shutil.rmtree(FIGURE_DIR)
            else:
                print("Exit")
                exit()

        os.mkdir(FIGURE_DIR)
        os.mkdir(os.path.join(FIGURE_DIR, "pdf"))
        os.mkdir(os.path.join(FIGURE_DIR, "png"))
        os.mkdir(os.path.join(FIGURE_DIR, "data"))

    data = pd.read_pickle(FILE)

    correctedData, X = SLAM(data)

    if IS_PLOTTING_MODE:
        s = FIGURE_DIR + "/data/{}"

        dataFile = open(s.format("origData.pickle"), "wb")
        correctedDataFile = open(s.format("correctedData.pickle"), "wb")

        pickle.dump(data, dataFile)
        pickle.dump(correctedData, correctedDataFile)


def SLAM(origData):
    """ Apply a SLAM algorithm on LiDAR data.

    It recognize walls as landmarks using the RANSAC algorithm. Then landmarks
    are associated with a nearest-neighbor approach. The SLAM problem is solved
    by an EKF procedure.

    origData is a DataFrame object from the pandas module corresponding to
    post-precessed data acquired by the drone.
    """

    correctedData = pd.DataFrame(columns=origData.columns)

    timeIdx = 0
    offset = np.array([0.0, 0.0, 0.0])

    # Initialization of SLAM relevant matrix
    X, P, R, Jxr, Jz = initSlamMatrix()
    while timeIdx < origData.iloc[-1].time - TIME_WINDOW:

        # Reducing the data to the current time window
        curData = origData.copy()
        curData = curData[(curData.time > timeIdx) &
                          (curData.time < timeIdx + TIME_WINDOW)]
        timeIdx += TIME_STEP

        # Correction
        curData = dataCorrection(curData, offset)

        # Saving data for figures
        if IS_PLOTTING_MODE:
            previousCurData = curData.copy()  # Saving for plot
            previousX = copy.deepcopy(X)

        # Consensus law
        if C_RANSAC*len(curData) >= C_RANSAC_MIN:
            Consensus = len(curData) * C_RANSAC
        else:
            Consensus = C_RANSAC_MIN

        # Detecting landmarks using the RanSaC algorithm
        localLandmarks, trial = RanSaC(curData.copy(), D_RANSAC, S_RANSAC,
                                       X_RANSAC, Consensus, N_RANSAC)

        # Distinguishing which landmarks are observed for the first time
        localLandmarks = [makeLandmark(m) for m in localLandmarks]
        newLandmarkds, reobservedLandmark = \
            lmAssociation(X, localLandmarks)

        # Updating drone position using EKF methodology
        # STEP 1: update state using odometry data
        if len(curData) > 0:
            P, X, Jxr, Jz = updateFromOdometry(curData, X, P, Jxr, Jz)

        # STEP 2: update state using re-observed landmarks
        if len(reobservedLandmark) > 0:
            X, P, offset = updateFromReobservedLandmarks(reobservedLandmark,
                                                         P, X, R)

            curData = dataCorrection(curData, offset)

        # STEP 3: add new landmarks to the state
        if len(newLandmarkds) > 0:
            P, X = addNewLandmarks(newLandmarkds, X, P, Jxr, Jz, R)

        for i in curData.index:
            correctedData.loc[i] = curData.loc[i]

        if IS_VERBOSE_MODE:
            s = "\nTime window {:2.1f}s-{:2.1f}s: {} Data."
            section = s.format(timeIdx, timeIdx+TIME_WINDOW, len(curData))

            if len(newLandmarkds) + len(reobservedLandmark) > 0:
                print(section)
                for m in newLandmarkds:
                    s = "\tNew wall index {:d}."
                    print(s.format(m['index']))
                for m in reobservedLandmark:
                    s = "\tWall {:d} re-observed {:d} times."
                    print(s.format(m['index'], X.lmOccurrence[m['index']]))
                if len(reobservedLandmark) > 0:
                    s = "\tSLAM offset:\n"
                    s += "\t\tx: {:.3f} m\n"
                    s += "\t\ty: {:.3f} m\n"
                    s += "\t\tyaw: {:.3f} rad\n"
                    print(s.format(*offset))

            elif IS_HARD_VERBOSE_MODE and trial == 0:
                print(section, "\nNot enough data.")

            elif IS_HARD_VERBOSE_MODE and trial == N_RANSAC:
                print(section, "\nNo consensus.")

        if IS_PLOTTING_MODE:

            if len(curData) > 0:
                plotStep1(correctedData, previousCurData, previousX,
                          timeIdx, localLandmarks)

            # Re-observation
            if len(reobservedLandmark) > 0:
                plotStep2(correctedData, curData, previousCurData, X,
                          previousX, timeIdx)
            # New landmarks
            #if len(newLandmarkds) > 0:
            if len(curData) > 0:
                plotStep3(correctedData, previousCurData, curData, X, timeIdx)

    if IS_VERBOSE_MODE:
        print("-"*40, "\nSummary:\nFound {} walls".format(X.lmLength))
        for m in range(X.lmLength):
            s = "\tWall {:d} observed {:d} times"
            print(s.format(m, X.lmOccurrence[m]))

        print("Distances:")
        for m in range(X.lmLength):
            for i in range(m+1, X.lmLength):
                s = "\tFrom wall {:d} to {:d}: {:5.2f} m."
                print(s.format(m, i, distPointToPoint(X.getLm(m), X.getLm(i))))

    if IS_PLOTTING_MODE:
        plt.clf()
        plt.scatter(origData.lidarX, origData.lidarY, color=PLOT_LIGHT_GREY,
                    marker='.', s=3, label='Raw data')
        plt.scatter(correctedData.lidarX, correctedData.lidarY, marker='.',
                    color=PLOT_BLUE, label='Corrected data', s=3)

        for idx in range(X.lmLength):
            plotLandmark(X, idx)

        if X.lmLength == 2:
            plotDistance(X.getLm(0), X.getLm(1))
        elif X.lmLength == 4:
            plotDistance(X.getLm(0), X.getLm(2), textRatio=0.15)
            plotDistance(X.getLm(1), X.getLm(3), textRatio=0.8)

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.legend()
        plt.axis(PLOT_AXIS)
        plt.savefig("{}/png/Final.png".format(FIGURE_DIR), bbox_inches='tight')
        plt.savefig("{}/pdf/Final.pdf".format(FIGURE_DIR), bbox_inches='tight')

    return correctedData, X


def plotDroneMarker(row, color='black'):
    angleOffset = row.yaw * 180/math.pi - 45

    plt.scatter(row.droneX, row.droneY, color=color, facecolor=color, s=100,
                marker=(3, 0, angleOffset), zorder=3)


def plotStep1(correctedData, previousCurData, previousX, time, localLandmarks):
    plt.clf()

    # Drone trajectory
    correctedData = \
        correctedData[time - correctedData.time < PLOT_DRONE_HISTORY]
    correctedData = correctedData.loc[:previousCurData.iloc[0].name][:-1]
    correctedData = pd.concat([correctedData, previousCurData])

    plt.plot(correctedData.droneX, correctedData.droneY,
             color=PLOT_LIGHT_GREY, zorder=1)
    plt.plot(previousCurData.droneX, previousCurData.droneY,
             color='gray', zorder=2)
    plotDroneMarker(previousCurData.iloc[-1])

    # Corrected points
    plt.scatter(previousCurData.lidarX, previousCurData.lidarY,
                color=PLOT_BLUE, s=1)

    # Landmarks
    for idx in range(previousX.lmLength):
        plotLandmark(previousX, idx)

    # Ransac landmarks
    for m in localLandmarks:
        plotLandmark(*m['param'], mkr='--', color='darkmagenta')

    # Figure properties
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis(PLOT_AXIS)
    s = "{}/png/{}-01.png"
    plt.savefig(s.format(FIGURE_DIR, time), bbox_inches='tight')
    s = "{}/pdf/{}-01.pdf"
    plt.savefig(s.format(FIGURE_DIR, time), bbox_inches='tight')


def plotStep2(correctedData, curData, previousCurData, X, previousX, time):
    plt.clf()

    # Drone trajectory
    history = \
        correctedData[time - correctedData.time < PLOT_DRONE_HISTORY]
    plt.plot(history.droneX, history.droneY, color=PLOT_LIGHT_GREY, zorder=1)
    plt.plot(previousCurData.droneX, previousCurData.droneY,
             color=PLOT_RED, zorder=2)
    plotDroneMarker(previousCurData.iloc[-1], color=PLOT_RED)

    plt.plot(curData.droneX, curData.droneY,
             color=PLOT_GREEN, zorder=2)
    plotDroneMarker(curData.iloc[-1], color=PLOT_GREEN)

    # Before and after point correction
    plt.scatter(previousCurData.lidarX, previousCurData.lidarY,
                color=PLOT_RED, s=2)
    plt.scatter(curData.lidarX, curData.lidarY, color=PLOT_GREEN, s=2)

    # Landmarks
    # Old
    for idx in range(previousX.lmLength):
        plotLandmark(previousX, idx, color=PLOT_RED)
    # Updated
    for idx in range(X.lmLength):
        plotLandmark(X, idx, color=PLOT_GREEN)

    # Figure properties
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis(PLOT_AXIS)
    s = "{}/png/{}-02.png"
    plt.savefig(s.format(FIGURE_DIR, time), bbox_inches='tight')
    s = "{}/pdf/{}-02.pdf"
    plt.savefig(s.format(FIGURE_DIR, time), bbox_inches='tight')


def plotStep3(correctedData, previousCurData, curData, X, time):
    plt.clf()

    # Drone trajectory
    correctedData = \
        correctedData[time - correctedData.time < PLOT_DRONE_HISTORY]
    correctedData = correctedData.loc[:previousCurData.iloc[0].name][:-1]
    correctedData = pd.concat([correctedData, previousCurData])

    plt.plot(correctedData.droneX, correctedData.droneY,
             color=PLOT_LIGHT_GREY, zorder=1)
    plt.plot(previousCurData.droneX, previousCurData.droneY,
             color='gray', zorder=2)
    plotDroneMarker(previousCurData.iloc[-1])

    # Corrected points
    plt.scatter(curData.lidarX, curData.lidarY, color=PLOT_BLUE, s=1)

    # Landmarks
    for idx in range(X.lmLength):
        plotLandmark(X, idx)

    # Figure properties
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis(PLOT_AXIS)
    s = "{}/png/{}-03.png"
    plt.savefig(s.format(FIGURE_DIR, time), bbox_inches='tight')
    s = "{}/pdf/{}-03.pdf"
    plt.savefig(s.format(FIGURE_DIR, time), bbox_inches='tight')


def plotLandmark(p1, p2, color=False, mkr='-', label=''):
    """ param = [p1, p2] or [X, idx]. """

    if type(p1).__name__ == StateMatrix.__name__:
        label = "Wall {:d}: {:d} times".format(p2, p1.lmOccurrence[p2])
        if not color:
            color = PLOT_COLORS[p2]
        coordLm = p1.getLm(p2)
        p1, p2 = lmPointToParam(*coordLm)
    else:
        if not color:
            color = 'purple'
        coordLm = lmParamToPoint(p1, p2)

    l1, l2 = getLineCircleIntersection([p1, p2], coordLm, 2)
    plt.plot(l1, l2, mkr, color=color, label=label)
    plt.scatter(l1, l2, color=color, marker='x', s=50)
    plt.scatter(*coordLm, color=color, marker='o', s=100)


def plotDistance(p1, p2, textRatio=0.5):

    d = distPointToPoint(p1, p2)
    text = "{:.2f} m".format(d)

    textPos = [(p1[0] - p2[0])*textRatio, (p1[1] - p2[1])*textRatio]
    textPos = [p1[0] - (p1[0] - p2[0])*textRatio - (p1[1] - p2[1])*0.08,
               p1[1] - (p1[1] - p2[1])*textRatio + (p1[0] - p2[0])*0.08]

    plt.annotate('', xy=p1, xytext=p2, arrowprops={'arrowstyle': '<->'})
    plt.annotate(text, xy=p1, xytext=textPos)


def getLineCircleIntersection(line, center, R):
    """ Return the intersection of a line and circle.

    line is a vector (a, c) such as the line equation is y = ax + c.
    center is the coordinate of the center of the circle.
    R is the radius of the circle.
    """

    a, c = line
    xl, yl = center

    A = a**2 + 1
    B = 2*a*(c - yl) - 2*xl
    C = xl**2 + (c - yl)**2 - R**2

    Delta = B**2 - 4*A*C

    x1 = (-B + math.sqrt(Delta)) / (2*A)
    x2 = (-B - math.sqrt(Delta)) / (2*A)
    y1 = a*x1 + c
    y2 = a*x2 + c

    return [x1, x2], [y1, y2]


def dataCorrection(data, offset):
    """ Add a position offset to relevant column of data.

    data is a DataFrame object from the pandas module.
    offset is list of offset such as [x_offset, y_offset, yaw_offset].
    """

    data['droneX'] = data['droneX'].map(lambda x: x-offset[0])
    data['droneY'] = data['droneY'].map(lambda x: x-offset[1])
    data['yaw'] = data['yaw'].map(lambda x: x-offset[2])
    data['lidarX'] = data['lidarX'].map(lambda x: x-offset[0])
    data['lidarY'] = data['lidarY'].map(lambda x: x-offset[1])

    return data


def initSlamMatrix():
    """ Return intial value of all SLAM relevant matrices.

    Return:
    X as a StateMatrix object.
    P as a CovarianceMatrix object.
    Jxr as the prediction model jacobian with respect to position.
    Jz as the prediction model jacobian with respect to landmarks observations.
    """

    # State Matrix
    X = StateMatrix()

    # Covariance matrix
    P = CovarianceMatrix()

    # Modified prediction model jacobian
    Jxr = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]])

    # Modified prediction model jacobian
    Jz = np.zeros([2, 2])

    # LiDAR Noise
    V = np.identity(2)
    R = V @ np.identity(2) * [RANGE_LIDAR_NOISE, BEARING_LIDAR_NOISE] @ V.T

    return X, P, R, Jxr, Jz


def updateFromOdometry(data, X, P, Jxr, Jz):
    """ 1st step: update state from odometry data.

    Return the modified X, P and prediction model jacobian  matrices.
    data is a DataFrame object from the pandas module.
    X is a StateMatrix object.
    P is a CovarianceMatrix object.
    Jxr is the prediction model jacobian with respect to position.
    Jz is the prediction model jacobian with respect to landmarks observations.
    """

    dx = data.iloc[0].droneX - data.iloc[-1].droneX
    dy = data.iloc[0].droneY - data.iloc[-1].droneY
    dt = data.iloc[0].yaw - data.iloc[-1].yaw

    # State matrix
    X.X[0] += dx
    X.X[1] += dy
    X.X[2] += dt

    # Prediction model jacobian
    A = np.identity(3)
    A[0, 2] = -dy
    A[1, 2] = dx

    # Process noise matrix
    W = np.identity(3) * [dx, dy, dt]
    Q = np.ones([3, 3]) * ODOMETRY_NOISE**2

    # Covariance on the robot position
    P.Prr = A @ P.Prr @ A.T + W @ Q @ W.T

    # Cross-correlations covariance
    for i in range(int((len(P.P)-3)/2)):
        Pri = P.getE(i)
        E = A @ Pri
        D = E.T

        P.setD(i, D)
        P.setE(i, E)

    # Modified prediction model jacobian with respect to position
    Jxr[0, 2] = -dy
    Jxr[1, 2] = dx

    # Modified prediction model jacobian with respect to landmarks observations
    Jz[0, 0] = math.cos(X.X[2] + dt)
    Jz[1, 0] = math.sin(X.X[2] + dt)
    Jz[0, 1] = -math.sin(X.X[2] + dt) * dt
    Jz[1, 1] = math.cos(X.X[2] + dt) * dt

    return P, X, Jxr, Jz


def updateFromReobservedLandmarks(reobservedLandmark, P, X, R):
    """ 2nd step: update state from re-observed landmarks.

    Return the modified X matrix and the offset to apply on measurments.
    reobservedLandmark is a list of re-observe landmarks define by the landmark
    association process.
    X is a StateMatrix object.
    P is a CovarianceMatrix object.
    R is the LiDAR noise.
    """

    for landmark in reobservedLandmark:

        # Landmark index
        idx = 3+landmark['index']*2

        # Compute the Jacobian of the measurement model
        r = distPointToLine(X.X[:2], landmark['point'])
        H11 = (X.X[0] - landmark['point'][0]) / r
        H12 = (X.X[1] - landmark['point'][1]) / r
        H21 = (landmark['point'][1] - X.X[1]) / (r**2)
        H22 = (landmark['point'][0] - X.X[0]) / (r**2)
        H = np.zeros([2, len(X.X)])
        H[:2, :3] = [[H11, H12, 0],
                     [H21, H22, -1]]
        H[:, idx:idx+2] = [[-H11, -H12],
                           [-H21, -H22]]

        # Compute innovation covariance
        d = distPointToPoint(X.X[0:2], landmark['point'])
        V = np.identity(2)
        R = np.identity(2) * [RANGE_LIDAR_NOISE*d, BEARING_LIDAR_NOISE]
        S = H @ P.P @ H.T + V @ R @ V.T

        # Compute Kalman gain
        K = P.P @ H.T @ np.linalg.inv(S)

        # Compute range and bearing of the observed landmark and its previous
        z = np.array(getRelativePos(X.X[:3], X.getLm(landmark['index'])))
        h = np.array(getRelativePos(X.X[:3], landmark['point']))

        # Update X state
        corr = K @ (z-h)
        X.X += corr

        # TODO make sure we don't need this
        # This instruction is not in 'SLAM for dummies' but according to EKF
        # algorithm it should be. Anyway it doesn't work.
        # I = np.identity(3)
        # P.Prr = (I - K[:3, :3] @ H[:3, :3]) @ P.Prr

        # TODO the offset is overwritten by last landmark
        offset = corr[:3]

    return X, P, offset


def addNewLandmarks(newLandmarkds, X, P, Jxr, Jz, R):
    """ 3rd step: Adding new landmarks to state and covariance matrices.

    Return the modified X and P matrices.
    newLandmarks is a list of new landmarks define by the landmark association
    process.
    X is a StateMatrix object.
    P is a CovarianceMatrix object.
    Jxr is the prediction model jacobian with respect to position.
    Jz is the prediction model jacobian with respect to landmarks observations.
    R is the LiDAR noise.
    """

    for m in newLandmarkds:
        # State matrix
        X.addLandmark(m['point'])

        P.addLandmark()

        # Landmark covariance
        C = Jxr @ P.Prr @ Jxr.T
        d = distPointToPoint(X.X[0:2], m['point'])
        R = np.identity(2) * [RANGE_LIDAR_NOISE*d, BEARING_LIDAR_NOISE]
        C += Jz @ R @ Jz.T

        P.setC(m['index'], C)

        # Robot-Landmark covariance
        E = P.Prr @ Jxr.T
        D = E.T

        P.setD(m['index'], D)
        P.setE(m['index'], E)

        # Landmark-Landmark covariance
        for i in range(m['index']):
            Pri = P.getD(i)
            F = Jxr @ Pri.T
            G = F.T

            P.setF(m['index'], i, F)
            P.setG(m['index'], i, G)

    return P, X


def getRelativePos(dr, lm):
    """ Return polar coordinate of a landmark relative to the drone.

        dr is a the pose of the drone such as dr = [x, y, yaw].
        lm is the cartesian coordinate of the landmark.
    """

    r = np.linalg.norm([lm[0]-dr[0], lm[1]-dr[1]])
    theta = math.atan2((lm[1]-dr[1]), (lm[0]-dr[0]))-dr[2]

    return [r, theta]


class StateMatrix(object):
    """ Represent a matrix containing the drone pose and landmarks position.

    Getter and setter functions are set up for easier access.
    """

    def __init__(self):
        self.X = np.array([0.0, 0.0, 0.0])
        self.lmOccurrence = []

    def addLandmark(self, point):
        """ Add new landmark."""

        self.X = np.concatenate((self.X, point), axis=0)
        self.lmOccurrence += [1]

    def getLm(self, i):
        """" Getter: landmark point (2x1). """

        index = 3+i*2

        return self.X[index:index+2]

    def setLm(self, i, value):
        """" Setter: landmark point (2x1). """

        index = 3+i*2

        self.X[index:index+2] = value

    @property
    def lmLength(self):
        """ Number of landmarks. """
        return len(self.lmOccurrence)


class CovarianceMatrix(object):
    """ Represent a matrix containing all relevant covariance matrix:
        - Covariance on the robot position
        - Landmark covariance
        - Robot - landmark covariance
        - Landmark - robot covariance
        - Landmark - landmark  covariance

    Getter and setter functions are set up for easier access.
    """

    def __init__(self):
        self.P = np.identity(3)

    @property
    def Prr(self):
        """ Covariance on the robot position. """
        return self.P[:3, :3]

    @Prr.setter
    def Prr(self, value):
        """Setter: Covariance on the robot position. """
        self.P[:3, :3] = value

    def addLandmark(self):
        """ Add rows and columns for new landmarks."""
        z = np.zeros((2, len(self.P)+2), dtype=self.P.dtype)

        # add two rows
        self.P = np.concatenate((self.P, z[:, :-2]), axis=0)

        # add two columns
        self.P = np.concatenate((self.P, z.T), axis=1)

    def getC(self, i):
        """" Getter: landmark covariance (2x2). """
        index = 3 + 2*i

        return self.P[index:index+2, index:index+2]

    def setC(self, i, value):
        """" Setter: landmark covariance (2x2). """
        index = 3 + 2*i

        self.P[index:index+2, index:index+2] = value

    def getD(self, i):
        """" Getter: robot - landmark covariance (2x3). """
        index = 3 + 2*i

        return self.P[index:index+2, :3]

    def setD(self, i, value):
        """" Setter: robot - landmark covariance (2x3). """
        index = 3 + 2*i

        self.P[index:index+2, :3] = value

    def getE(self, i):
        """" Getter: landmark - robot covariance (3x2). """
        index = 3 + 2*i

        return self.P[:3, index:index+2]

    def setE(self, i, value):
        """" Setter: landmark - robot covariance (3x2). """
        index = 3 + 2*i

        self.P[:3, index:index+2] = value

    def getF(self, i, j):
        """" Getter: landmark - landmark covariance on the row (2x2). """
        rowIdx = 3 + 2*i
        colIdx = 3 + 2*j

        return self.P[rowIdx:rowIdx+2, colIdx:colIdx+2]

    def setF(self, i, j, value):
        """" Setter: landmark - landmark covariance on the row (2x2). """
        rowIdx = 3 + 2*i
        colIdx = 3 + 2*j

        self.P[rowIdx:rowIdx+2, colIdx:colIdx+2] = value

    def getG(self, i, j):
        """" Getter: landmark - landmark covariance on the columns (2x2). """

        return self.getF(j, i)

    def setG(self, i, j, value):
        """" Setter: landmark - landmark covariance on the columns (2x2). """

        self.setF(j, i, value)


def RanSaC(data, D, S, X, C, N, plt=False):
    """ Detect wall inside a set of data using the RanSaC algorithm.

    data is a DataFrame object from the pandas module.
    D is distance max in degree between min and max angle in the random set.
    S is the number of samples to compute initial line.
    X is max distance a point may be from wall to get associated to wall.
    C is the number of points that must lie on a line for it to be validated.
    N is the max number of trials to attempt to find a lines.
    plt is an optional matplotlib object for plotting.
    """

    landmarks = []
    trial = count = 0
    while len(data) > C and trial < N:
        trial += 1

        # Select a random set of S samples in a range of D degrees
        sampleSet = getSampleSet(data, S, D)

        # Find the nearest line and count all points within a distance X
        lineParam = fitLeastSquareLine(sampleSet.lidarX, sampleSet.lidarY)
        count, data = countPointsNearLine(data, lineParam, X)

        # If there is enough points (more than consensus C)
        if count > C:

            # Recompute the nearest line with all points near previous line
            x = data["lidarX"][(data["distToWall"] == True)]
            y = data["lidarY"][(data["distToWall"] == True)]
            wallParam = fitLeastSquareLine(x, y)

            # Remove points near the new line from the data set
            count, data = countPointsNearLine(data, wallParam, X)
            data = data[data.distToWall != True]

            # Add the line to the list of wall detected
            landmarks.append(wallParam)

            if plt:
                p = np.poly1d(wallParam)
                xp = np.linspace(-4, 5)
                xp = np.linspace(-8, -4)
                plt.scatter(x, y, color=PLOT_BLUE, s=2)
                plt.plot(xp, p(xp), color=PLOT_BLUE)
                plt.scatter(sampleSet.lidarX, sampleSet.lidarY, s=3,
                            color='tomato')

    return landmarks, trial


def lmAssociation(X, localLandmarks):
    """ Return two lists of new and re-observed landmarks.

    Landmarks are associated using a nearest-neighbor approach.
    localLandmarks is the list of landmarks observed.
    X is a StateMatrix object.
    """

    newLandmarkds, reobservedLandmark = [], []
    count = 0
    for candidate in localLandmarks:

        # Selecting the nearest landmark already observed
        minDistance = [10**9, -1]
        for j in range(X.lmLength):
            d = distPointToPoint(candidate['point'], X.getLm(j))
            if d < minDistance[0]:
                minDistance = [d, j]

        # If it is near enough it is a re-observation
        if minDistance[0] < D_VALIDATION:
            candidate['index'] = minDistance[1]
            reobservedLandmark.append(candidate)
            X.lmOccurrence[candidate['index']] += 1

        # If not it is a new landmark
        else:
            candidate['index'] = X.lmLength + count
            count += 1
            newLandmarkds.append(candidate)

    # Only keep landmarks that have been observe enough time
    def condition(m): return X.lmOccurrence[m['index']] >= N_VALIDATION
    reobservedLandmark = [m for m in reobservedLandmark if condition(m)]

    return newLandmarkds, reobservedLandmark


def makeLandmark(param):
    """ Return a dictionary characterizing a landmark from its parameters.

    Wall landmarks are characterized by the point corresponding to the
    intersection of the wall line and its perpendicular passing through the
    origin (0, 0). The wall line is characterized by a vector (a, c) such as
    its equation is given by y = ax + c.

    Return a dictionary with landmark coordinate, parameters and occurrence.
    """

    return {"point": lmParamToPoint(*param),
            "param": param,
            "occurrence": 1}


def lmParamToPoint(a, c):
    """ Return the coordinates of a landmark from its line parameters.

    Wall landmarks are characterized by the point corresponding to the
    intersection of the wall line and its perpendicular passing through the
    origin (0, 0). The wall line is characterized by a vector (a, c) such as
    its equation is given by y = ax + c.
    """

    xp = float(-c*a / (1+a**2))
    yp = float(c / (1+a**2))

    return [xp, yp]


def lmPointToParam(xp, yp):
    """ Return the line parameters of a landmark from its coordinates.

    Wall landmarks are characterized by the point corresponding to the
    intersection of the wall line and its perpendicular passing through the
    origin (0, 0). The wall line is characterized by a vector (a, c) such as
    its equation is given by y = ax + c.

    xp and yp are the landmark coordinate.
    """

    a = -xp/yp
    b = yp*(1+a**2)

    return [a, b]


def countPointsNearLine(data, line, X):
    """ Count the number of point near a given line.

    Return the number of point within a distance X from a given line and add
    the column 'distToWall' to the data set corresponding to if the point is
    inside the range or not.
    data is a DataFrame object from the pandas module.
    line is a vector (a, c) such as the line equation is y = ax + c.
    """

    data['distToWall'] = False

    count = 0
    for idx, row in data.iterrows():
        d = distPointToLine([row.lidarX, row.lidarY], line)

        if d < X:
            data.loc[idx, 'distToWall'] = True
            count += 1

    return count, data


def distPointToPoint(p1, p2):
    """ Return the euclidean distance between two points.

    p1 and p2 are coordinate vectors (x, y).
    """

    return np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1]])


def distPointToLine(point, line):
    """ Return the euclidean distance between a point and a line.

    point is a coordinate vector (x, y).
    line is a vector (a, c) such as the line equation is y = ax + c.
    """

    [xp, yp] = point
    [a, c] = line
    b = -1

    return abs((a*xp + b*yp + c) / np.linalg.norm([a, b]))


def fitLeastSquareLine(x, y):
    """ Least squares single degree polynomial fit.

    Fit a polynomial y = ax + c to points (x, y) and returns the vector of
    coefficients [a, c] that minimizes the squared error.
    x and y or vectors of points.
    """

    return np.polyfit(x, y, 1)


def getSampleSet(data, S, D):
    """ Return a random set of S elements within D degrees inside a data set.

    data is a DataFrame object from the pandas module.
    S is the size of the random set.
    D is distance max in degree between min and max angle in the random set.
    """

    sampleSet = pd.DataFrame(columns=data.columns)

    # Getting a first random sample
    sample = getRandomSample(data)
    sampleSet.loc[0] = sample

    for i in range(S-1):

        # Appending a new sample within the D range
        s = getRandomSample(data, sample.angle-D/2, sample.angle+D/2)
        sampleSet.loc[len(sampleSet)] = s

    return sampleSet


def getRandomSample(data, minimum=-1, maximum=361):
    """ Return a random sample within a data set.

    data is a DataFrame object from the pandas module.
    minimum and maximum arguments are optional angles (degree) limits in which
    the random sample needs to be, by default there is no limits.
    """

    # Get a random sample
    sample = data.sample(n=1).iloc[0]  # take a random sample

    # Take another one if it is not in the limits
    while sample.angle < minimum or sample.angle > maximum:
        sample = data.sample(n=1).iloc[0]

    return sample


def printMatrix(*args):
    """ Print matrices in a readable way. """

    for M in args:
        if type(M).__module__ == np.__name__ or type(M) == "list":
            for row in M:
                s = ["{:8.3}"] * len(row)
                s = ", ".join(s)
                print(s.format(*[i for i in row]))
        else:
            print(M)


if __name__ == '__main__':
    main()
