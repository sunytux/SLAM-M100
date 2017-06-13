#! /usr/bin/python3

# Copyright © 2017 Sami Mezhoud <sami.mezhoud@ulb.ac.be>
# This work is free. You can redistribute it and/or modify it under the
# terms of the Do What The Fuck You Want To Public License, Version 2,
# as published by Sam Hocevar. See the COPYING file for more details.

""" This script reads a csv file containing raw LiDAR data and drone
    measurements, process its data by computing absolute locations of LiDAR
    measurements, pose of the drone, etc and saves it to a pickle file.

Usage:
    plotLidar.py <CSV_FILE> <PICKLE_FILE>

Arguments:
    <CSV_FILE>         Input csv file containing raw data.
    <PICKLE_FILE>      Output pickle file containing processed data.

Options:
    -h, --help
"""


import math
import numpy as np
import pandas as pd
import time
import sys
from docopt import docopt

BYTE_SEPARATOR = "-"
LIDAR_OFFSET_1 = 0 * math.pi/180  # LiDAR test
LIDAR_OFFSET_2 = 29 * math.pi/180  # manSlam02
LIDAR_OFFSET_3 = 50 * math.pi/180  # manSlam03
LIDAR_OFFSET_4 = 63 * math.pi/180  # manSlam04


def polarToCartesian(angle, distance):
    """ angle are in degrees """
    return [math.cos(angle * math.pi/180) * distance,
            math.sin(angle * math.pi/180) * distance]


class LiDARFrame(object):

    """Data content of a message sent by the XV-11 LiDAR device to the Embedded
    Onboard Device (EOD). It takes as argument the corresponding list of 22
    bytes. """

    INDEX_OFFSET = 0xa0

    def __init__(self, byteList):
        self.bytes = byteList

        self.len = len(self.bytes)

        self.index = int(self.bytes[1]) - self.INDEX_OFFSET

        self.angles = [self.index*4,
                       self.index*4 + 1,
                       self.index*4 + 2,
                       self.index*4 + 3]

        self.chkSum = self.bytes[-2] + (self.bytes[-1] << 8)

        self.speed = (self.bytes[2] | self.bytes[3] << 8) / 64.0

        self.data0 = self.bytes[4:8]
        self.data1 = self.bytes[8:12]
        self.data2 = self.bytes[12:16]
        self.data3 = self.bytes[16:20]

        dist0 = (self.data0[0] | ((self.data0[1] & 0b111111) << 8)) / 1000  # m
        dist1 = (self.data1[0] | ((self.data1[1] & 0b111111) << 8)) / 1000
        dist2 = (self.data2[0] | ((self.data2[1] & 0b111111) << 8)) / 1000
        dist3 = (self.data3[0] | ((self.data3[1] & 0b111111) << 8)) / 1000

        self.distances = [dist0, dist1, dist2, dist3]

        quality0 = self.data0[2] | (self.data0[3] << 8)
        quality1 = self.data1[2] | (self.data1[3] << 8)
        quality2 = self.data2[2] | (self.data2[3] << 8)
        quality3 = self.data3[2] | (self.data3[3] << 8)

        self.qualities = [quality0, quality1, quality2, quality3]

    def checksum(self):
        """Compute and check if the checksum is correct"""
        # group the data by word, little-endian
        data = self.bytes

        data_list = []
        for t in range(10):
            data_list.append(data[2*t] + (data[2*t+1] << 8))

        # compute the checksum on 32 bits
        chk32 = 0
        for d in data_list:
            chk32 = (chk32 << 1) + d

        # return a value wrapped around on 15bits, and truncated to still fit
        # into 15 bits
        # wrap around to fit into 15 bits
        checksum = (chk32 & 0x7FFF) + (chk32 >> 15)
        checksum = checksum & 0x7FFF  # truncate to 15 bits

        return self.chkSum == checksum

    def printDebug(self):
        output = ""

        output += format("Angles [°]", '15') + "|"
        output += format(self.angles[0], '12')
        output += format(self.angles[1], '12')
        output += format(self.angles[2], '12')
        output += format(self.angles[3], '12') + "\n"
        output += "-"*65 + '\n'

        output += format("Distance [mm]", '15') + "|"
        output += format(self.distances[0], '12')
        output += format(self.distances[1], '12')
        output += format(self.distances[2], '12')
        output += format(self.distances[3], '12') + "\n"

        output += format("Quality", '15') + "|"
        output += format(self.qualities[0], '12')
        output += format(self.qualities[1], '12')
        output += format(self.qualities[2], '12')
        output += format(self.qualities[3], '12') + "\n"

        output += format("Speed [rpm]: ", '15') + "|"
        output += '{:^15}'.format(self.speed) + "\n"

        output += format("chkSum: ", '15') + "|"
        output += '{:^15}'.format(self.checksum()) + "\n"

        output += ", ".join([str(hex(b)) for b in self.bytes])

        print(output)

    def __str__(self):
        return str(self.bytes)


def convertToByteList(strFrame):
    l = strFrame.strip().split(BYTE_SEPARATOR)[:-1]
    l = [int("0x"+j, 16) for j in l]
    return bytes(l)


def getRawData(path):
    data = pd.read_csv(path, header=None,)
    data.columns = ['time', 'q0', 'q1', 'q2', 'q3',
                    'droneX', 'droneY', 'droneZ', 'frame']
    return data


def getData(path, lidarOffset=0):
    start = time.time()
    print('Loading data: {}'.format(path.split('/')[-1]))

    data = getRawData(path)

    print('Processing data... ', end="")
    sys.stdout.flush()  # Avoid printing delay

    newDataSet = pd.DataFrame(columns=['time', 'roll', 'pitch', 'yaw',
                                       'quality',
                                       'angle', 'distance',
                                       'droneX', 'droneY', 'droneZ',
                                       "lidarX", "lidarY", "lidarZ"])

    index = 0
    for _, row in data.iterrows():
        frame = LiDARFrame(convertToByteList(row.frame))

        [roll, pitch, yaw] = quaternionToRpy(row.q0, row.q1, row.q2, row.q3)
        rotationMatrix = getRotationMatrix(roll, pitch, yaw + lidarOffset)

        for i in range(4):
            if frame.qualities[i] > 0:

                # Local coordinates
                [x_local, y_local] = polarToCartesian(frame.angles[i],
                                                      frame.distances[i])

                # Rotated coordinates
                [u, v, w] = np.dot([x_local, y_local, 0], rotationMatrix)

                # Absolute coordinates
                u += row.droneX
                v += row.droneY
                w += row.droneZ

                newDataSet.loc[index+i] = [
                    (row.time - data.time[0]) / 1000,  # seconds
                    roll, pitch, yaw,
                    frame.qualities[i],
                    frame.angles[i],
                    frame.distances[i],
                    row.droneX, row.droneY, row.droneZ,
                    u, v, w
                ]
                index += 1

    end = time.time()

    print(" done in {:,.2f}s.".format(end - start))

    return newDataSet


def getRotationMatrix(a, b, c):
    """ Compute Rotation Matrix given RPY euler angles. """

    # Roll
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(a), -math.sin(a)],
                    [0, math.sin(a), math.cos(a)]
                    ])

    # pitch
    R_y = np.array([[math.cos(b), 0, math.sin(b)],
                    [0, 1, 0],
                    [-math.sin(b), 0, math.cos(b)]
                    ])
    # yaw
    R_z = np.array([[math.cos(c), -math.sin(c), 0],
                    [math.sin(c), math.cos(c), 0],
                    [0, 0, 1]
                    ])

    return np.dot(R_z, np.dot(R_y, R_x))


def quaternionToRpy(q0, q1, q2, q3):
    """ Compute the roll, pitch and yaw angles corresponding to drone
        quaternion. Angles are in radian."""
    roll = math.atan2(2.0 * (q3*q2 + q0*q1), 1.0 - 2.0 * (q1*q1 + q2*q2))
    pitch = math.asin(2.0 * (q2*q0 - q3*q1))
    yaw = math.atan2(2.0 * (q3*q0 + q1*q2), - 1.0 + 2.0 * (q0*q0 + q1*q1))

    return[roll, pitch, yaw]


def main():
    data = getData(docopt(__doc__)['<CSV_FILE>'], LIDAR_OFFSET_1)

    # TODO automatize cleaning process ?
    # Cleaning exp 'manSlam2'
    # data = cleanManSlam3(data)

    print('Saving to {}'.format(docopt(__doc__)['<PICKLE_FILE>']))
    data.to_pickle(docopt(__doc__)['<PICKLE_FILE>'])


def cleanManSlam2(data):
    return data[(data.lidarY < -2) | (data.lidarY > 4)]


def cleanManSlam3(data):
    return data[(data.distance > 2) & (data.distance < 5) & (data.droneZ > 1)]


if __name__ == '__main__':
    main()
