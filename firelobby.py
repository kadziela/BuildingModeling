"""Airflow through lobby - simulation"""

import numpy as np
from itertools import combinations


class Space():
    """Space like room, firelobby, staircase. Class contains all needed
    parameters of space"""

    def __init__(self, name='', pressure=0, flow=0, isConstantPressure=True):
        self.name = name
        self.pressure = pressure
        self.flow = flow
        self.isConstantPressure = isConstantPressure

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return f'{self.name}'


class Opening():
    """Opening is every route through which air can flow like: undercutting
    the door, leaks on the door, air transfer etc."""

    def __init__(self, name, space1, space2, Kfactor=0, coefficient=0.5,
                 flow=0):
        self.name = name
        self.Kfactor = Kfactor
        self.coefficient = coefficient
        self.flow = flow
        self.s1 = space1
        self.s2 = space2

    def set_Kfactor(self, flow, resistance, coefficient=0.5):
        self.Kfactor = flow / (resistance) ** coefficient

    def __str__(self):
        return self.name


class Zone():
    """Zone containing Spaces with openings between that Spaces"""

    def __init__(self, spaces=[], openings=[]):
        self.spaces = spaces
        self.openings = openings
        self.X = np.nan
        self.Y = np.nan
        self.Z = np.nan

    def add_space(self, space):
        self.spaces.append(space)

    def add_opening(self, opening):
        self.spaces.append(opening)

    def calculate(self):
        # 1 Calculation of substitute Kv
        # openingList = []
        # for opening in self.openings:
        #    openingList.append((opening.s1, opening.s2))
        # partitions = list(combinations(self.spaces, 2))
        # fKv3=kv1*kv2*kv3/(kv2**2*kv3**2+kv1**2*kv3**2+kv1**2*kv2**2)**0.5
        # fKv4=kv1*kv2*kv3*kv4/(kv2**2*kv3**2*kv4**2+kv1**2*kv3**2*kv4**2+kv1**2*kv2**2*kv4**2+kv1**2*kv2**2*kv3**2)**0.5

        # 2 Array construction
        s = len(self.spaces)
        o = len(self.openings)
        A = np.eye(s)
        B = np.zeros((o, s))
        C = np.zeros(shape=(s, o))
        D = np.eye(o)
        E = np.zeros(s)
        F = np.zeros(o)

        for opening in self.openings:
            x = self.openings.index(opening)
            y1 = self.spaces.index(opening.s1)
            y2 = self.spaces.index(opening.s2)
            F[x] = opening.Kfactor * \
                abs(opening.s1.pressure - opening.s2.pressure) ** \
                opening.coefficient

            if opening.s1.pressure > opening.s2.pressure:
                C[y1][x] = -1
                C[y2][x] = 1
            elif opening.s1.pressure < opening.s2.pressure:
                C[y1][x] = 1
                C[y2][x] = -1

        self.X = np.concatenate(
            (np.concatenate((A, C), axis=1), np.concatenate((B, D), axis=1)),
            axis=0)
        self.Y = np.concatenate((E, F), axis=0)

        # 3 Calculation
        self.Z = np.linalg.solve(self.X, self.Y)


S1 = Space('Staircase', pressure=50)
S2 = Space('FireLobby', pressure=20)
S3 = Space('LiftShaft', pressure=30)
S4 = Space('Corridor')
spaces = [S1, S2, S3, S4]

O1 = Opening('StaircaseDoorLeakage', S1, S2, 30)
O2 = Opening('LiftDoorLeakage', S3, S2, 140)
O3 = Opening('Transfer', S2, S4, 370)
openings = [O1, O2, O3]

Lobby = Zone()
Lobby.spaces = spaces
Lobby.openings = openings

Lobby.calculate()

# Tests

X = np.array([[1, 0, 0, 0, -1, 0, 0],
             [0, 1, 0, 0, 1, 1, -1],
             [0, 0, 1, 0, 0, -1, 0],
             [0, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]]
             )

Y = np.array([0, 0, 0, 0, 30*30**0.5, 140*10**0.5, 370*20**0.5])
Z = np.linalg.solve(X, Y)
