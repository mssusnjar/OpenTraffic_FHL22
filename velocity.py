from enum import Enum
import numpy as np
import math

# Configuration
MIN_COVERAGE = 0.7  # 70%
FPS = 24
LIFETIME = 20*FPS
MIN_VEHICLES = 10

# Path constants
way_x = [295, 500, 1420, 1485]
way_y = [728, 574, 811, 617]
L0 = 34.2  # m

# Median line
x1 = (way_x[0] + way_x[1]) / 2
y1 = (way_y[0] + way_y[1]) / 2

x2 = (way_x[2] + way_x[3]) / 2
y2 = (way_y[2] + way_y[3]) / 2

L = (x2-x1)**2 + (y2-y1)**2


class State(Enum):
    APPROACHING = 1
    MOVING = 2
    FINISHED = 3
    LOST = 4

class Vehicle:
    def __init__(self, state, startX, startY, startT, currX, currY, currT, vel):
        self.state = state
        self.startX = startX
        self.startY = startY
        self.startT = startT
        self.currX = currX
        self.currY = currY
        self.currT = currT
        self.vel = vel

def insideSegment(x0, y0, x1, y1, x2, y2):
    dotproduct = (x1 - x0) * (x2 - x0) + (y1 - y0) * (y2 - y0)
    return dotproduct < 0

def triangleArea(x1, y1, x2, y2, x3, y3):
    return abs(1/2*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))

def insidePolygon(x0, y0, x1, x2, x3, x4, y1, y2, y3, y4):
    area1 = triangleArea(x0,y0,x1,y1,x2,y2) + triangleArea(x0,y0,x2,y2,x3,y3) + triangleArea(x0,y0,x3,y3,x4,y4) + triangleArea(x0,y0,x4,y4,x1,y1)
    area2 = triangleArea(x1,y1,x2,y2,x4,y4) + triangleArea(x2,y2,x3,y3,x4,y4)
    return abs(area1-area2) < 0.2*area2

# project point (x0,y0) onto a line defined with (x1,y1) and (x2,y2)
def projection(x0, y0, x1, y1, x2, y2):
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    if a == 0:
        x = x0
    else:
        d = y0 + x0/a
        x = (d-b)/(a+1/a)
    y = a*x + b
    return x, y

def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def addMeasurement(board, frame, id, bb1, bb2, bb3, bb4):
    item = board.get(id, False)
    x = bb1 + bb3/2  # left + width/2
    y = bb2 + bb4/2  # top + height/2
    xp, yp = projection(x, y, x1, y1, x2, y2)
    #isBetween = insidePolygon(x, y, *way_x, *way_y)
    isBetween = insideSegment(xp, yp, x1, y1, x2, y2) and insidePolygon(x, y, *way_x, *way_y)
    state = State.APPROACHING if item == False else item.state
    match state:
        case State.APPROACHING:
            if isBetween:
                board[id] = Vehicle(State.MOVING, xp, yp, frame, xp, yp, frame, 0)
        case State.MOVING:
            item.currX = xp
            item.currY = yp
            item.currT = frame
            if (not isBetween):
                item.state = State.FINISHED
                percentage = ((item.currX - item.startX) ** 2 + (item.currY - item.startY) ** 2) / L
                #print('Finished',id,'in frame',frame,'with coverage',percentage)
                if (percentage > MIN_COVERAGE):
                    time = (item.currT-item.startT)/FPS
                    item.vel = percentage * L0 / time * 3.6
                    #print('and velocity',item.vel)
    return board

def estimateSpeed(board, tf, vel):
    velocities = []
    delete = []
    for key, item in board.items():
        if (item.currT + LIFETIME < tf):
            delete.append(key)
            #print('Removed',key,'in frame',tf)
        elif item.state == State.FINISHED:
            if (item.vel > 0):
                velocities.append(item.vel)
            else:
                delete.append(key)
                #print('Removed', key, 'in frame', tf)
    for key in delete:
        del board[key]
    if len(velocities) == 0:
        return board, vel
    avgVel = np.mean(velocities)
    if len(velocities) < MIN_VEHICLES:
        ratio = len(velocities)/MIN_VEHICLES
        avgVel = ratio*avgVel + (1-ratio)*vel
    return board, avgVel

if __name__ == "__main__":
    txt_path = "D:/repos/OpenTrafficTest/Yolov5_DeepSort_Pytorch/runs/track/exp16/tracks/traffic_out_1"
    board = {}
    vel = 40  # expected velocity
    currFrame = 0
    with open(txt_path + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            frame, id, bb1, bb2, bb3, bb4, _, _, _, _ = np.loadtxt(line.split())
            board = addMeasurement(board, frame, id, bb1, bb2, bb3, bb4)
            if (currFrame != frame and frame % 10 == 1):
                currFrame = frame
                board, vel = estimateSpeed(board, currFrame, vel)
                print(currFrame, vel)


