import cv2
import numpy as np
import pygame
from cvzone.HandTrackingModule import HandDetector
import random

width = 1280
height = 720
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

scale = 4
widthKeyboard = 275 * scale
heightKeyboard = 72 * scale
initialWarpPoints = [[37, 271], [1197, 254], [134, 481], [1107, 468]]
pts1 = np.float32(initialWarpPoints)
pts2 = np.float32([[0, 0], [widthKeyboard, 0], [0, heightKeyboard], [widthKeyboard, heightKeyboard]])

# Variables
isFirstFrame = True
currentKey = 'y'
currentKeyPressed = False
scoreCorrect = 0
scoreWrong = 0
delayCount = 0

# Bounding box of each key and the correct finger
keyLocations = {
    # First Row
    'q': [28, 0, 16, 15, 'left_pinky'],
    'w': [47, 0, 16, 15, 'left_ring'],
    'e': [67, 0, 16, 15, 'left_middle'],
    'r': [86, 0, 16, 15, 'left_index'],
    't': [105, 0, 16, 15, 'left_index'],
    'y': [124, 0, 16, 15, 'right_index'],
    'u': [143, 0, 16, 15, 'right_index'],
    'i': [162, 0, 16, 15, 'right_middle'],
    'o': [182, 0, 16, 15, 'right_ring'],
    'p': [201, 0, 16, 15, 'right_pinky'],
    # Second Row
    'a': [32, 19, 16, 15, 'left_pinky'],
    's': [52, 19, 16, 15, 'left_ring'],
    'd': [71, 19, 16, 15, 'left_middle'],
    'f': [90, 19, 16, 15, 'left_index'],
    'g': [109, 19, 16, 15, 'left_index'],
    'h': [129, 19, 16, 15, 'right_index'],
    'j': [148, 19, 16, 15, 'right_index'],
    'k': [167, 19, 16, 15, 'right_middle'],
    'l': [187, 19, 16, 15, 'right_ring'],
    # Third Row
    'z': [42, 37, 16, 16, 'left_pinky'],
    'x': [62, 37, 16, 16, 'left_ring'],
    'c': [81, 37, 16, 16, 'left_middle'],
    'v': [100, 37, 16, 16, 'left_index'],
    'b': [119, 37, 16, 16, 'left_index'],
    'n': [138, 37, 16, 16, 'right_index'],
    'm': [158, 37, 16, 16, 'right_index']
}

pygame.init()

# Create Window/Display
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ai Typing Tutor")

# Initialize Clock for FPS
fps = 30
clock = pygame.time.Clock()

# Creating a dic for positions of the keys in the background image
rows = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm']]
rowsStart = [192, 214, 235]
keyLocationsBackground = {}
for i, row in enumerate(rows):
    for j, alphabet in enumerate(row):
        keyLocationsBackground[alphabet] = [rowsStart[i] + 76 * j, 364 + 74 * i, 62, 62]

detector = HandDetector(detectionCon=0.4, maxHands=2)

fingerIds = {
    "8": "index",
    "12": "middle",
    "16": "ring",
    "20": "pinky"
}


# warp point to find the correct locations of the finger tips on the warped keyboard image
def warpPoint(p, matrix):
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    return int(px), int(py)


# check if finger tip point is in the key bounding box
def checkInside(point, x, y, w, h):
    return x < point[0] < x + w and y < point[1] < y + h


while True:

    imgBackground = cv2.imread("Background.png")
    currentKeyPressed = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            myKey = getattr(pygame, 'K_{}'.format(currentKey))
            if event.key == myKey:
                print(f'{currentKey} key was pressed')
                currentKeyPressed = True
            # else:
            #         currentKeyPressed = False

    success, img = cap.read()
    if isFirstFrame: cv2.imwrite("Sample.jpg", img)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (widthKeyboard, heightKeyboard))
    imgWarp = cv2.flip(imgWarp, 1)

    hands, img = detector.findHands(img, flipType=False)

    for point in initialWarpPoints:
        cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED)

    if hands:
        if len(hands) == 2:
            cv2.putText(imgBackground, "Detection: Solid", (50, 50), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 2)
        else:
            cv2.putText(imgBackground, "Detection: Weak", (50, 50), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 255), 2)

        for hand in hands:
            handType = hand['type'].lower()

            if currentKeyPressed and delayCount == 0:
                ######### BBOX OF KEY #########
                # get the bbox info of the correct key to check the finger location
                key = currentKey
                value = keyLocations[key]
                x, y, w, h = value[0] * scale, value[1] * scale, value[2] * scale, value[3] * scale
                correctFinger = value[4]
                cv2.rectangle(imgWarp, (x, y), (x + w, y + h), (50, 200, 50), cv2.FILLED)

                # which finger is pressing the y key

                for id, finger in fingerIds.items():

                    ######### TIP OF THE FINGER #########
                    point = hand["lmList"][int(id)]
                    px, py = warpPoint(point, matrix)
                    px = widthKeyboard - px  # flip the point
                    cv2.circle(imgWarp, (px, py), 5, (0, 0, 255), cv2.FILLED)
                    if checkInside((px, py), x, y, w, h):
                        print(handType + "_" + finger, correctFinger)
                        if handType + "_" + finger == correctFinger:
                            print("Correct")
                            scoreCorrect += 1
                            currentKey = list(keyLocations)[random.randint(0, 25)]
                            color = (0, 255, 0)
                        else:
                            print("wrong")
                            scoreWrong += 1
                            color = (0, 0, 255)
                        delayCount = 1

                        valueCurrent = keyLocationsBackground[currentKey]
                        x, y, w, h = valueCurrent
                        cv2.rectangle(imgBackground, (x, y), (x + w, y + h), color, cv2.FILLED)
                        cv2.rectangle(imgBackground, (505, 129), (505 + 285,129+ 177), color, cv2.FILLED)

    # Draw the bounding bbox on the warp image
    for key, value in keyLocations.items():
        x, y, w, h = value[0] * scale, value[1] * scale, value[2] * scale, value[3] * scale
        cv2.rectangle(imgWarp, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.imshow("Image Warp", imgWarp)

    # # Draw on the Background image
    # if currentKeyPressed:

    # Draw all the alphabets on the background image
    for key, val in keyLocationsBackground.items():
        x, y, w, h = val
        cv2.putText(imgBackground, key, (x + 15, y + 45), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.putText(imgBackground, currentKey, (590, 260), cv2.FONT_HERSHEY_PLAIN,
                10, (255, 255, 255), 20)
    cv2.putText(imgBackground, str(scoreCorrect), (185, 245), cv2.FONT_HERSHEY_PLAIN,
                4, (255, 255, 255), 5)
    cv2.putText(imgBackground, str(scoreWrong), (1035, 245), cv2.FONT_HERSHEY_PLAIN,
                4, (255, 255, 255), 5)

    if delayCount != 0:
        delayCount += 1
        if delayCount >= 5:
            delayCount = 0

    # OpenCV  Display
    imgRGB = cv2.cvtColor(imgBackground, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    frame = pygame.surfarray.make_surface(imgRGB).convert()
    frame = pygame.transform.flip(frame, True, False)
    window.blit(frame, (0, 0))

    # Update Display
    pygame.display.update()
    # Set FPS
    clock.tick(fps)
