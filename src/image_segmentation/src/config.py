BATCH_SIZE = 4
LR = 0.001

ALL_CLASSES = {
    'background': [[0,0,0], 0],
    'quadrat': [[255, 0, 0], 1],
    'soil': [[153, 51, 255], 2],
    'kura_clover': [[0, 200, 0], 3]
}

REDUCED_CLASSES = {
    'quadrat': [[255, 0, 0], 0],
    'soil': [[153, 51, 255], 1],
    'kura_clover': [[0, 200, 0], 2]
}

NUM_CLASSES = len(ALL_CLASSES)


