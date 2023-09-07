class DefaultHyperparams:
    EPOCHS = 64
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    LABEL_SMOOTHING = 0.01
    DECAY_FACTOR = 10
    WEIGHT_DECAY = 5e-2
    OPTIMIZER_TYPE = "adam"
    IOU_WEIGHT = 5.0
    NO_OBJ_WEIGHT = 0.5
    CLASSIFY_WEIGHT = 1.0
    HEALTHY_WEIGHT = 1.0
    ANCHOR_H = 0.05551774140353888
    ANCHOR_W = 0.04250100424705710
