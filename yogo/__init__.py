class DefaultHyperparams:
    EPOCHS = 32
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    LABEL_SMOOTHING = 0.01
    DECAY_FACTOR = 10
    WEIGHT_DECAY = 1e-2
    OPTIMIZER_TYPE = "adam"


from . import infer
from . import model
from . import utils
from . import data

__all__ = ("infer", "model", "utils", "data")
