"""
    THIS FILE CONTAINS ALL THE MODIFYABLE CONFIGURATION PARAMETERS FOR IMPORTANT THINGS
    SUCH AS THE TRAINING LOOP
"""

# TRAINING
ROLLOUT_LENGTH = 2048 # size of the buffer that stores environment steps (2048 is about 4.5 games in early stages)
NUM_UPDATES = 5000 # number of times to fill upp the rollout buffer and train
STORE_FREQUENCY = 10 # how many updates to play before storing a game
DEFAULT_TRAIN_UPDATE_STEPS = 2048
