import os


def formatSecs(secs):
    return "{0:.3f} secs".format(secs)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
