from catanatron.gym.utils import get_matrices_path


def generate_arrays_from_file(
    games_directory,
    batchsize,
    label_column,
    learning="Q",
    label_threshold=None,
):
    inputs = []
    targets = []
    batchcount = 0

    (
        samples_path,
        board_tensors_path,
        actions_path,
        rewards_path,
        main_path,
    ) = get_matrices_path(games_directory)
    while True:
        with open(samples_path) as s, open(actions_path) as a, open(rewards_path) as r:
            next(s)  # skip header
            next(a)  # skip header
            rewards_header = next(r)  # skip header
            label_index = rewards_header.rstrip().split(",").index(label_column)
            for i, sline in enumerate(s):
                try:
                    srecord = sline.rstrip().split(",")
                    arecord = a.readline().rstrip().split(",")
                    rrecord = r.readline().rstrip().split(",")

                    state = [float(n) for n in srecord[:]]
                    action = [float(n) for n in arecord[:]]
                    reward = float(rrecord[label_index])
                    if label_threshold is not None and reward < label_threshold:
                        continue

                    if learning == "Q":
                        sample = state + action
                        label = reward
                    elif learning == "V":
                        sample = state
                        label = reward
                    else:  # learning == "P"
                        sample = state
                        label = action

                    inputs.append(sample)
                    targets.append(label)
                    batchcount += 1
                except Exception as e:
                    print(i)
                    print(s)
                    print(e)
                if batchcount > batchsize:
                    X = np.array(inputs, dtype="float32")
                    y = np.array(targets, dtype="float32")
                    yield (X, y)
                    inputs = []
                    targets = []
                    batchcount = 0
