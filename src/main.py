from pylsl import StreamInlet, resolve_streams
import numpy as np
import matplotlib.pyplot as plt
from ccn import CCN
from map_to_input import InputPeripherals

from pathlib import Path
import json
import time

streams = resolve_streams()
inlet = StreamInlet(streams[0])
channel_labels = ["Electrode", "Delta", "Theta", "Low Alpha", "High Alpha",
                  "Low Beta", "High Beta", "Low Gamma", "Mid Gamma"]
num_channels = len(channel_labels)

sampling_freq = 512
buf_size = 2048
chunk_size = 32
data = np.zeros((buf_size, num_channels))

states = ["NEUTRAL", "UP", "DOWN", "LEFT", "RIGHT"]
n_state = len(states)
calibration_duration = 60  # seconds per state
calibration_dict = {k: [] for k in range(n_state)}
ccn = CCN(n_components_gmm=n_state, n_components_hmm=n_state, n_trials_hmm=1,
            wavelet='db6', level=1, random_state=0)

def get_complete_chunk(inlet, chunk_size):
    buffer = []
    total = 0
    while total < chunk_size:
        chunk, t = inlet.pull_chunk(max_samples=chunk_size-total)
        if t:
            chunk = np.array(chunk)
            buffer.append(chunk)
            total += chunk.shape[0]
        else:
            time.sleep(0.005)

    full_chunk = np.concatenate(buffer, axis=0)
    return full_chunk[:chunk_size]

if Path(Path(__file__).parent / "features_dict.txt").is_file():
    with open(Path(__file__).parent / "features_dict.txt", 'r') as f:
        calibration_dict = {int(k): v for k, v in json.loads(f.read()).items()}
        ccn.calibrate(calibration_dict, using_preset=True)
else:
    print("Starting calibration...")
    for k, state in enumerate(states):
        input(f"\nPress Enter to start calibration for '{state}' ({calibration_duration} sec)")
        start = time.time()

        while time.time()-start < calibration_duration:
            chunk = get_complete_chunk(inlet, chunk_size)
            calibration_dict[k].append((chunk[:, 0]).tolist())
            remaining = int(calibration_duration - (time.time() - start))
            print(f"THINK OF {state}: {remaining} sec remaining", end="\r")

        print(f"\nFinished calibration for '{state}'.")

    ccn.calibrate(calibration_dict)

inputs = InputPeripherals()

##############################
#
# DISPLAY EEG STREAM
#
##############################

plot = False
fig = None
axs = None
lines = None 

if plot:
    plt.ion()
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 12), sharex=True)
    time_axis = np.linspace(0, buf_size/sampling_freq, buf_size)

    lines = []
    for ch in range(num_channels):
        line, = axs[ch].plot(time_axis, data[:, ch], 'k')
        axs[ch].set_ylabel(channel_labels[ch])
        lines.append(line)

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('EEG Signal')

##############################
#
# DEBUG: Accuracy
#
##############################

with open(Path(__file__).parent / "test_dict.txt", 'r') as f:
    test_dict = {int(k): v for k, v in json.loads(f.read()).items()}
    samples = 0
    successes = 0
    categorical_samples = {k: 0 for k in range(len(states))}
    categorical_successes = {k: 0 for k in range(len(states))}
    for k, features in test_dict.items():
        samples += len(features)
        categorical_samples[k] = len(features)
        for v in features:
            label, hidden_states = ccn.process(v)
            if label==k and hidden_states[chunk_size-1]==k:
                successes += 1
                categorical_successes[k] += 1

    print(f"\nTotal Samples: {samples}")
    print(f"\nSuccessful Predictions: {successes}")
    print(f"\nAccuracy: {(successes/samples)*100}%")

    for k, v in categorical_samples.items():
        print(f"\n{k} Samples: {categorical_samples[k]}")
        print(f"\n{k} Successes: {categorical_successes[k]}")
        print(f"\n{k} Accuracy: {(categorical_successes[k]/categorical_samples[k])*100}%")

while True:
    chunk = get_complete_chunk(inlet, chunk_size)
    n_samples = chunk.shape[0]

    # Display data prototype
    data = np.roll(data, -n_samples, axis=0)
    data[-n_samples:,:] = chunk[-n_samples:,:]
    
    # Predict rolling wave
    label, hidden_states = ccn.process(chunk[:,0])
    # if(label==hidden_states[chunk_size-1]):
    #     inputs.wasd_map(label)

    if plot:
        for ch in range(num_channels):
            lines[ch].set_ydata(data[:,ch])
            axs[ch].relim()
            axs[ch].autoscale_view()
        
        plt.pause(0.1)