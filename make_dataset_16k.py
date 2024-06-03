import librosa
import os
import numpy as np
import noisereduce as nr
# from noisereduce.noisereducev1 import reduce_noise
from tqdm.auto import tqdm

root_dir = "D:/work/Nomo/data/audio_dataset-fixed/"

def save_STFT(file, name, activity):
    # read audio data
    audio_data, sample_rate = librosa.load(file, sr=16000)

    # noise reduction
    noisy_part = audio_data[0:25000]
    #     reduced_noise = nr.reduce_noise(audio_clip=audio_data, noise_clip=noisy_part, verbose=False)
    # reduced_noise = nr.reduce_noise(y=audio_data, y_noise=noisy_part, sr=sample_rate)
    reduced_noise = audio_data

    # trimming
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)

    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))

    # save features
    np.save("features_16k/" + name[:-4] + "_" + activity + ".npy", stft)


activities = ['Alarm', 'Alarm_clock', 'bathtub_filling_or_washing', 'Bell', 'Busy_signal', 'Car_alarm',
              'Cellphone_buzz_vibrating_alert',
              'Fill_with_liquid', 'Fire_alarm', 'audioset-smoke_alarm',
              'Shower', 'Siren', 'smoke_detector_smoke_alarm', 'Stream_river', 'Telephone_bell_ringing', 'toilet_flush',
              'Water', 'Water_tap_faucet', 'Waterfall',
              'dishes_pots_and_pans', 'microwave_oven', 'Inside large room or hall', 'Inside public space', 'Other sourceless', 'Bird']

if not os.path.exists("features_16k"):
    os.makedirs("features_16k")

for i, activity in enumerate(activities):
    innerDir = activity
    files = os.listdir(root_dir + innerDir)
    for file in tqdm(files, total=len(files), desc=f'Processing Activity {i + 1}'):
        if (file.endswith(".wav")):
            save_STFT(root_dir + innerDir + "/" + file, file, activity)