import os
from pathlib import Path

import tensorflow as tf

SAMPLING_RATE = 16000

# Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
# and save it to the 'Downloads' folder in your HOME directory
DATASET_ROOT = os.path.join(os.path.expanduser("~"), "Downloads/16000_pcm_speeches")

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)


def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print("Sampling rate for {} is incorrect. Ignoring it".format(path))
        return None


def get_list_of_noise_paths():
    """
    ## Noise preparation
    In this section:
    - We load all noise samples (which should have been resampled to 16000)
    - We split those noise samples to chuncks of 16000 samples which
    correspond to 1 second duration each
    """
    noise_paths = []
    for subdir in os.listdir(DATASET_NOISE_PATH):
        subdir_path = Path(DATASET_NOISE_PATH) / subdir
        if os.path.isdir(subdir_path):
            noise_paths += [
                os.path.join(subdir_path, filepath)
                for filepath in os.listdir(subdir_path)
                if filepath.endswith(".wav")
            ]
    print(
        "Found {} files belonging to {} directories".format(
            len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))
        )
    )
    return noise_paths


def resample_noise_samples():
    """
    Resample all noise samples to 16000 Hz
    """
    command = (
            "for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
                                                        "for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
                                                                                                     "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
                                                                                                     "$file | grep sample_rate | cut -f2 -d=`; "
                                                                                                     "if [ $sample_rate -ne 16000 ]; then "
                                                                                                     "ffmpeg -hide_banner -loglevel panic -y "
                                                                                                     "-i $file -ar 16000 temp.wav; "
                                                                                                     "mv temp.wav $file; "
                                                                                                     "fi; done; done"
    )
    os.system(command)


def get_noises(noise_paths):
    noises = []
    for path in noise_paths:
        sample = load_noise_sample(path)
        if sample:
            noises.extend(sample)
    noises = tf.stack(noises)

    print(
        "{} noise files were split into {} noise samples where each is {} sec. long".format(
            len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
        )
    )
    return noises


def add_noise(audio, noises=None, scale=0.5, dist="uniform"):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        if dist is "normal":
            tf_rnd = tf.random.normal(
                (tf.shape(audio)[0],), 1, noises.shape[0], dtype=tf.int32
            )
        else:
            tf_rnd = tf.random.uniform(
                (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
            )
        noise = tf.gather(noises, tf_rnd, axis=0)
        noise = tf.squeeze(noise)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        print(f"Shape of audio {audio.get_shape()}, "
              f"noise {noise.get_shape()}"
              f"prop {prop.get_shape()}")
        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio
