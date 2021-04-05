import tensorflow as tf
import numpy as np


SAMPLE_RATE = 16000


def decode_audio(audio_binary):
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    assert sample_rate == SAMPLE_RATE, f"sample_rate: {sample_rate}"
    waveform = tf.squeeze(audio, axis=-1)
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([SAMPLE_RATE] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    return tf.concat([waveform, zero_padding], 0)


def get_spectrogram(waveform):
    return extract_features(waveform)


def extract_features(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


def get_commands(data_dir):
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != "README.md"]
    return commands