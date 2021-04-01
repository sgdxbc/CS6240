import tensorflow as tf
import numpy as np


def decode_audio(audio_binary):
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    assert sample_rate == 16000, f'sample_rate: {sample_rate}'
    return tf.squeeze(audio, axis=-1)


MIN_SAMPLE = 16000


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([MIN_SAMPLE] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def get_commands(data_dir):
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != "README.md"]
    return commands