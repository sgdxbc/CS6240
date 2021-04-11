import tensorflow as tf
import numpy as np
from config import *

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)


def decode_audio(audio_binary, check=False):
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    if check:
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


commands = get_commands(data_dir)
model = tf.keras.models.load_model(str(model_path))
audio_map = {
    origin: [
        decode_audio(tf.io.read_file(filename))
        for i, filename in enumerate(tf.io.gfile.glob(str(data_dir / origin) + "/*"))
        if i < train_count + val_count
    ]
    for origin, _ in target_map
}

xs, ys, val_xs, val_yi = [], [], [], []
for origin, target in target_map:
    for index, audio in enumerate(audio_map[origin]):
        if index < train_count:
            xs.append(audio)
            ys.append(tf.cast(commands == target, dtype=tf.float32))
        else:
            val_xs.append(audio)
            val_yi.append(tf.argmax(commands == target))

x_mat, y_mat = tf.stack(xs), tf.stack(ys)
yi = tf.argmax(y_mat, axis=1)
xx_mat = tf.tile(x_mat, [batch_size, 1])
yy_mat = tf.tile(y_mat, [batch_size, 1])
val_x_mat, val_yi = tf.stack(val_xs), tf.stack(val_yi)


def pred(x):
    return model(tf.expand_dims(extract_features(x), -1), training=False)


last_loss = tf.zeros([])
