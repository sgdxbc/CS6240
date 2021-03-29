from pathlib import Path
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from utils import *

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir = Path() / "data" / "mini_speech_commands"
audio_path = data_dir / "right" / "0ab3b47d_nohash_0.wav"
adv_path = Path() / "right2left.wav"
mix_path = Path() / "right2left(mix).wav"
model_path = Path() / "classifier_model_42"
target = "left"
alpha = 0.01

commands = get_commands(data_dir)
target = target == commands

model = tf.keras.models.load_model(str(model_path))

audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(str(audio_path)))
audio = tf.squeeze(audio, -1)
adv = tf.Variable([1.0] * MIN_SAMPLE)


def pred(mix):
    spec = tf.signal.stft(mix, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    spec = tf.expand_dims(spec, -1)
    spec = tf.expand_dims(spec, 0)
    return model(spec, training=False)[0]


def loss_fn():
    dist = tf.keras.losses.mse(target, pred(audio + adv))
    norm = tf.keras.losses.mse(tf.zeros([1, MIN_SAMPLE]), adv)
    return dist + alpha * norm


def opt_loop(loss_fn, var):
    opt = tf.keras.optimizers.Adam()
    prev_loss = None
    for i in range(1000000):
        opt.minimize(loss_fn, [var])
        loss = loss_fn()
        if prev_loss is not None and tf.abs(loss - prev_loss) < 1e-9:
            break
        prev_loss = loss
        if i % 100 == 0:
            print(prev_loss)


opt_loop(loss_fn, adv)

print(pred(audio))
print(pred(audio + adv))

tf.io.write_file(str(adv_path), tf.audio.encode_wav(tf.expand_dims(adv, -1), sample_rate))
tf.io.write_file(str(mix_path), tf.audio.encode_wav(tf.expand_dims(adv + audio, -1), sample_rate))
