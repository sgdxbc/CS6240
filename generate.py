from pathlib import Path
import tensorflow as tf
from tensorflow.python.ops.variables import trainable_variables
import tensorflow_probability as tfp
from utils import *

data_dir = Path() / "data" / "mini_speech_commands"
audio_path = data_dir / "right" / "0ab3b47d_nohash_0.wav"
adv_path = Path() / "right2left.wav"
mix_path = Path() / "right2left(mix).wav"
model_path = Path() / "classifier_model_42"
target = "left"
alpha = 1.0

commands = get_commands(data_dir)
target = target == commands

model = tf.keras.models.load_model(str(model_path))

audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(str(audio_path)))
audio = tf.squeeze(audio, -1)
spectrogram = get_spectrogram(audio)
spectrogram = tf.expand_dims(spectrogram, -1)
spectrogram = tf.expand_dims(spectrogram, 0)
adv = tf.Variable([0.] * MIN_SAMPLE)

def loss_fn():
    spec = tf.signal.stft(adv, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    spec = tf.expand_dims(spec, -1)
    spec = tf.expand_dims(spec, 0)
    mix = spectrogram + spec
    pred = model(mix, training=False)[0]
    pred_min, pred_max = tf.reduce_min(pred), tf.reduce_max(pred)
    pred = (pred - pred_min) / (pred_max - pred_min)
    dist = tf.norm(pred - target)
    norm = tf.norm(adv)
    return dist + alpha * norm

losses = tfp.math.minimize(loss_fn, num_steps=1, optimizer=tf.optimizers.Adam(), trainable_variables=[adv])
with tf.control_dependencies([losses]):
    opt_adv = tf.identity(adv)
# print(opt_adv[:100])
tf.io.write_file(str(adv_path), tf.audio.encode_wav(tf.expand_dims(adv, -1), sample_rate))
tf.io.write_file(str(mix_path), tf.audio.encode_wav(tf.expand_dims(adv + audio, -1), sample_rate))
