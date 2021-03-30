from pathlib import Path
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from utils import *

# *** Configuration
data_dir = Path() / "data" / "mini_speech_commands"
audio_path = data_dir / "right" / "0ab3b47d_nohash_0.wav"
adv_path = Path() / "right2left.wav"
mix_path = Path() / "right2left(mix).wav"
model_path = Path() / "classifier_model_42"
target = "left"
alpha = 0.5
adv_sample_number = 2000  # ~50ms
adv_delay_interval = 50  # ~1ms
# *** Configuration End

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

commands = get_commands(data_dir)
target = target == commands
model = tf.keras.models.load_model(str(model_path))
audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(str(audio_path)))
audio = tf.squeeze(audio, -1)
adv = tf.Variable(tf.random.normal([adv_sample_number]))
mix_per_audio = (MIN_SAMPLE - adv_sample_number) // adv_delay_interval


def pred(mix):
    spec = tf.signal.stft(mix, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    spec = tf.expand_dims(spec, -1)
    return model(spec, training=False)


def mix(audio):
    padded = tf.concat(
        [
            adv,
            tf.zeros([MIN_SAMPLE - adv_sample_number]),
        ],
        0,
    )
    return tf.stack(
        [
            audio + tf.roll(padded, adv_delay_interval * i, 0)
            for i in range(mix_per_audio)
        ]
    )


def loss_fn():
    dist = tf.keras.losses.mse(target, pred(mix(audio)))
    norm = tf.keras.losses.mse(tf.zeros([adv_sample_number]), adv)
    return tf.math.reduce_mean(dist) + alpha * norm


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

print("previous")
print(pred(tf.expand_dims(audio, 0))[0])
print("after")
print(pred(mix(audio))[:10])

tf.io.write_file(
    str(adv_path), tf.audio.encode_wav(tf.expand_dims(adv, -1), sample_rate)
)
tf.io.write_file(
    str(mix_path),
    tf.audio.encode_wav(
        tf.expand_dims(mix(audio)[mix_per_audio // 2], -1), sample_rate
    ),
)
