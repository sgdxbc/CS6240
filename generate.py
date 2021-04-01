from pathlib import Path
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from utils import *

# *** Configuration
# common
data_dir = Path() / "data" / "mini_speech_commands"
adv_path = Path() / "right2left.wav"
mix_name = "right2left(mix)"
model_path = Path() / "classifier_model_42"
alpha = 5.0
# flipping
adv_sample_number = 2000  # ~50ms
adv_delay_interval = 50  # ~1ms
target_map = [
    (data_dir / "right" / "0ab3b47d_nohash_0.wav", "left"),
    (data_dir / "left" / "0b09edd3_nohash_0.wav", "right"),
]
# *** Configuration End

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

commands = get_commands(data_dir)
model = tf.keras.models.load_model(str(model_path))
audio_target_map = [
    (decode_audio(tf.io.read_file(str(path))), commands == command)
    for path, command in target_map
]
mix_per_audio = (MIN_SAMPLE - adv_sample_number) // adv_delay_interval
audio_list, target_list = [], []
for audio, target in audio_target_map:
    for i in range(mix_per_audio):
        audio_list.append(audio)
        target_list.append(target)
audio_list, target_list = tf.stack(audio_list), tf.stack(target_list)

adv = tf.Variable(tf.random.normal([adv_sample_number]))


def pred(mix):
    spec = tf.signal.stft(mix, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    spec = tf.expand_dims(spec, -1)
    return model(spec, training=False)


def mix():
    padded = tf.concat(
        [
            adv,
            tf.zeros([MIN_SAMPLE - adv_sample_number]),
        ],
        0,
    )
    padded_list = tf.stack(
        [
            tf.roll(padded, adv_delay_interval * i, 0)
            for i in range(mix_per_audio)
            for _ in range(len(audio_target_map))
        ]
    )
    return audio_list + padded_list


def loss_fn():
    dist = tf.keras.losses.mse(target_list, pred(mix()))
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
print(pred(mix())[:10])

sample_rate = 16000
tf.io.write_file(
    str(adv_path), tf.audio.encode_wav(tf.expand_dims(adv, -1), sample_rate)
)
tf.io.write_file(
    str(Path() / (mix_name + "-1.wav")),
    tf.audio.encode_wav(tf.expand_dims(mix()[mix_per_audio // 2], -1), sample_rate),
)
tf.io.write_file(
    str(Path() / (mix_name + "-2.wav")),
    tf.audio.encode_wav(
        tf.expand_dims(mix()[mix_per_audio + mix_per_audio // 2], -1), sample_rate
    ),
)
