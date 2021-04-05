from pathlib import Path
import tensorflow as tf
from utils import *

# *** Configuration
# common
data_dir = Path() / "data" / "mini_speech_commands"
adv_path = Path() / "adv.wav"
model_path = Path() / "classifier_model_42"
alpha = 2.0
train_commands_per_class = 2
val_commands_per_class = 10
# flipping
adv_sample_number = 110 * (SAMPLE_RATE // 1000)
adv_delay_interval = 10 * (SAMPLE_RATE // 1000)
target_map = [("left", "right"), ("right", "left")]
# *** Configuration End

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

commands = get_commands(data_dir)
model = tf.keras.models.load_model(str(model_path))
mix_per_audio = (SAMPLE_RATE - adv_sample_number) // adv_delay_interval
audio_map = {
    origin: [
        decode_audio(tf.io.read_file(filename))
        for i, filename in enumerate(tf.io.gfile.glob(str(data_dir / origin) + "/*"))
        if i < train_commands_per_class + val_commands_per_class
    ]
    for origin, _ in target_map
}

adv = tf.Variable(tf.random.normal([adv_sample_number]))


def get_base(index_range, audio=True):
    return tf.stack(
        [
            audio_map[origin][index] if audio else commands == target
            for origin, target in target_map
            for index in index_range
            for _ in range(mix_per_audio)
        ]
    )


audio_base = tf.concat(
    [
        get_base(range(train_commands_per_class)),
        get_base(
            range(
                train_commands_per_class,
                train_commands_per_class + val_commands_per_class,
            )
        ),
    ],
    axis=0,
)
truth_base = tf.concat(
    [
        get_base(range(train_commands_per_class), False),
        get_base(
            range(
                train_commands_per_class,
                train_commands_per_class + val_commands_per_class,
            ),
            False,
        ),
    ],
    axis=0,
)


def get_mask(count):
    row = tf.concat([adv, tf.zeros([SAMPLE_RATE - adv_sample_number])], axis=0)
    rows = tf.stack(
        [
            tf.roll(row, shift=step * adv_delay_interval, axis=0)
            for step in range(mix_per_audio)
        ]
    )
    return tf.tile(rows, [count, 1])


def pred(mix):
    return model(tf.expand_dims(extract_features(mix), -1), training=False)


print(
    f"predict size: {len(target_map) * train_commands_per_class * mix_per_audio}(train) "
    + f"{len(target_map) * val_commands_per_class * mix_per_audio}(val)"
)


def x_y(training):
    sep = len(target_map) * train_commands_per_class * mix_per_audio
    if training:
        return (
            audio_base[:sep] + get_mask(len(target_map) * train_commands_per_class),
            truth_base[:sep],
        )
    else:
        return (
            audio_base[sep:] + get_mask(len(target_map) * val_commands_per_class),
            truth_base[sep:],
        )


def loss_fn(training=True):
    x, y = x_y(training)
    dist = tf.keras.losses.mse(y, pred(x))
    norm = tf.keras.losses.mse(tf.zeros([adv_sample_number]), adv)
    return tf.math.reduce_mean(dist) + alpha * norm


def accuracy(y, fx):
    truth, pred = tf.argmax(y, axis=1), tf.argmax(fx, axis=1)
    return len(truth[truth == pred]) / len(truth)


def opt_loop(loss_fn, var):
    opt = tf.keras.optimizers.Adam()
    prev_loss = float("inf")
    for i in range(1000000):
        opt.minimize(loss_fn, [var])
        loss = loss_fn()
        if tf.abs(loss - prev_loss) < 1e-9:
            break
        prev_loss = loss
        if i % 100 == 0:
            print(f"i = {i}, loss = {loss.numpy()}")
            norm = tf.keras.losses.mse(tf.zeros([adv_sample_number]), adv)
            print(f"norm = {norm.numpy()}")
            train_x, train_y = x_y(True)
            val_x, val_y = x_y(False)
            print(
                f"acc(train) = {accuracy(train_y, pred(train_x))}, acc(val) = {accuracy(val_y, pred(val_x))}"
            )


opt_loop(loss_fn, adv)

tf.io.write_file(
    str(adv_path), tf.audio.encode_wav(tf.expand_dims(adv, -1), SAMPLE_RATE)
)
