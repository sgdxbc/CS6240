from pathlib import Path
from collections import deque
import tensorflow as tf
from utils import *

# *** Configuration
data_dir = Path() / "data" / "mini_speech_commands"
model_path = Path() / "classifier_model_42"
target_map = [("left", "right"), ("up", "down")]
train_commands_per_class = 500
val_commands_per_class = 10
alpha = 1.0
adv_sample_number = 600 * (SAMPLE_RATE // 1000)
adv_delay_interval = 10 * (SAMPLE_RATE // 1000)
delay_sample_number_per_epoch = 50
adv_path = Path() / "adv.wav"
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

xs, ys, val_xs = [], [], []
val_yi = []
for origin, target in target_map:
    for index, audio in enumerate(audio_map[origin]):
        if index < train_commands_per_class:
            xs.append(audio)
            ys.append(commands == target)
        else:
            val_xs.append(audio)
            val_yi.append(tf.argmax(commands == target))

x_mat, y_mat = tf.stack(xs), tf.stack(ys)
val_x_mat = tf.stack(val_xs)
val_yi = tf.convert_to_tensor(val_yi)


delay_of_this_batch = 0


def mix(x_mat):
    padded = tf.roll(
        tf.concat([adv, tf.zeros([SAMPLE_RATE - adv_sample_number])], axis=0),
        shift=delay_of_this_batch,
        axis=0,
    )
    return x_mat + tf.repeat([padded], repeats=[x_mat.shape[0]], axis=0)


def pred(mix):
    return model(tf.expand_dims(extract_features(mix), -1), training=False)


def loss_fn():
    x, y = mix(x_mat), y_mat
    dist = tf.keras.losses.mse(y, pred(x))
    norm = tf.keras.losses.mse(tf.zeros([adv_sample_number]), adv)
    return tf.math.reduce_mean(dist) + alpha * norm


def accuracy(yi, x):
    truth, predicate = yi, tf.argmax(pred(x), axis=1)
    return len(truth[truth == predicate]) / len(truth)


opt = tf.keras.optimizers.Adam()


def opt_step():
    global delay_of_this_batch
    losses = []
    for _ in range(delay_sample_number_per_epoch):
        delay_of_this_batch = tf.random.uniform(
            [], maxval=SAMPLE_RATE - adv_sample_number, dtype=tf.int32
        )
        opt.minimize(loss_fn, [adv])
        losses.append(loss_fn())
    return tf.math.reduce_max(losses)


def opt_loop():
    train_yi = tf.argmax(y_mat, axis=1)
    prev_loss = deque()
    for i in range(1000000):
        loss = opt_step()
        prev_loss.append(float(loss.numpy()))
        if len(prev_loss) > 3:
            prev_loss.popleft()
            if tf.math.reduce_max(prev_loss) - tf.math.reduce_min(prev_loss) < 1e-5:
                break
        if i % 1 == 0:
            print(f"i = {i}, loss_max = {loss.numpy()}")
            norm = tf.keras.losses.mse(tf.zeros([adv_sample_number]), adv)
            print(f"norm = {norm.numpy()}")

            train_acc_list, val_acc_list = [], []
            for delay_step in range(mix_per_audio):
                global delay_of_this_batch
                delay_of_this_batch = delay_step * adv_delay_interval
                train_mixed, val_mixed = mix(x_mat), mix(val_x_mat)
                train_acc_list.append(accuracy(train_yi, train_mixed))
                val_acc_list.append(accuracy(val_yi, val_mixed))
            print(
                f"acc(train) = {tf.math.reduce_mean(train_acc_list)}, acc(val) = {tf.math.reduce_mean(val_acc_list)}"
            )


opt_loop()

tf.io.write_file(
    str(adv_path), tf.audio.encode_wav(tf.expand_dims(adv, -1), SAMPLE_RATE)
)
