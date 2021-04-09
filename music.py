from pathlib import Path
import tensorflow as tf
from utils import *


data_dir = Path() / "data" / "mini_speech_commands"
model_path = Path() / "classifier_model_42"
music_path = Path() / "underlay.wav"
perturbated_path = Path() / "pert.wav"
target_map = [("left", "right")]
train_count = 20
val_count = 10
alpha = 1.0
adv_chunk_length = 200 * (SAMPLE_RATE // 1000)
chunk_count = 3
adv_delay_interval = 10 * (SAMPLE_RATE // 1000)
delay_per_batch = 1
batch_per_epoch = 1
opt = tf.keras.optimizers.Adam()
stop_when_no_progress_in = 20


physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

commands = get_commands(data_dir)
model = tf.keras.models.load_model(str(model_path))
music, sample_rate = tf.audio.decode_wav(tf.io.read_file(str(music_path)))
print(music.shape)
assert sample_rate == SAMPLE_RATE
music = tf.cast(tf.squeeze(music, axis=-1), tf.float32)
music = music[: chunk_count * (SAMPLE_RATE + adv_chunk_length) + SAMPLE_RATE]
audio_map = {
    origin: [
        decode_audio(tf.io.read_file(filename))
        for i, filename in enumerate(tf.io.gfile.glob(str(data_dir / origin) + "/*"))
        if i < train_count + val_count
    ]
    for origin, _ in target_map
}

adv = tf.Variable(tf.random.normal([adv_chunk_length * chunk_count]))


def underlay_mask(chuck_index, delays, count):
    music_chunk = music[
        chuck_index
        * (SAMPLE_RATE + adv_chunk_length) : (chuck_index + 1)
        * (SAMPLE_RATE + adv_chunk_length)
        + SAMPLE_RATE
    ]
    adv_chunk = tf.concat(
        [
            tf.zeros([SAMPLE_RATE]),
            adv[chuck_index * adv_chunk_length : (chuck_index + 1) * adv_chunk_length],
            tf.zeros([SAMPLE_RATE]),
        ],
        axis=0,
    )
    whole_chunk = music_chunk + adv_chunk
    mask_group = tf.stack(
        [whole_chunk[delay : delay + SAMPLE_RATE] for delay in delays]
    )
    return tf.repeat(mask_group, repeats=count, axis=0)


xs, ys, val_xs, val_yi = [], [], [], []
for origin, target in target_map:
    for index, audio in enumerate(audio_map[origin]):
        if index < train_count:
            xs.append(audio)
            ys.append(commands == target)
        else:
            val_xs.append(audio)
            val_yi.append(tf.argmax(commands == target))

x_mat, y_mat = tf.stack(xs), tf.stack(ys)
yi = tf.argmax(y_mat, axis=1)
xx_mat = tf.tile(x_mat, [delay_per_batch, 1])
yy_mat = tf.tile(y_mat, [delay_per_batch, 1])
val_x_mat, val_yi = tf.stack(val_xs), tf.stack(val_yi)


def pred(x):
    return model(tf.expand_dims(extract_features(x), -1), training=False)


last_loss = tf.zeros([])


def loss_fn(delays):
    dist_mean = []
    for chunk_index in range(chunk_count):
        xx, yy = xx_mat + underlay_mask(chunk_index, delays, x_mat.shape[0]), yy_mat
        dist_mean.append(tf.math.reduce_mean(tf.keras.losses.mse(yy, pred(xx))))
    norm = tf.keras.losses.mse(tf.zeros([adv.shape[0]]), adv)
    global last_loss
    last_loss = tf.math.reduce_mean(dist_mean) + alpha * norm
    return last_loss


def accuracy(x_mat, yi):
    accuracy_list = []
    for delay in range(0, adv_chunk_length + SAMPLE_RATE, adv_delay_interval):
        if delay == 0:
            continue
        for chunk_index in range(chunk_count):
            p = tf.argmax(
                pred(
                    x_mat
                    + underlay_mask(chunk_index, tf.constant([delay]), x_mat.shape[0])
                )
            )
            accuracy_list.append(tf.math.count_nonzero(p == yi) / len(yi))
    return tf.math.reduce_mean(accuracy_list)


def opt_step():
    for _ in range(batch_per_epoch):
        delays = (
            tf.random.uniform(
                [delay_per_batch], maxval=SAMPLE_RATE + adv_chunk_length, dtype=tf.int32
            )
            // adv_delay_interval
        ) * adv_delay_interval
        opt.minimize(lambda: loss_fn(delays), [adv])


def opt_loop():
    losses = []
    epoch_count = 0
    while True:
        opt_step()
        losses.append(last_loss)
        print(f"epoch = {epoch_count}, loss = {last_loss}")
        print(f"norm = {tf.keras.losses.mse(tf.zeros([adv.shape[0]]), adv)}")
        print(
            f"train_acc = {accuracy(x_mat, yi)}, val_acc = {accuracy(val_x_mat, val_yi)}"
        )
        epoch_count += 1
        if (
            len(losses) >= stop_when_no_progress_in
            and tf.math.reduce_min(losses[-stop_when_no_progress_in:])
            == losses[-stop_when_no_progress_in]
        ):
            break


opt_loop()


result = music + tf.concat(
    [
        *[
            tf.concat(
                [
                    tf.zeros([SAMPLE_RATE]),
                    adv[i * adv_chunk_length : (i + 1) * adv_chunk_length],
                ],
                axis=0,
            )
            for i in range(chunk_count)
        ],
        tf.zeros([SAMPLE_RATE]),
    ],
    axis=0,
)
tf.io.write_file(
    str(perturbated_path), tf.audio.encode_wav(tf.expand_dims(result, -1), SAMPLE_RATE)
)
