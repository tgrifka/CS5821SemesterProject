import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.keras import layers
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential

def main():
    data_dir = "C:\\Users\\grifk\\OneDrive\\Documents\\SchoolStuff\\CS5821\\SemesterProject\\Code\\6Resized"
    data_dir = pathlib.Path(data_dir)
    print(f"Data Dir: {data_dir}")
    img_count = len(list(data_dir.glob('*/*.png')))
    print(f"Image Count: {img_count}")
    image_height = 464
    image_width = 616
    batch_size = 8

    overall_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )


    AUTOTUNE = tf.data.AUTOTUNE

    overall = overall_ds.cache().shuffle(buffer_size=3200, seed=123).prefetch(buffer_size=AUTOTUNE)

    num_classes = 6

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.load_weights('C:\\Users\\grifk\\OneDrive\\Documents\\SchoolStuff\\CS5821\\SemesterProject\\Code\\CheckPoints\\FinalModel4')

    model.summary()

    loss, acc = model.evaluate(overall, verbose=2)
    print("\nOverall Accuracy:", acc)
    print("\nOverall Loss: ", loss)



if __name__=='__main__':
    main()