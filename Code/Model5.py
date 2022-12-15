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

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.36,
        subset="training",
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.16,
        subset="validation",
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )
    # Janky way to split dataset
    testing_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )
    class_names = train_ds.class_names
    print(f"Class Names: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    checkpoint_path = 'C:\\Users\\grifk\\OneDrive\\Documents\\SchoolStuff\\CS5821\\SemesterProject\\Code\\CheckPoints\\Model5'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    num_classes = 6

    num_epochs = 5

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
        tf.keras.layers.Conv2D(32, 9, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 9, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 9, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 9, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[cp_callback]
    )

    model.save('Model5Save')
    model_json = model.to_json()
    with open("Model5Save.json", "w") as json_file:
        json_file.write(model_json)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(num_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Train and Val Accuracy of Model5 Model')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Train and Val Loss of Model5 Model')
    plt.show()

if __name__=='__main__':
    main()