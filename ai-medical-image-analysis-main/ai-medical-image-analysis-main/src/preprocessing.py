import tensorflow as tf

def load_data():
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        "data/chest_xray/train",
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    test_data = test_gen.flow_from_directory(
        "data/chest_xray/test",
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    return train_data, test_data