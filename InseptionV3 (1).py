import os
from pathlib import Path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

SEED = 999
np.random.seed(SEED)

def load_paths():
    """
    Load file paths and class labels.

    Returns:
        Tuple: A tuple containing a list of file paths and a set of class labels.
    """
    base_path = Path('./dataset')
    images_pattern = str(base_path / '*' / '*.jpg')
    image_paths = [*glob(images_pattern)]
    image_paths = [p for p in image_paths]
    return image_paths

def load_images_and_labels(image_paths, target_size=(224, 224)):
    """
    Load images and corresponding labels from file paths.

    Args:
        image_paths (list): List of file paths.
        target_size (tuple): Target size for resizing images.

    Returns:
        Tuple: A tuple containing numpy arrays of images and labels.
    """
    images = []
    labels = []
    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        label = image_path.split(os.path.sep)[-2]
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data(X, y):
    """
    Preprocess the data by normalizing images and one-hot encoding labels.

    Args:
        X (numpy array): Array of images.
        y (numpy array): Array of labels.

    Returns:
        Tuple: A tuple containing preprocessed arrays of images and labels.
    """
    X = X.astype('float') / 255.0
    y = LabelBinarizer().fit_transform(y)
    return X, y

def build_model(input_shape=(224, 224, 3), num_classes=6):
    """
    Build an InceptionV3 model for image classification.

    Args:
        input_shape (tuple): Input shape of images.
        num_classes (int): Number of output classes.

    Returns:
        Sequential: Compiled Keras model.
    """
    base_model = InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
    
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=40, batch_size=64, early_stopping_patience=10):
    """
    Train the model on the given data.

    Args:
        model (Sequential): Compiled Keras model.
        X_train (numpy array): Training set images.
        y_train (numpy array): Training set labels.
        X_val (numpy array): Validation set images.
        y_val (numpy array): Validation set labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        early_stopping_patience (int): Patience for early stopping.

    Returns:
        Tuple: A tuple containing the trained model and training history.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print the test accuracy.

    Args:
        model (Sequential): Trained Keras model.
        X_test (numpy array): Test set images.
        y_test (numpy array): Test set labels.
    """
    result = model.evaluate(X_test, y_test)
    
    test_loss = result[0]
    test_accuracy = result[1]
    print(f'Test accuracy: {result[1]}')


    # Save accuracy to a text file
    with open("./Models/InceptionV3/InceptionV3_eval.txt", 'w') as file:
        file.write(f'Test loss: {test_loss}\n')
        file.write(f'Test accuracy: {test_accuracy}\n')

def plot_history(history, save_path=None):
    """
    Plot and optionally save the training history.

    Args:
        history (History): Training history from Keras.
        save_path (str): Optional path to save the plots.
    """
    print(history.history.keys())
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Inception V3 Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_path:
        plt.savefig(save_path + '_accuracy.png')
    else:
        plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Inception V3 Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path + '_loss.png')
    else:
        plt.show()

def main():
    """
    Main function for loading data, training the model, and evaluating the performance.
    """
    image_paths = load_paths()
    # print(len(image_paths), classes)

    X, y = load_images_and_labels(image_paths)
    X, y = preprocess_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = build_model()
    trained_model, history = train_model(model, X_train, y_train, X_test, y_test)

    evaluate_model(trained_model, X_test, y_test)
    plot_history(history, save_path='./Models/InceptionV3/InceptionV3')

    save_model(trained_model, './Models/InceptionV3/InceptionV3.h5')

if __name__ == "__main__":
    main()
