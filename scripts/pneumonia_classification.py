from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.optimizers import RMSprop,Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from keras.applications import MobileNetV2
from keras.layers import Input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tf_explain.core.grad_cam import GradCAM

batch_size = 12
num_classes = 3
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True #make fit false if you do not want to train the network again
train_dir = 'C:\\Users\\Bulal\\Desktop\\College\\Year 4\\Sem 2\\Computer Vision\\Assignment2\\data\\chest_xray\\train'
test_dir = 'C:\\Users\\Bulal\\Desktop\\College\\Year 4\\Sem 2\\Computer Vision\\Assignment2\\data\\chest_xray\\test'

with tf.device('/gpu:0'):
    
    #create training,validation and test datatsets
    train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    class_names = train_ds.class_names
    print('Class Names: ',class_names)
    num_classes = len(class_names)
    
    # compute class weights from training data
    y_train_labels = []

    for _, labels in train_ds:
        y_train_labels.extend(labels.numpy())

    y_train_labels = np.array(y_train_labels)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )

    class_weights = {i: float(w) for i, w in enumerate(class_weights_array)}
    print("Class weights:", class_weights)
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()
    
    # data augmentation to increase the size of the training dataset and 
    # reduce overfitting by introducing more variety in the training samples
    data_augmentation = tf.keras.Sequential([
        RandomRotation(0.05),
        RandomZoom(0.1),
        RandomTranslation(0.05, 0.05)
    ])

    #create model
    # model = tf.keras.models.Sequential([
    #     data_augmentation,
    #     Rescaling(1.0/255),
    #     Conv2D(16, (3,3), activation = 'relu', input_shape = (img_height,img_width, img_channels)),
    #     MaxPooling2D(2,2),
    #     Conv2D(32, (3,3), activation = 'relu'),
    #     MaxPooling2D(2,2),
    #     Conv2D(32, (3,3), activation = 'relu'),
    #     MaxPooling2D(2,2),
    #     GlobalAveragePooling2D(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.3),
    #     Dense(num_classes, activation = 'softmax')
    # ])
    
    # Transfer learning using MobileNetV2 as the base model 
    # and adding a custom classification head on top of it
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, img_channels),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = tf.keras.models.Sequential([
        Input(shape=(img_height, img_width, img_channels)),
        data_augmentation,
        Rescaling(1.0/255),
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia_transfer.keras",save_freq='epoch',save_best_only=True)

    if fit:
        history = model.fit(
            train_ds,
            batch_size=batch_size,
            validation_data=val_ds,
            callbacks=[save_callback, earlystop_callback],
            class_weight=class_weights,
            epochs=epochs)
    else:
        model = tf.keras.models.load_model("pneumonia_transfer.keras")

    #if shuffle=True when creating the dataset, samples will be chosen randomly   
    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])
    
    # Precision, Recall, F1-score
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


    if fit:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(),0))#perform a prediction on this image
            plt.title('Actual:' + class_names[labels[i].numpy()]+ '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.show()
    
    explainer = GradCAM()

    for images, labels in test_ds.take(1):
        image = images[0].numpy()
        true_label = int(labels[0].numpy())

        prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predicted_class = np.argmax(prediction)

        heatmap = explainer.explain(
            validation_data=(np.expand_dims(image, axis=0), np.array([predicted_class])),
            model=model,
            class_index=predicted_class
        )

        plt.figure(figsize=(6,6))
        plt.imshow(image.astype("uint8"))          # X-ray
        plt.imshow(heatmap, cmap='jet', alpha=0.5) # heatmap overlay
        plt.title(f"Grad-CAM\nActual: {class_names[true_label]} | Predicted: {class_names[predicted_class]}")
        plt.axis("off")
        plt.show()