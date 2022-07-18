import cv2
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# import wandb
# from wandb.keras import WandbCallback
from ruamel.yaml import YAML
from dvclive.keras import DvcLiveCallback

# load params from params.yaml file
yaml = YAML(typ="safe")
with open("params.yaml") as f:
    params = yaml.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
**************************************************
Operating Instructions:
1 Configure weights_folder for saving model weights
2 Configure driver_file, dataset_folder for reading datasets and labels
3 Configure map for classification and label mapping
4 Configure classes for classification number
5 Configure driver_valid_list  driver_test_list for dataset split
6 Configure early stopping patience
7 Configure epochs
***************************************************
'''

if __name__ == '__main__':
    root = os.getcwd()

# __________________________Wandb Online Version Train Visualization_________________________

    # wandb.init(project="MobileNet_Classification")

    # config = wandb.config
    # config.learning_rate = 0.0015
    # config.batch_size = 64
    # config.epochs = 2
    # config.stopping_patience = 10
    # config.classes = 10
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = params['epochs']
    stopping_patience = params['stopping_patience']
    classes = params['classes']

# 1 Configure weights_folder for saving model weights

    weights_folder = 'weights'
    save_weights_path = os.path.join(root, weights_folder)
    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
    print('save model path:', save_weights_path)

    driver_file = 'driver_imgs_list.csv'
    dataset_folder = 'SF_dataset'

# 2 Configure driver_file, dataset_folder for reading datasets and labels

    driver_details = pd.read_csv(os.path.join(
        root, dataset_folder, driver_file), na_values='na')
    driver_details.set_index('img')
    print('driver list detail: ', driver_details.head(5))
    driver_list = list(driver_details['subject'].unique())
    print('driver list:', driver_list)
    print('total drivers:', len(driver_list))
    print(driver_details.groupby(by=['subject'])['img'].count())
    class_distribution = driver_details.groupby(by=['classname'])[
        'img'].count()
    img_quantity = list(class_distribution.values)
    print('img amount of every class: ', img_quantity)

    map = {0: 'safe drive', 1: 'text-right', 2: 'phone-talk-right', 3: 'text-left',
           4: 'phone-talk-left', 5: 'operate-radio', 6: 'drink', 7: 'reach-behind',
           8: 'hair&makeup', 9: 'talk-passenger'}
# 3 Configure map for classification and label mapping

    train_image = []
    # classes = config.classes
# 4 Configure classes for classification number

    for folder_index in range(classes):
        class_folder = 'c' + str(folder_index)
        print(f'now we are in the folder {class_folder}')

        imgs_folder_path = os.path.join(
            root, dataset_folder, 'train', class_folder)
        imgs = os.listdir(imgs_folder_path)

        for img_index in tqdm(range(len(imgs))):
            img_path = os.path.join(imgs_folder_path, imgs[img_index])
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (224, 224))
            img = np.repeat(img[..., np.newaxis], 3, -1)
            label = folder_index
            driver = driver_details[driver_details['img']
                                    == imgs[img_index]]['subject'].values[0]

            train_image.append([img, label, driver])

    print('total images:', len(train_image))
    save_img_name = map[train_image[-1][1]] + \
        '_driver' + train_image[-1][-1] + '.jpg'
    cv2.imwrite(save_img_name, train_image[-1][0])


# ______________________Splitting the train, valid and test dataset_________________________

    random.shuffle(train_image)
    driver_valid_list = {'p015', 'p022', 'p056'}
    driver_test_list = {'p050'}
# 5 Configure driver_valid_list  driver_test_list for dataset split

    X_train, y_train = [], []
    X_valid, y_valid = [], []
    X_test, y_test = [], []

    for image, label, driver in train_image:
        if driver in driver_test_list:
            X_test.append(image)
            y_test.append(label)
        elif driver in driver_valid_list:
            X_valid.append(image)
            y_valid.append(label)
        else:
            X_train.append(image)
            y_train.append(label)

    X_train = np.array(X_train).reshape(-1, 224, 224, 3)
    X_valid = np.array(X_valid).reshape(-1, 224, 224, 3)
    X_test_array = np.array(X_test).reshape(-1, 224, 224, 3)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_valid shape: {X_valid.shape}')
    print(f'X_test shape: {X_test_array.shape}')

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test_array = np.array(y_test)
    print(f'y_train shape: {y_train.shape}')
    print(f'y_valid shape: {y_valid.shape}')
    print(f'y_test shape: {y_test_array.shape}')

# ___________________________Build Model_____________________________________
    if False:
        model = tf.keras.models.load_model('weights_gray')
    else:
        base_model = MobileNet(input_shape=(224, 224, 3),
                               weights='imagenet', include_top=False)
        # imports the mobilenet model and discards the last 1000 neuron layer.
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        preds = Dense(classes, activation='softmax')(
            x)  # final layer with softmax activation
        model = tf.keras.Model(inputs=base_model.input, outputs=preds)
    # print(model.summary())

# ___________________________Model Train_______________________________________

    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(
        optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[acc])
    checkpointer = ModelCheckpoint(
        filepath=save_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystopper = EarlyStopping(
        monitor='val_loss', patience=stopping_patience, verbose=1, min_delta=0.001, mode='min')
# 6 Configure early stopping patience


#    datagen = ImageDataGenerator(
#        rotation_range=10,
#        horizontal_flip=True,
#    )
#    train_data_generator = datagen.flow(X_valid,y_valid, batch_size=config.batch_size)

    # Fits the model on batches with real-time data augmentation:
    # WandbCallback()
    mobilenet_history = model.fit(X_train, y_train, steps_per_epoch=len(X_train) / batch_size, callbacks=[checkpointer, earlystopper, DvcLiveCallback()],
                                  epochs=epochs, verbose=1, validation_data=(X_valid, y_valid))
# 7 Configure epochs

# __________________________Train Visualization_________________________

    # Can be replaced by online version

    plt.plot(mobilenet_history.history['loss'])
    plt.plot(mobilenet_history.history['val_loss'])
    plt.title('loss vs epochs')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('train_loss.jpg')

    plt.plot(mobilenet_history.history['sparse_categorical_accuracy'])
    plt.plot(mobilenet_history.history['val_sparse_categorical_accuracy'])
    plt.title('accuracy vs epochs')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.savefig(os.path.join("dvclive", "train_accuracy.jpg"))


# ____________________________ Evaluate on test dataset_______________________________
    loss, acc = model.evaluate(X_test_array, y_test_array)
    print(f'last epoch loss: {loss}')
    print(f'last epoch test accuracy:{acc:.2%}')

    model_load = tf.keras.models.load_model(save_weights_path)
    loss, acc = model_load.evaluate(X_test_array, y_test_array)
    print(f'best loss: {loss}')
    print(f'best test accuracy:{acc:.2%}')
