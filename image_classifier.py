import keras
from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping
from keras import optimizers
import numpy as np
import cv2
import os
import sys

model_vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
preprocess_input_vgg16 = keras.applications.vgg16.preprocess_input

def prepared_image(img_array):
    img_array = cv2.resize(img_array, (150, 150))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def vgg16(img_array):
    tmp_img = prepared_image(img_array)
    return model_vgg16.predict(preprocess_input_vgg16(tmp_img))[0]

def bnf_list_from(dirname):
    ret = []
    all_num = len(os.listdir(dirname))
    for idx, path in enumerate(os.listdir(dirname)):
        fullpath = os.path.join(dirname, path)
        print("[%d/%d] %s" % (idx+1, all_num, fullpath))
        img = cv2.imread(fullpath)
        bnf = vgg16(img).flatten()
        ret.append(bnf)
    return ret

def train_fcn_model(train_data, train_labels):
    # モデルを構成
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=train_data.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
                  metrics=['accuracy'])
    model.summary()

    # 学習
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    history = model.fit(train_data,
                        train_labels,
                        nb_epoch=1000,
                        batch_size=32,
                        validation_split=0.25,
                        shuffle=True,
                        callbacks=[early_stopping])

    return model

def train(correct_dir, incorrect_dir, output_path):
    correct_bnf = bnf_list_from(correct_dir)
    incorrect_bnf = bnf_list_from(incorrect_dir)

    train_labels = np.array([0] * len(correct_bnf) + [1] * len(incorrect_bnf))
    train_data = []
    train_data.extend(correct_bnf)
    train_data.extend(incorrect_bnf)
    train_data = np.array(train_data)

    model = train_fcn_model(train_data, train_labels)
    model.save(output_path)

def validate(correct_dir, incorrect_dir, model_path):
    correct_bnf = bnf_list_from(correct_dir)
    incorrect_bnf = bnf_list_from(incorrect_dir)

    model = load_model(model_path)

    correct_count = 0
    all_count = len(correct_bnf) + len(incorrect_bnf)

    for bnf in correct_bnf:
        bnf = np.expand_dims(bnf, axis=0)
        predict_val = model.predict(bnf)[0][0]
        if predict_val < 0.5:
            correct_count += 1

    for bnf in incorrect_bnf:
        bnf = np.expand_dims(bnf, axis=0)
        predict_val = model.predict(bnf)[0][0]
        if predict_val >= 0.5:
            correct_count += 1
    
    return correct_count, all_count

if __name__ == "__main__":
    process = sys.argv[1]
    print("process", process)

    if process == "train":
        train('./dataset/correct', './dataset/incorrect', './dataset/model.h5')
    elif process == "validate":
        correct_count, all_count = validate('./dataset/correct', './dataset/incorrect', './dataset/model.h5')
        print("all", all_count)
        print("correct", correct_count)
        print("accuracy", correct_count / all_count)
