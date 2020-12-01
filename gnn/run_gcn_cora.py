#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import  Model
from gcn import GCN
from utils import preprocess_adj,plot_embeddings, load_data_v1

if __name__ == "__main__":

    FEATURE_LESS = False

    # 加载数据，划分train/val/test集，A:对称邻接矩阵(无向图)，features:数据feature
    # A: (node_num, node_num)
    # features shape: (node_num, feature_dim)
    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_v1(
        'cora')

    # 邻接矩阵归一化: D^(-1/2) * A * D^(-1/2)
    A = preprocess_adj(A)

    # 特征矩阵归一化: 每条数据特征和为1，类型从稀疏矩阵转为稠密矩阵
    features /= features.sum(axis=1, ).reshape(-1, 1)

    if FEATURE_LESS:
        # X.shape: (2708,)
        X = np.arange(A.shape[-1])
        # feature_dim = 2708
        feature_dim = A.shape[-1]
    else:
        # X.shape: (2708, 1433)
        X = features
        # feature_dim = 1433
        feature_dim = X.shape[-1]
    model_input = [X, A]

    # Compile model
    model = GCN(A.shape[-1], feature_dim, 16, y_train.shape[1],  dropout_rate=0.5, l2_reg=2.5e-4,
                feature_less=FEATURE_LESS, )
    model.compile(optimizer=Adam(0.01), loss='categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])

    NB_EPOCH = 200
    PATIENCE = 200  # early stopping patience

    val_data = (model_input, y_val, val_mask)
    mc_callback = ModelCheckpoint('./best_model.h5',
                                  monitor='val_weighted_categorical_crossentropy',
                                  save_best_only=True,
                                  save_weights_only=True)

    # train
    print("start training")
    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data,
              batch_size=A.shape[0], epochs=NB_EPOCH, shuffle=False, verbose=2, callbacks=[mc_callback])
    # test
    model.load_weights('./best_model.h5')
    eval_results = model.evaluate(
        model_input, y_test, sample_weight=test_mask, batch_size=A.shape[0])
    print('Done.\n'
          'Test loss: {}\n'
          'Test weighted_loss: {}\n'
          'Test accuracy: {}'.format(*eval_results))

    embedding_model = Model(model.input, outputs=Lambda(lambda x: model.layers[-1].output)(model.input))
    embedding_weights = embedding_model.predict(model_input, batch_size=A.shape[0])
    y  = np.genfromtxt("{}{}.content".format('../data/cora/', 'cora'), dtype=np.dtype(str))[:, -1]
    plot_embeddings(embedding_weights, np.arange(A.shape[0]), y)
