import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import argparse

FLAGS=None

if __name__=="__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)

    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--save_model",
        type=bool,
        default=True,
        help="save trained model?"
    )
    FLAGS,unparsed=parser.parse_known_args()

    X_train=np.load("X_train.npy")
    X_test=np.load("X_test.npy")
    Y_train=np.load("Y_train.npy")
    Y_test=np.load("Y_test.npy")

    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(X_train.shape[1:]))
    # model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Conv1D(400,kernel_size=3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200))
    model.add(tf.keras.layers.Dense(120))
    model.add(tf.keras.layers.Dense(40))
    model.add(tf.keras.layers.Dense(Y_train.shape[1],activation="sigmoid"))
    model.compile(optimizer="adam",loss="mean_squared_error",metrics="accuracy")

    print(model.summary())
    model.fit(X_train,Y_train,epochs=3,use_multiprocessing=True,batch_size=256)
    model.evaluate(X_train,Y_train)
    model.evaluate(X_test,Y_test)

    if FLAGS.save_model:
        model.save("CNN.h5")
