import tensorflow as tf
import tensorflow_io as tfio
import argparse
from sklearn.utils import shuffle
import numpy as np

WINDOW_LEN=100
SAMPLE_RATE=44100
FLAGS=None

def process_tmp_XY(X_tmp,window,stride,Y_tmp=None):
    if Y_tmp:
        X=[]
        Y=[]
        for i in range(window-1,X_tmp.shape[0]-1,stride):
            X.append(X_tmp[i+1-window:i+1])
            Y.append(Y_tmp[i])
        X=np.array(X)
        Y=np.array(Y)
        return shuffle(X,Y,random_state=1)
    else:
        X=[]
        for i in range(window-1,X_tmp.shape[0]-1,stride):
            X.append(X_tmp[i+1-window:i+1])
        X=np.array(X)
        return X


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
        "--model",
        type=str,
        help="path to the model"
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="path to the audio file"
    )
    FLAGS,unparsed=parser.parse_known_args()

    audio=tfio.audio.AudioIOTensor(FLAGS.audio)
    audio_slice=audio[0:WINDOW_LEN]
    X_tmp=[]
    for i in range(int(len(audio)//SAMPLE_RATE*1000)):
        if (i+1)*SAMPLE_RATE/1000<len(audio):
            X_tmp.append(np.average(audio[i*SAMPLE_RATE//1000:(i+1)*SAMPLE_RATE//1000],axis=0))
        else:
            X_tmp.append(np.average(audio[i*SAMPLE_RATE//1000:],axis=0))
            break
    X_tmp=np.array(X_tmp)
    X=process_tmp_XY(X_tmp,WINDOW_LEN,1)
    print("audio process completed...")

    model=tf.keras.models.load_model(FLAGS.model)
    pred=model.predict(X)

    #TODO: convert predictions to notes
