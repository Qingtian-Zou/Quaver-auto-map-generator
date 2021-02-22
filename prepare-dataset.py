import tensorflow_io as tfio
import numpy as np
from sklearn.utils import shuffle
import argparse

WINDOW_LEN=100
SAMPLE_RATE=44100
FLAGS=None

def process_notes(lines,song_length):
    Y=np.zeros(shape=(song_length,5),dtype=np.int)
    i=0
    # k: status flag
    # k==0: Initial state
    # k==1: One new note found!
    # k==2: Lane found!
    # k==3: End time found! Assuming long note
    k=0
    while i<len(lines):
        if k==1 and "Lane: " in lines[i]:
            lane=int(lines[i].split(": ")[1])
            i+=1
            k=2
        elif k==2 and "EndTime: " in lines[i]:
            end_time=int(lines[i].split(": ")[1])
            i+=1
            k=3
        elif "StartTime:" in lines[i]:
            if k==2:
                Y[start_time,lane]=1
            elif k==3:
                Y[start_time:end_time,lane]=1
            start_time=int(lines[i].split(": ")[1])
            k=1
            i+=1
        else:
            i+=1
    for i in range(len(Y)):
        if all(Y[i]==np.array([0,0,0,0,0])):
            Y[i][0]=1
    return Y

def process_tmp_XY(X_tmp,Y_tmp,window,stride):
    X=[]
    Y=[]
    for i in range(window-1,X_tmp.shape[0]-1,stride):
        X.append(X_tmp[i+1-window:i+1])
        Y.append(Y_tmp[i])
    X=np.array(X)
    Y=np.array(Y)
    return shuffle(X,Y,random_state=1)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=str,
        help="path to the audio file"
    )
    parser.add_argument(
        "--map",
        type=str,
        help="path to the map file"
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
    print("audio process completed...")

    note_fi=open(FLAGS.map,'r')
    lines=note_fi.readlines()
    note_fi.close()
    Y_tmp=process_notes(lines,X_tmp.shape[0])
    print("notes process competed...")

    X,Y=process_tmp_XY(X_tmp,Y_tmp,WINDOW_LEN,1)
    print("data sample ready...")
    n_data = len(X)
    X_train, X_test = X[:n_data//5*4], X[n_data//5*4:]
    Y_train, Y_test = Y[:n_data//5*4], Y[n_data//5*4:]
    np.save("X_train.npy",X_train)
    np.save("X_test.npy",X_test)
    np.save("Y_train.npy",Y_train)
    np.save("Y_test.npy",Y_test)
