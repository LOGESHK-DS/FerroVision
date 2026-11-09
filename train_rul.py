"""Training RNN & LSTM for RUL Prediction"""

from __future__ import annotations
import argparse, csv, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setting up the main project directories
ROOT = Path(__file__).resolve().parent
RAW = ROOT/'data'/'raw'; OUT=ROOT/'models'/'rul'
METRICS = ROOT/'outputs'/'metrics'

# Features used by the models (engine_id is only used for grouping)
FEATURES = ['cycle','op_setting_1','op_setting_2','op_setting_3'] + [f'sensor_{i}' for i in range(1,22) if i != 19]

def seq(frame, scaler, window):
    # Creating input sequences for RNN and LSTM models
    xs = []
    ys = []

    # Grouping data by engine so that sequences don't mix different engines
    for _, g in frame.sort_values(['engine_id','cycle']).groupby('engine_id'):

        x = scaler.transform(g[FEATURES]).astype('float32')
        y = g.RUL.to_numpy('float32')

        # Creating sliding windows of past cycles
        for end in range(window, len(g)+1):
            xs.append(x[end-window:end])

            # RUL value of the last cycle in the current window
            ys.append(y[end-1])

    return np.asarray(xs),np.asarray(ys)

def make(kind, window=30):

    # Building the ANN model using individual cycle readings
    if kind=='ann':
        m = tf.keras.Sequential([tf.keras.layers.Input((len(FEATURES),)),
                               tf.keras.layers.Dense(256,activation='relu'),
                               tf.keras.layers.Dropout(.2),
                               tf.keras.layers.Dense(128,activation='relu'),
                               tf.keras.layers.Dense(1)])

    # RNN model for learning patterns across multiple cycles    
    elif kind=='rnn':
        m = tf.keras.Sequential([tf.keras.layers.Input((window,len(FEATURES))),
                               tf.keras.layers.SimpleRNN(96),
                               tf.keras.layers.Dropout(.2),
                               tf.keras.layers.Dense(48,activation='relu'),
                               tf.keras.layers.Dense(1)])
        
    # LSTM model
    else:
        m = tf.keras.Sequential([tf.keras.layers.Input((window,len(FEATURES))),
                               tf.keras.layers.LSTM(96),
                               tf.keras.layers.Dropout(.2),
                               tf.keras.layers.Dense(48,activation='relu'),
                               tf.keras.layers.Dense(1)])

    # Compiling the model
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='mse',
              metrics=['mae'])
    return m

def main():

    # Taking model type and training settings from command-line arguments
    p = argparse.ArgumentParser()
    p.add_argument('--model',choices=['ann','rnn','lstm'],required=True)
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--window',type=int,default=30)
    a=p.parse_args()

    # Creating output folders if they don't already exist
    OUT.mkdir(parents=True,exist_ok=True)
    METRICS.mkdir(parents=True,exist_ok=True)

    tf.keras.utils.set_random_seed(42)

    full = pd.read_csv(RAW/'train_data.csv')

    # Splitting engines instead of individual rows to avoid data leakage
    engines=full.engine_id.unique()

    # Creating separate datasets using the selected engine IDs
    train_val_engines, test_engines = train_test_split(engines, test_size=.20, random_state=42)
    train_engines, valid_engines = train_test_split(train_val_engines, test_size=.20, random_state=42)
    train = full[full.engine_id.isin(train_engines)].copy()
    valid = full[full.engine_id.isin(valid_engines)].copy()
    test = full[full.engine_id.isin(test_engines)].copy()

    # Fitting the scaler only on training data to prevent data leakage
    scaler = StandardScaler().fit(train[FEATURES])

    # ANN takes individual cycle readings as input
    if a.model == 'ann':
        x=scaler.transform(train[FEATURES]).astype('float32')
        y=train.RUL.to_numpy('float32')

        xv=scaler.transform(valid[FEATURES]).astype('float32')
        yv=valid.RUL.to_numpy('float32')

        xt=scaler.transform(test[FEATURES]).astype('float32')
        yt=test.RUL.to_numpy('float32')

    # RNN and LSTM take sequences of multiple cycles as input
    else:
        x, y = seq(train, scaler, a.window)
        xv, yv = seq(valid, scaler, a.window)
        xt, yt = seq(test, scaler, a.window)

    # Path where the best performing model will be saved
    path = OUT/f'best_{a.model}_final.keras'

    # Building the selected model
    m = make(a.model,a.window)

    # Early stopping
    # ModelCheckpoint saves the model whenever a better validation result is achieved.
    cb=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(path,monitor='val_loss',save_best_only=True)]

    # Training the model and monitoring its performance on validation data
    m.fit(x,y,validation_data=(xv,yv),epochs=a.epochs,batch_size=256,callbacks=cb,verbose=2)

    # Loading the best saved model before evaluating it on test data
    m = tf.keras.models.load_model(path)

    # Predicting RUL values for the test dataset
    pred = m.predict(xt,batch_size=512,verbose=0).ravel()

    # Calculating the model performance metrics
    vals={'model':a.model.upper(),
          'window':a.window if a.model!='ann' else 1,
          'features':len(FEATURES),
          'MAE':mean_absolute_error(yt,pred),
          'MSE':mean_squared_error(yt,pred),
          'RMSE':mean_squared_error(yt,pred)**.5,
          'R2':r2_score(yt,pred)}

    # Saving evaluation results in a CSV file
    with open(METRICS/f'{a.model}_final_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=vals.keys())
        w.writeheader()
        w.writerow(vals)

    # Saving the scaler so the same preprocessing can be used during prediction
    with open(OUT/f'{a.model}_feature_scaler.pkl','wb') as f:
        pickle.dump(scaler,f)

    # Saving feature names to maintain the same input order during prediction
    with open(OUT/f'{a.model}_feature_columns.pkl','wb') as f:
        pickle.dump(FEATURES,f)

    # Displaying the final evaluation results    
    print(vals)

if __name__=='__main__': main()