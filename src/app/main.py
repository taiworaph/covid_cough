print('***************************************************************************************************************')
print('***************************************************************************************************************')
print('***************************************************************************************************************')
print('###############################################################################################################')
print('___________#######   ########    ##      ##    ####    ######                ###   ######_____________________')
print('___________##        ##    ##     ##    ##      ##     ##   ##                ##   ##  ##_____________________')
print('___________##        ##    ##      ##  ##       ##     ##    ##    ######     ##   ######_____________________')
print('___________##        ##    ##       ####        ##     ##     ##              ##       ##_____________________')
print('___________#######   ########        ##        ####    ##########            ####  ######_____________________')
print('###############################################################################################################')
print('***************************************************************************************************************')
print('***************************************************************************************************************')
print('***************************************************************************************************************')


import os
import flask
from flask import jsonify,render_template,request,send_file,redirect,url_for
from flask_cors import CORS
import keras
from keras.models import load_model
import tensorflow as tf
import librosa
import librosa.display
import pylab
import os
import cv2
import numpy as np
import logging
import copy
import datetime

import pickle
from scipy.io import wavfile
import sys
from coughvid.DSP import classify_cough

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('file_error.log')
file_handler.setLevel(logging.ERROR)

app=flask.Flask(__name__,template_folder="jinja_templates")

@app.route('/upload', methods=['GET'])
def uploading():

    response = flask.Response(render_template('upload.html'))
    return response



@app.route('/coughsound', methods=['POST'])
def upload():


    """ Takes in a waveform sound and outputs the model prediction value """
    # process the waveform file
    # Put this in a try-except block to catch errors and if an error occurs just route to an error HTML page

    try:
        waveform_file = request.files['audio']
        batch_0 = preprocessing(waveform_file)

    except Exception as e:
        print('A wrong file format or a cough sound wave that was too long was sent to the algorithm')
        return render_template('upload.html',message=f'Error: {e}')

    # create a default graph to use for prediction of the keras model
    graph=tf.compat.v1.get_default_graph()

    with graph.as_default():
        model = load_model('../data/020--0.610--0.050.hdf5')
        results = model.predict(batch_0)

    print('\n')
    print('\n')

    print('This is our predicted result ...... {}'.format(results))
    probability = results[0][0]
    if probability < 0.0001:
        return render_template('negative_response.html',probability='less than 0.01')
    elif probability < 0.2:
        # non-covid patient lets return page for  non-covid
        return render_template('negative_response.html',probability=str(round(probability*100, 2)))
    else:
        # likely a covid patient
        return render_template('positive_response.html',probability=str(round(probability*100, 2)))




def preprocessing(waveform_file):

    ''' Takes in a wave file and returns the mel frequency and the mel spectogram '''
    audio, sr = librosa.load(waveform_file)
    waveform_file.stream.seek(0)
    
    ## Validate Cough
    sr1, audio1 = wavfile.read(waveform_file)
    prob_cough = validate_cough(audio1, sr1)

    if prob_cough < 0.5:
        raise ValueError("""The audio was not detected to be a cough. Please
                          use a different file or re-record the cough.  
                          Try to remove background noise if possible.""")

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=39)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    mfccs_1 = np.expand_dims(mfccs_scaled, axis=0)

    # additional MFCCs
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    melspect = librosa.feature.melspectrogram(y=audio, sr=sr)
    s_db = librosa.power_to_db(melspect, ref=np.max)  # convert to image + save + reload
    # save and reload spectrogram image files
    librosa.display.specshow(s_db)
    savepath = os.path.join('../data/', 'test_file' + '.png')
    pylab.savefig(savepath, bbox_inches=None, pad_inches=0)
    pylab.close()

    # reload the image
    img = cv2.imread(savepath)
    img_resize = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
    img_rescale = img_resize / 255.

    image_1 = np.expand_dims(img_rescale, axis=0)
    print('This is the image shape after re-sampling .....{}'.format(image_1.shape))

    # additional user information passed in as one-hot encoded vector
    sample = [1, 1]
    sample_2 = np.expand_dims(sample, axis=0)

    final = [mfccs_1,image_1,sample_2]
    return final

def validate_cough(audio, sr):
    model_dir = '../data'
    with open (f'{model_dir}/cough_classifier', 'rb') as pickle_f:
        detection_model = pickle.load(pickle_f)
    with open (f'{model_dir}/cough_classification_scaler', 'rb') as pickle_f2:
        detection_scaler = pickle.load(pickle_f2)
    prob_cough = classify_cough(audio, sr, detection_model, detection_scaler)
    print(f"The file has a {round(prob_cough*100, 2)}% probability of being a cough.")
    return prob_cough


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)