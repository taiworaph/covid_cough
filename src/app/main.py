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

app=flask.Flask(__name__,template_folder="jinja_templates")

@app.route('/upload', methods=['GET'])
def uploading():

    response = flask.Response(render_template('index.html'))

    return response

@app.route('/coughsound', methods=['POST'])
def upload():


    """ Takes in a waveform sound and outputs the model prediction value"""

    # process the waveform file
    waveform_file = request.files['audio']
    #waveform_file = request.files['waveform']
    audio,sr = librosa.load(waveform_file)
    mfccs = librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=39)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    mfccs_1 = np.expand_dims(mfccs_scaled,axis=0)

    print('This is the shape of the waveform file after resampling .... {}'.format(mfccs_1.shape))


    #additional MFCCs
    pylab.axis('off')
    pylab.axes([0.,0.,1.,1.],frameon=False,xticks=[],yticks=[])
    melspect = librosa.feature.melspectrogram(y=audio,sr=sr)
    s_db = librosa.power_to_db(melspect,ref=np.max) # convert to image + save + reload
    # save and reload spectrogram image files
    librosa.display.specshow(s_db)
    savepath= os.path.join('../data/','test_file'+'.png')
    pylab.savefig(savepath,bbox_inches=None,pad_inches=0)
    pylab.close()

    #reload the image
    img = cv2.imread(savepath)
    img_resize = cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
    img_rescale = img_resize/255.

    image_1 = np.expand_dims(img_rescale,axis=0)
    print('This is the image shape after re-sampling .....{}'.format(image_1.shape))



    #additional user information passed in as one-hot encoded vector
    sample= [1,1]
    sample_2 = np.expand_dims(sample,axis=0)
    print('This is the shape of sample 2 ...{}'.format(sample_2.shape))


    print('This is the type of file we have ....... {}'.format(type(waveform_file)))
    print('We were able to get the audio waveform ....... {}'.format(audio))
    print('This is the type of audio waveform .......... {}'.format(type(audio)))

    # create a default graph to use for prediction of the keras model
    graph=tf.compat.v1.get_default_graph()

    with graph.as_default():
        model = load_model('../data/020--0.610--0.050.hdf5')
        Xbatch_0 = [mfccs_1,image_1,sample_2]
        results = model.predict(Xbatch_0)

    print('\n')
    print('\n')

    print('This is our predicted result ...... {}'.format(results))
    probability = results[0][0]
    if probability < 0.3:
        # non-covid patient lets return page for  non-covid
        return render_template('negative_response.html',probability=probability)
    else:
        # likely a covid patient
        return render_template('positive_response.html',probability=probability)


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)