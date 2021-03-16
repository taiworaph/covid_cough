import os
import tensorflow as tf
from keras.models import load_model
import librosa
import librosa.display
import pylab
import os
import cv2
import numpy as np
import sys

def automated_testing(path_to_models,cut_off):

    """
        The automated testing takes as input the path to models stored as an .hdf5 file and also takes as input the
        path to data files. The data files are stored as wave sound with the associated .json file indicating the PCR
        result  of the individual.

    """
    path_to_data = '../data/data_positive_negative/'
    os.chdir(path_to_models)
    # generates a list of models and  runs model  evaluation using multi-processing
    model_names = [doc for doc in os.listdir() if doc.endswith(".hdf5")]

    model_results = {}
    for model_name in model_names:
        model_results[model_name] = {}
        all_positive = 0
        true_positive = 0
        false_positive = 0
        all_negative = 0
        false_negative = 0
        true_negative = 0

        # change directory to the data_path directory and run the same algorithm through the wave files as run
        # through main.py

        graph = tf.compat.v1.get_default_graph()

        with graph.as_default():
            model = load_model(model_name)

            # loop through the data files - they are constant on a path
            # positive data
            positive = path_to_data + 'positive/'
            files = [filename for filename in os.listdir(positive) if filename.endswith('.wav')]
            # iterate and model evaluation
            for file in files:
                all_positive += 1
                wave_path = positive + file
                wavedata = process(model_name,wave_path)
                results = model.predict(wavedata)
                if float(results[0][0]) > float(cut_off):
                    true_positive += 1
                else:
                    false_positive += 1


            negative = path_to_data + 'negative/'
            files = [filename for filename in os.listdir(negative) if filename.endswith('.wav')]

            # iterate and model evaluation
            for file in files:
                all_negative += 1
                wave_path = negative + file
                wavedata = process(model_name,wave_path)
                results = model.predict(wavedata)
                if float(results[0][0]) < float(cut_off):
                    true_negative += 1
                else:
                    false_negative += 1
        model_results[model_name]['sensitvity'] = true_positive / (true_positive + false_negative)
        model_results[model_name]['specificity'] = true_negative / (true_negative + false_positive)
        print('Here are the model performance for model: {} and result {}'.format(model_name, model_results[
            model_name]))
    return model_results


def process(model_name,wavepath):
    ''' Processes the waveform file and generates the vector to send to the model for inference '''
    audio, sr = librosa.load(wavepath)
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
    savepath = os.path.join('../data/data_positive_negative/', 'test_file' + '.png')
    pylab.savefig(savepath, bbox_inches=None, pad_inches=0)
    pylab.close()

    # reload the image
    img = cv2.imread(savepath)
    img_resize = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img_rescale = img_resize / 255.

    image_1 = np.expand_dims(img_rescale, axis=0)

    # print('This is the image shape after re-sampling .....{}'.format(image_1.shape))
    # additional user information passed in as one-hot encoded vector
    if model_name == '020--0.610--0.050.hdf5':
        sample = [1, 1]
        sample_2 = np.expand_dims(sample, axis=0)

        final = [mfccs_1, image_1, sample_2]
    else:
        final = [mfccs_1, image_1]
    return final

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('Please run with path to models and the cut_off value to use')
    else:
        path = sys.argv[1]
        cut_off= sys.argv[2]
        statistics = automated_testing(path,cut_off)

        print('These are the full statistics : {}'.format(statistics))
