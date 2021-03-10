import os
import tensorflow as tf


def automated_testing(path_to_models,path_to_data):

    """
        The automated testing takes as input the path to models stored as an .hdf5 file and also takes as input the
        path to data files. The data files are stored as wave sound with the associated .json file indicating the PCR
        result  of the individual.

    """

    os.chdir(path_to_models)
    # generates a list of models and  runs model  evaluation using multi-processing
    model_names = [doc for doc in os.listdir() if doc.endswith(".hdf5")]

    for model_name in model_names:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        # change directory to the data_path directory and run the same algorithm through the wave files as run
        # through main.py

        graph = tf.compat.v1.get_default_graph()

        with graph.as_default():
            model = load_model(model_name)

            # loop through the




