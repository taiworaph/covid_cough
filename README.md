# Detection of Asymptomatic / Early Stage COVID carriers using Artificial Intelligence

### Introduction and Motivation:
    The Corona virus or COVID-19 virus attacks the lining of the lungs resulting to reduced lung capacity
    and in severe cases complete and fatal failure of the lungs and death of the individual.
    
    The Cloud team in CS32S is trying to help speed up the detection of COVID using machine learning.
    Since the virus is mainly spread by [asymptomatic carriers] (https://abcnews.go.com/Health/video/us-hits-500k-covid-deaths-76052848) properly tracking anf diagnosing
    patients that they have early-stage or asymptomatic corona virus is key to preventing further spread of the disease.
    
    While thermometers and PCRs are great equipments for properly diagnosing corona virus; thermometers not sensitive enough to
    diagnosing actual fevers or minute lung inflammation, while PCRs, though are more accurate: They do take couple of hours
    to provide accurate result.
    
    The goal of this project is to provide a high-through put, scalable, fast, and accurate web-based platform
    to provide clear diagnosis of presence or absence of asymptomatic infection in an individual via the recording 
    of their cough wave pattern.
    

### Model Architecture:
    The model serving this initial iteration of covid detection is using a convolutional neural network algorithm
    for training. The model also accepts, as one-hot-encoding, additional information about user including the absence 
    or presence of other respiratory conditions, gender and other demographic data.
    
    Pre-processing of the cough waveform data is done with the librosa package before the mel spectogram is fed to the 
    the trained model for inference.


### Machine Learning System Architecture:
    The initial demo app was hosted on a medium tier EC2 inference platform hosted on Amazon. An Ubuntu instance works best
    with minimal need to update packages except for packages in the requirements.txt file.
    The service was then served to the public through the webpage [raphaelalabi.com](http://raphaelalabi.com).
    
    Inference on the initial iteration is super fast < 3 sec/cough sound. The web-page allows the uploading of only
    .wav cough file format for processinng.
    
    The next iteration of this Machine Learning System will attempt to solve the problem of horizontal scalability using 
    AWS lambda services.
    
    There is automated model evaluation script called automated_model_testing.py. This ensures that multiple models can be 
    uploaded and evaluated for latency and performance prior to being deployed to customers/client.
    
    
### Deployment and Using this code:
    This code was written and deployed to an Ubuntu environment on EC2. The deploy.sh file can be run and 
    used to make all necessary deployment of packages to the computer server instance.
    
    For installation please navigate to the  deployment folder.
    Please run source deploy.sh
    Take a cup of coffee and wait about 10 mins and it should finally deploy the main.py file and start serving
    

    

