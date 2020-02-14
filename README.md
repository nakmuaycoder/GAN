# Generative adversarial Network (GAN) [work in progress]
Generative adversarial Network personal library
The goal of this project is to create my own library to easily create and train Generative adversarial Network.


## Requirement: 
- tensorflow version 2.0 and above
- numpy
- datetime


## About GAN

https://en.wikipedia.org/wiki/Generative_adversarial_network

Gan implemented and tutorial here:<br>
https://github.com/nakmuayFarang/Testing-on-minst-MNIST/blob/master/Keras/GAN/README.md

## Implemented GAN 
- Deep Convolutional Generative Adversarial Network: DCGAN [mnist DCGAN](https://github.com/nakmuayFarang/Testing-on-minst-MNIST/blob/master/Keras/GAN/DCGAN.ipynb)
- Wasserstein GAN: WGAN [mnist WGAN](https://github.com/nakmuayFarang/Testing-on-minst-MNIST/blob/master/Keras/GAN/WGAN.ipynb)
- Conditional Generative Adversarial Network: cGAN [mnist cGAN](https://github.com/nakmuayFarang/Testing-on-minst-MNIST/blob/master/Keras/GAN/cGAN.ipynb)
- Least-squares Generative Adversarial Network: LSGAN: [mnist LSGAN](https://github.com/nakmuayFarang/Testing-on-minst-MNIST/blob/master/Keras/GAN/LSGAN.ipynb)
- Auxiliary classifier Generative Adversarial Network: ACGAN



## Gan instantiation
Create a GAN instance using 4 parametres:
- Generator (tensorflow.keras.models): From a random input will generate fake data
- Discriminator (tensorflow.keras.models): A binary classifier for fake/original data
- DiscrOptimizer (tensorflow.keras.optimizers): optimizer for the Discriminator
- GanOptimizer (tensorflow.keras.optimizers): optimizer for the Adversial model


```python
from gan2 import WCGAN, wasserstein_loss
gan = WCGAN(generator=generator,discriminator=discriminator,DiscrOptimizer=RMSprop(lr=5e-5),GanOptimizer=RMSprop(lr=5e-5))
```


## Gan Methods and Attributes


### Attributes
- **evaluationInpt (numpy.ndarray):** or a list of numpy.ndarray used for evaluation of the generator over the training epochs
- **generator (tensorflow.keras.models):** the generator model
- **discriminator (tensorflow.keras.models):** the discriminator model
- **adversial (tensorflow.keras.models):** discriminator(generator())
- **InitialEpoch (int):** Number of training step

...
### Methods
1) ```saveGan(path)```: Save the generator, evaluationInpt and the discriminator and create config.json
- **path (str):** directory store the gan


2)  ```loadGAN(configFile=configFile)``` : Load all the attributes of the gan
- **configFile (str):** path to configfile.json


3) ```rdmGenInput(batchSize)``` (numpy.ndarray): Create a sample of generator input
- **batchSize (int):** number of record to generate


4) ```generateBatchEval(batchSize)``` (numpy.ndarray or list of numpy.ndarray): Set evaluationInpt
- **batchSize (int):** number of record to generate

5) ```GenerateOutputs(xtest=None,batchSize=10,returnArray=True,viewFunction=None,ep=None)``` (numpy.ndarray): Use the generator to generate an output
- **xtest (numpy.ndarray):** A generator input; default None. If None the output is generate from random inputs 
- **batchSize (int):** if xtest is None; number of outputs to generate; default 10
- **returnArray (boolean):** Return the result of the generator as numpy.ndarray; default True
- **viewFunction (function):** Function used to transform the output of the generator in an other format (ex jpg, mp3...); default None
- **ep (int):** Current eppoch; default None

6) ```train(x_train,epoch,batch_size=1024,outputTr=None,evalStep=(10,10),pathSave=None,n_critic = 5,clip_value = 0.01)``` : train the gan
- **x_train (numpy.ndarray or iterable of numpy.ndarray):** training data
- **epoch (int):** Number of training epochs 
- **batch_size (int):** size of training batch; default 1024
- **outputTr (function):** Function used to transform the output of the generator in an other format (ex jpg, mp3...); default None
- **evalStep (evalGen,evalSave) (iterable or int):** Number of Epoch before evaluate the model on evaluationInpt (evalGen) and number of epoch before save the GAN (evalSave), if int evalStep=(evalStep,evalStep)  ; default (10,10)
- **pathSave (str):** path where the model are saved every evalStep epoch; default None, the gan is not saved
- **n_critic (int) (WGAN _only)_:** Number of discriminator training strep before training the generator;  default  5
- **clip_value (float) (WGAN _only)_:** Bound of discriminator weights, after each training step each weight are updated to be between clip_value bound; default 0.01


7) ```DisOutput(batchsize,true=1,false=None)``` : return a batch of size batchsize of discriminator outputs. If ```false=None``` this method return a full batch of true value. Otherwhile it return an array of 50% true value and 50% of fake value
- **batchsize (int):** size of batch
- **true (int):** The value returned by the discriminator when the input is real; default 1
- **false (int):** The value returned by the discriminator when the input is fake; default None


