# Generative adversarial Network (GAN) [work in progress]
Generative adversarial Network personal library
The goal of this project is to create my own library to easily create and train Generative adversarial Network.


## Objective

This program is made for quickly instantiate train and save a Generative Adversial Network ([GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network)).<br>



## Requirement: 
- tensorflow version 2.0 and above
- numpy
- PIL


## Tutorials
Gan implementation and tutorial here:<br>
https://github.com/nakmuaycoder/Testing-on-minst-MNIST/blob/master/Keras/GAN/



## Implemented GAN 
- Deep Convolutional Generative Adversarial Network: DCGAN [mnist DCGAN](https://github.com/nakmuaycoder/Testing-on-minst-MNIST/blob/master/Keras/GAN/DCGAN-V2.ipynb)
- Wasserstein GAN: WGAN [mnist WGAN](https://github.com/nakmuaycoder/Testing-on-minst-MNIST/blob/master/Keras/GAN/WGAN-V2.ipynb)
- Conditional Generative Adversarial Network: cGAN [mnist cGAN](https://github.com/nakmuaycoder/Testing-on-minst-MNIST/blob/master/Keras/GAN/cGan-V2.ipynb)
- Least-squares Generative Adversarial Network LSGAN: [mnist LSGAN](https://github.com/nakmuaycoder/Testing-on-minst-MNIST/blob/master/Keras/GAN/LSGAN-V2.ipynb)
- Auxiliary classifier Generative Adversarial Network ACGAN: [mnist ACGAN](https://github.com/nakmuaycoder/Testing-on-minst-MNIST/blob/master/Keras/GAN/ACGAN-V2.ipynb)
- InfoGAN: [mnist InfoGAN](https://github.com/nakmuaycoder/Testing-on-minst-MNIST/blob/master/Keras/GAN/InfoGAN-V2.ipynb)

- StackedGAN


## Package architecture

### utils

This script contain different usefull objects for deal with the GAN:
- createGif: a function creating GIF showing the evolution of the generator output over the training epochs
- dataViewer: an object used to visualise and save as file a generator output.
- load_GAN: function loading a saved GAN object.

### Loss

Implementation of the custom loss function used for GAN's training.
- mi_loss
- wasserstein_loss

### GAN module
- SimpleGAN: Single input generator
- LabelGAN: Double
- AttributeGAN: Disentangled Representation GAN


## Gan instantiation
A GAN instance requiere 4 parametres:
- Generator (tensorflow.keras.models): From a random input will generate fake data
- Discriminator (tensorflow.keras.models): A binary classifier for fake/original data
- DiscrOptimizer (tensorflow.keras.optimizers): optimizer for the Discriminator
- GanOptimizer (tensorflow.keras.optimizers): optimizer for the Adversial model


```python
from GAN.GAN.SimpleGAN import DCGAN
gan = DCGAN(generator=generator,discriminator=discriminator,DiscrOptimizer=RMSprop(lr=2e-4, decay=6e-8),GanOptimizer=RMSprop(lr=1e-4, decay=3e-8))
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
```python
gan.saveGan(path)
```
Save the generator, evaluationInpt and the discriminator and create config.json
- **path (str):** directory store the gan


```python
gan.loadGAN(configFile=configFile)
```
Load all the attributes of the gan, and the weights of the generator and discriminator.
- **configFile (str):** path to configfile.json


```python
gan.rdmGenInput(batchSize)
``` 
(numpy.ndarray) Create a sample of generator input
- **batchSize (int):** number of record to generate


```python
gan.generateBatchEval(batchSize)
```
(numpy.ndarray or list of numpy.ndarray): Set evaluationInpt
- **batchSize (int):** number of record to generate default value 16

```python
gan.GenerateOutputs(xtest=None,batchSize=16,returnArray=True,dataViewer=None,save=False,View=True,epoch=None)
```
(numpy.ndarray): Return a generator output
- **xtest (numpy.ndarray or iterable of numpy.ndarray):** A generator input; default value None, the output is generate from random inputs 
- **batchSize (int):** if xtest is None; number of outputs to generate; default 10
- **returnArray (boolean):** Return the result of the generator as numpy.ndarray; default True
- **dataViewer (GAN.utils.dataViewer):** A dataViewer object used transform, visualize and save generator's outputs in an other format (ex jpg, mp3...); default None
- **epoch (int):** Current eppoch; default None
- **save (boolean):** Save the dataViewer outputs; Default False
- **View (boolean):** Display the dataViewer outputs; Default False


```python
gan.train(x_train,epoch,batch_size=1024,evalStep=10,pathSave=None,dataViewer=None)
```
train the gan
- **x_train (numpy.ndarray or iterable of numpy.ndarray):** training data
- **epoch (int):** Number of training epochs 
- **batch_size (int):** size of training batch; default 1024
- **evalStep (evalGen,evalSave) (iterable or int):** Number of Epoch before evaluate the model on evaluationInpt (evalGen) and number of epoch before save the GAN (evalSave), if int evalStep=(evalStep,evalStep)  ; default (10,10)
- **pathSave (str):** path where the model are saved every evalStep epoch; default None, the gan is not saved
- **n_critic (int) (WGAN _only)_:** Number of discriminator training strep before training the generator;  default  5
- **clip_value (float) (WGAN _only)_:** Bound of discriminator weights, after each training step each weight are updated to be between clip_value bound; default 0.01
- **dataViewer (GAN.utils.dataViewer):** A dataViewer object used transform, visualize and save generator's outputs in an other format (ex jpg, mp3...); default None


```python
gan.DisOutput(batchsize,lbl,true=1,false=None,**args)
```
return a batch of size batchsize of discriminator outputs. If ```false=None``` this method return a full batch of true value. Otherwhile it return an array of 50% true value and 50% of fake value
- **batchsize (int):** size of batch
- **true (int):** The value returned by the discriminator when the input is real; default 1
- **false (int):** The value returned by the discriminator when the input is fake; default None
- **args :** Other arguments (X vector(s) for GAN with Disentangled Representation)

