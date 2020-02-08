# GAN [work in progress]
Generative adversarial Network personal library
The goal of this project is to create my own library to easily create and train Generative adversarial Network.


## Requirement: 
- tensorflow version 2.0 and above
- numpy
- datetime




## Gan instantiation
Create a GAN instance using 4 parametres:
- Generator (tensorflow.keras.models): From a random input will generate fake data
- Discriminator (tensorflow.keras.models): A binary classifier for fake/original data
- DiscrOptimizer (tensorflow.keras.optimizers): optimizer for the Discriminator
- GanOptimizer (tensorflow.keras.optimizers): optimizer for the Adversial model


```python
from gan import WCGAN, wasserstein_loss
gan = WCGAN(generator=generator,discriminator=discriminator,DiscrOptimizer=RMSprop(lr=5e-5),GanOptimizer=RMSprop(lr=5e-5))
```


## Implemented GAN
- Deep Convolutional Generative Adversarial Network: DCGAN
- Wasserstein GAN: WCGAN
- Conditional Generative Adversarial Network: cGAN
- LSGAN

## Gan Methods and Attributes


### Attributes
- evaluationInpt (numpy.ndarray): or a list of numpy.ndarray used for evaluation of the generator over the training epochs
- generator (tensorflow.keras.models): the generator model
- discriminator (tensorflow.keras.models): the discriminator model
- adversial (tensorflow.keras.models): discriminator(generator())
- InitialEpoch (int): Number of training step

...
### Methods
1) saveGan(path): Save the generator, evaluationInpt and the discriminator and create config.json
- path (str): directory store the gan


2) loadGAN(configFile): Load all the attributes of the gan
- configFile (str): path to configfile.json


2) rdmGenInput(batchSize) (numpy.ndarray): Create a sample of generator input
- batchSize (int): number of record to generate


3) generateBatchEval(batchSize) (numpy.ndarray or list of numpy.ndarray): Set evaluationInpt
- batchSize (int): number of record to generate


4) GenerateOutputs(xtest=None,batchSize=10,returnArray=True,viewFunction=None,ep=None) (numpy.ndarray): Use the generator to generate an output
- xtest (numpy.ndarray): A generator input; default None. If None the output is generate from random inputs 
- batchSize (int): if xtest is None; number of outputs to generate; default 10
- returnArray (boolean): Return the result of the generator as numpy.ndarray; default True
- viewFunction (function): Function used to transform the output of the generator in an other format (ex jpg, mp3...); default  =None
- ep (int): Current eppoch; default None

5) train(x_train,epoch,batch_size=1024,outputTr=None,evalStep=10,pathSave=None): train the gan
- x_train (numpy.ndarray): training data
- epoch (int): Number of training epochs 
- batch_size (int): size of training batch; default 1024
- outputTr (function): Function used to transform the output of the generator in an other format (ex jpg, mp3...); default  =None
- evalStep (int): Number of Epoch before evaluate the model; default 10
- pathSave (str): path where the model are saved every evalStep epoch; default None, the gan is not saved



