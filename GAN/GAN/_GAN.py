'''
Gan model; create easily a generative adversial network.
'''

import os
import tensorflow
assert tensorflow.__version__[0]=="2", "tensorflow version must be > 2.0"
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
from tensorflow.keras.utils import Progbar
import tensorflow.keras.backend as K
from  datetime import datetime
from json import dump, loads
from abc import ABC, abstractmethod




class _GAN(ABC):
    '''Class GAN, Parent class of all the other GAN
    - generator (tensorflow.keras.models.Model)
    - discriminator (tensorflow.keras.models.Model)'''
    def __init__(self,generator,discriminator):
        assert type(generator).__name__ == 'Model', 'generator must be a tensorflow.keras.Model object'
        assert type(discriminator).__name__ == 'Model', 'discriminator must be a tensorflow.keras.Model object'
        self.evaluationInpt = None
        self.generator = generator
        self.discriminator = discriminator
        self.InitialEpoch = 0
        super().__init__()
    def _rdmGenInput(self,batchsize=16):
        ''' Return a batch of generator input.
            - The result of this function and the first input of the generator have the same shape.
            - batchsize is the number of latent vectors returned.
            In case of multi inputs generator (ex conditionnal gan), the 2nd input must be the second input of the model,
            the first input must be the latent vector.
            The generation of the second must be done in rdmGenInput method of the child class
        '''
        if isinstance(self.generator.input_shape,list):
            latent_shape = list(self.generator.input_shape[0])
        else:
            latent_shape = list(self.generator.input_shape)
        latent_shape[0] = batchsize
        return np.random.uniform(-1.0, 1.0, size=latent_shape)
    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def rdmGenInput(self):
        pass
    @abstractmethod
    def generateBatchEval(self):
        pass
    @abstractmethod
    def DisOutput(self):
        pass
    
    def GenerateOutputs(self,xtest=None,batchSize=16,returnArray=True,dataViewer=None,save=False,View=True,epoch=None):
        '''Generator evaluate the model on a single batch of latent vector,
        - xtest (array or a tensor): input of the generator 
        If None, the generator is evaluated on a batch of random latent vector using rdmGenInput method.
        - returnArray (boolean): if True return the predicted values in an array
        - dta_Vwer (function): this function will be evaluate on the output of the generator
        '''       
        if xtest is None:
            #Case xtest is None: randomly create generator inputs
            xtest = self.rdmGenInput(batchSize)
        
        lbls = None
        if isinstance(xtest,list) or isinstance(xtest,tuple):
            #case generator with label inputs
            lbls = xtest[1]
            xtest2 = xtest
                
        z = self.generator.predict(xtest)#Generator output
        
        if not dataViewer is None  :
            for mtr in z:
                #loop through the generator output
                if View==True:
                    dataViewer.view(mtr=mtr)
                if save == True:                   
                    dataViewer.save(mtr=mtr,epoch=epoch)
        
        if returnArray == True:
            #return the array + parametres in a list [fakeMinst,lbl,X1,...,Xn]
            if isinstance(xtest,list):
                xtest[0] = z
                return xtest
            else:
                return z
                                             
    def _shuffleData(self,xtr):
        '''Shuffle the data before each epoch'''
        if isinstance(xtr,list):
            x = xtr[0]
        else:
            x = xtr
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)#Shuffle index
        if isinstance(xtr,list):
            return [xtr[0][idx],xtr[1][idx]]
        else:
            return xtr[idx]        
        
    def _discOutput(self,batchsize,true=1,false=None):
        '''return a ndarray with the shape of the first output of the discriminator; if false is None, return only true values '''
        y = np.ones([batchsize, 1])
        if not false is None:
            y[batchsize//2:, ] = false
        return y

    
    def saveGan(self,path):
        '''Save in path folder generator and discriminator'''
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = '/epoch{}_gan'.format(self.InitialEpoch)        
        conf = dict()
        
        if os.path.exists(path) and not os.path.exists( os.path.join(path,p)) :  
            os.mkdir(os.path.join(path,p))
            conf['DateTime'] = date
            conf['InitialEpoch'] = self.InitialEpoch
            self.generator.save(os.path.join(path,p,'generator.h5'))
            conf['generator'] = os.path.join(path,p,'generator.h5')
            self.discriminator.save(os.path.join(path,p, 'discriminator.h5'))
            conf['discriminator'] = os.path.join(path,p,'discriminator.h5')
                
                               
            if type(self.evaluationInpt).__name__ == 'list':
                np.save(os.path.join(path,p,'evalLbl.npy'),self.evaluationInpt[1])
                conf['evalLbl'] = os.path.join(path,p, 'evalLbl.npy')
                np.save(os.path.join(path,p,'evalVect.npy'),self.evaluationInpt[0])
                conf['evalVect'] = os.path.join(path,p,'evalVect.npy')
                
                if len(self.evaluationInpt)>3:
                    j = 1
                    for i in self.evaluationInpt[2:]:
                        conf['X{}'.format(i)] = os.path.join(path,p,'X{}.npy'.format(j))
                        np.save(os.path.join(path,p,'X{}.npy'.format(j)),i)
                        j += 1

            else:
                np.save(os.path.join(path,p,'eval.npy'),self.evaluationInpt)
                conf['eval'] = os.path.join(path,p,'eval.npy')
            with open(os.path.join(path,p,'config.json'),'w') as config:
                jsn = dump(conf,config)
        else:
            print('{} is not a valid path!'.format(path))
    def _loadGAN(self,conf):
        '''Load everything from config file (try to load the weights)'''        
        if 'generator' in conf.keys() and 'discriminator' in conf.keys() and 'evalLbl' in conf.keys() and 'evalVect' in conf.keys() and 'InitialEpoch'  in conf.keys():
            self.generator.load_weights( conf['generator']  )
            self.discriminator.load_weights( conf['discriminator'] )
            self.evaluationInpt = [np.load(conf['evalVect']),np.load(conf['evalLbl'])]
            self.InitialEpoch = conf['InitialEpoch']
        elif 'generator' in conf.keys() and 'discriminator' in conf.keys() and 'eval' in conf.keys() and 'InitialEpoch'  in conf.keys():
            self.generator.load_weights( conf['generator']  )
            self.discriminator.load_weights( conf['discriminator'] )
            self.evaluationInpt = np.load(conf['eval'])
            self.InitialEpoch = conf['InitialEpoch']
        else:
            print("Some Attributes are Missing")
            
        
    def loadGAN(self,configFile):
        ''' Load the model, or the weight in case of error'''
        if os.path.isfile(configFile):
            with open(configFile,'r') as config:
                conf = loads(config.read()) 
            try:
                self._loadGAN(conf)
            except:    
                print("Model can't be load; try to load weight")
            
        
            
            

class GAN(_GAN):
    '''Parents class with train method; all the differents gan except wgan are subclass of this GAN'''
    def __init__(self,generator,discriminator):
        super().__init__(generator,discriminator)                
    def train(self,x_train,epoch,batch_size=1024,evalStep=10,pathSave=None,dataViewer=None):
        '''train the gan: alternate between training discriminator then genereator
        outputTr is a facultative function that transform the output of the genereator in image music...
        evalStep, the batch will be evaluate every evalStep step'''
        #batch creation
        
        #get discriminator number of inputs
        if isinstance(self.discriminator.input_shape,list):
            disInpt = len(self.discriminator.input_shape)
        else:
            disInpt = 1
        
        try:
            evalGen, evalSave = evalStep
        except:
            evalGen, evalSave = evalStep,evalStep
        
        bSize = batch_size//2     
        if isinstance(x_train, list) :
            trainingStep = x_train[0].shape[0]//bSize
        else:
            trainingStep = x_train.shape[0]//bSize
        #Generate a single batch that will be use for evaluation.
        #The generator will be evaluate on the same batch.
        if self.evaluationInpt is None:
            self.generateBatchEval()
        batchEval = self.evaluationInpt      
        for epc in range(1+ self.InitialEpoch,self.InitialEpoch + epoch + 1):
            #loop through epoch           
            progress_bar = Progbar(target=trainingStep)
            print("Epoch {}/{}".format(epc,epoch))
            x_train = self._shuffleData(x_train)#Shuffle data
            for step in range(trainingStep):
                #Create Input array:                               
                xDis = self.GenerateOutputs(batchSize=2*bSize,returnArray=True,save=False,View=False)#mnist,lbl,X1,...,Xn
                xAdv = self.rdmGenInput(batchSize=bSize)#generator input, LatentVect, lbl,X1,...,Xn
                    
                if isinstance(x_train,list):
                    #Case Lael GAN
                    xDis[0][0:bSize] = x_train[0][step*bSize:(step+1)*bSize]#Replace bsize first fake mnist by original one
                    xDis[1][0:bSize] = x_train[1][step*bSize:(step+1)*bSize]#Replace bsize first fake labels by origibal one
                    XDis,XAdv = None,None
                    
                    if xDis[2:] != []:
                        #Test if X1,...,Xn
                        XDis = xDis[2:]
                        XAdv = xAdv[2:]                      
                    try:
                        lblDis = xDis[1]
                        lblAdv = xAdv[1]
                    except:
                        lblDis = None
                        lblAdv = None                         
                    xDis = xDis[:disInpt]#Remove unused input

                else:
                    xDis[0:bSize] = x_train[step*bSize:(step+1)*bSize]#Replace bsize first fake mnist by origibal one
                    XDis,XAdv = None, None
                    lblDis,lblAdv = None,None                
                
                yDis = self.DisOutput(bSize*2,true=1.,false=0.,X=XDis,lbl=lblDis)#Discriminator outputs, 1; lbl; X1,...,Xn 
                yAdv = self.DisOutput(bSize,true=1.,X=XAdv,lbl=lblAdv)

                #Train the discriminator
                mtr = self.discriminator.train_on_batch(x=xDis, y=yDis)
                log = "Discrimiator: "
                for a,b in zip(self.discriminator.metrics_names,mtr):
                    log += "({},{})".format(a,b)
                #Train generator
                #Fool the discriminator: train the generator with fake data; and train it to predict true data
                log += "\n \t Adversial: "
                mtr = self.adversial.train_on_batch(xAdv,yAdv)
                for a,b in zip(self.discriminator.metrics_names,mtr):
                    log += "({},{})".format(a,b)
                progress_bar.update(step + 1)
            print(log)
            self.InitialEpoch += 1
            if (epc + 1) % evalGen == 0 and not dataViewer is None :
                #Generate the output
                self.GenerateOutputs(xtest=batchEval,returnArray=False,dataViewer=dataViewer,save=True,View=False,epoch=epc)
            if not pathSave  is None and (epc + 1) % evalSave == 0 :
                    #Save the GAN
                    self.saveGan(pathSave)            
           
            
class WithoutLabel_GAN(_GAN):
    ''' This Class contain the different method for GAN without label; the adversial creation and all the compilation are done during during the init '''
    def __init__(self,generator,discriminator,DiscrOptimizer=Adam(1.5e-4,0.5),GanOptimizer=Adam(1.5e-4,0.5),loss='binary_crossentropy'):
        super().__init__(generator,discriminator)
        assert list(generator.output_shape) == list(discriminator.input_shape), 'generator output shape and discriminator input shape must be equal!'
        self.discriminator.compile(loss=loss,optimizer=DiscrOptimizer,metrics=['accuracy'])
        self.discriminator.trainable = False
        gen_inp = self.generator.input_shape[1:]#Shape of generator input (latent size)
        
        self.adversial = Model(self.generator.input, self.discriminator(self.generator( self.generator.input )) )
        self.adversial.compile(loss=loss,optimizer=GanOptimizer,metrics=['accuracy'])
    def rdmGenInput(self,batchSize=16):
        ''' return a batch of random input for the generator '''
        return self._rdmGenInput(batchSize)
    def DisOutput(self,batchsize,true=1,false=None,**args):
        '''Return a batch of discriminator output'''
        return self._discOutput(batchsize,true,false)
    def generateBatchEval(self,batchSize=16):
        '''Generate a batch of evaluation'''
        self.evaluationInpt = self.rdmGenInput(batchSize)
        
        
        
        
class Label_GAN(_GAN):
    ''' This Class contain the different method for GAN with label; the adversial creation and all the compilation are done during during the init '''
    def __init__(self,generator,discriminator):
        super().__init__(generator,discriminator)            
    
    def generateBatchEval(self,batchSize=16):
        '''Generate a batch of evaluation that contain all the labels'''
        z = self.rdmGenInput(batchSize)
        while np.unique(np.argmax(z[1],axis=1)).shape[0] != self.generator.input_shape[1][1]:
            z = self.rdmGenInput(batchSize)
        self.evaluationInpt = z
        
    def _rdmGenInput2(self,batchSize=16):
        """Generate a batch of label"""
        lblSize = self.generator.input_shape[1][1]
        z = np.eye(lblSize)[np.random.choice(lblSize,batchSize)].reshape(batchSize,lblSize)
        return z       
