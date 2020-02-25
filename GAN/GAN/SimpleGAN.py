'''

Simple GAN, Single input GAN


'''

from GAN.GAN._GAN import GAN as _GAN
from GAN.GAN._GAN import WithoutLabel_GAN as _WithoutLabel_GAN
from GAN.Loss import wasserstein_loss as _wasserstein_loss
from tensorflow.keras.utils import Progbar
import numpy as np
      
            
            
class LSGAN(_WithoutLabel_GAN,_GAN):
    ''' LSGAN '''
    def __init__(self,generator,discriminator,DiscrOptimizer,GanOptimizer):
        _GAN.__init__(self,generator,discriminator)
        _WithoutLabel_GAN.__init__(self,generator,discriminator,DiscrOptimizer,GanOptimizer,'mse')

        
        
       
        
        
        
class DCGAN(_WithoutLabel_GAN,_GAN):
    ''' DCGAN '''
    def __init__(self,generator,discriminator,DiscrOptimizer,GanOptimizer):
        _GAN.__init__(self,generator,discriminator)
        _WithoutLabel_GAN.__init__(self,generator,discriminator,DiscrOptimizer,GanOptimizer,'binary_crossentropy')


class WGAN(_WithoutLabel_GAN):
    '''WGAN: '''
    def __init__(self,generator,discriminator,DiscrOptimizer,GanOptimizer):
        super().__init__(generator,discriminator,DiscrOptimizer,GanOptimizer,loss=_wasserstein_loss)
        
    
    def train(self,x_train,epoch,batch_size=1024,evalStep=(10,10), n_critic = 5,clip_value = 0.01,pathSave=None,dataViewer=None):
        '''train the gan: alternate between training discriminator then genereator
        outputTr is a facultative function that transform the output of the genereator in image music...
        evalStep, the batch will be evaluate every evalStep step
        n_critic: number of discriminator training step before training the generator
        clip_value'''       
        try:
            evalGen, evalSave = evalStep
        except:
            evalGen, evalSave = evalStep,evalStep
        
        #batch creation
        trainingBatch = x_train.shape[0]//batch_size #Number of batchs of data
        trainingStep = trainingBatch//n_critic# number of training step; 1 training step requiere n_critic batch of data
        if self.evaluationInpt is None:
            self.generateBatchEval()        
        batchEval = self.evaluationInpt
        
        for epc in range(1+ self.InitialEpoch,self.InitialEpoch + epoch+1):
            #loop through epoch
            progress_bar = Progbar(target=trainingStep)
            x_train = self._shuffleData(x_train)#Shuffle data
            print("Epoch {}/{}".format(epc,epoch))
            btch = 0
            label = self.DisOutput(batch_size,true=1.)# label for real; -label for fake
            for step in range(trainingStep):
                #During 1 step, the adversial is trained on 1 batch
                loss = 0
                acc = 0
                for _ in range(n_critic):
                    #Train n_critic times the discriminator
                    realInp = x_train[btch*batch_size:(btch+1)*batch_size]#Real inpt
                    fakeInp = self.GenerateOutputs(batchSize=batch_size,returnArray=True)#Fake input
                    
                    #Train discriminator on fake and real data
                    l, a = self.discriminator.train_on_batch(realInp, label)
                    loss += l/(2*n_critic)
                    acc += a/(2*n_critic)
                    l, a = self.discriminator.train_on_batch(fakeInp, -1* label )
                    loss += l/(2*n_critic)
                    acc += a/(2*n_critic)                                        
                    log = "Discrimiator: (loss,acc)=({},{})".format(loss,acc)#mean of acc & loss on  n_critic batch
                    btch += 1
                    
                    #Clip generator weights
                    for layer in self.discriminator.layers:
                        weights =  layer.get_weights()
                        weights = [np.clip(weight,-clip_value,clip_value) for weight in weights]
                        layer.set_weights(weights)
                
                #training of the generator
                loss,acc = self.adversial.train_on_batch(self.rdmGenInput(batch_size),label)
                log += "\n \t Adversial: (loss,acc)=({},{})".format(loss,acc)
                progress_bar.update(step + 1)
            print(log)
            
            self.InitialEpoch += 1
            if (epc + 1) % evalGen == 0 and not dataViewer is None :
                self.GenerateOutputs(xtest=batchEval,returnArray=False,dataViewer=dataViewer,save=True,View=False,epoch=epc)
            if not pathSave is None and (epc + 1) % evalSave == 0 :
                    self.saveGan(pathSave)   


