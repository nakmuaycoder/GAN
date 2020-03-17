''' Attribute GAN '''


from datetime import datetime
from tensorflow.keras.models import Model
from GAN.GAN._GAN import GAN as _GAN
from GAN.GAN._GAN import Label_GAN as _Label_GAN
from GAN.Loss import mi_loss as _mi_loss
from tensorflow.keras.utils import Progbar
from tensorflow.keras.layers import Input
import numpy as np
import os


class INFOGAN(_Label_GAN,_GAN):
    def __init__(self,generator,discriminator,DiscrOptimizer,GanOptimizer,loss_weights):
        _Label_GAN.__init__(self,generator,discriminator)
        _GAN.__init__(self,generator,discriminator)
        
        for shp,sh in zip(discriminator.output_shape[1:],generator.input_shape[1:]):
            assert shp == sh,"Discriminator output label shape {} and Generator input label shape must be the same".format(shp,sh)
        
        self.generator = generator
        self.discriminator = discriminator
        
        loss=['binary_crossentropy', 'categorical_crossentropy']
        for _ in discriminator.output[2:]:
            loss.append(_mi_loss)        
        self.discriminator.compile( loss=loss, loss_weights=loss_weights, optimizer=DiscrOptimizer, metrics=['accuracy'])
        self.discriminator.trainable = False        
        self.adversial = Model(self.generator.input,self.discriminator(self.generator(self.generator.input)))
        self.adversial.compile( loss=loss,loss_weights=loss_weights,optimizer=DiscrOptimizer,metrics=['accuracy'])
    
    def _rdmGenInput3(self,outputNumber,batchSize=16):
        #disentangled
        sampleSize = self.generator.input_shape[outputNumber][1]
        return np.random.normal(scale=0.5, size=[batchSize, sampleSize])
    
    def rdmGenInput(self,batchSize=16):
        '''Return a batch of vector and label in a list'''
        lblSize = self.generator.input_shape[1][1]
        nbDsit = len(self.generator.input_shape)
        out = [self._rdmGenInput(batchSize),self._rdmGenInput2(batchSize)]
        for i in range(2,nbDsit):
            #Add disentangled input
            out.append(self._rdmGenInput3(batchSize,outputNumber=i))
        return out 
    
    def DisOutput(self,batchsize,lbl,X,true=1,false=None):
        out = [self._discOutput(batchsize,true,false),lbl]
        for f in X:
                out.append(f)            
        return out


    
    
class subStackedGAN(object):
    """ GAn used in stackedGan """
    def __init__(self,generator,discriminator,encoder,DiscrOptimizer,GanOptimizer,dis_loss_weights,adv_loss_weights,dis_loss,adv_loss):

        self.encoder = encoder
        self.discriminator = discriminator
        self.generator = generator
        
        self.discriminator.compile(loss=dis_loss,optimizer=DiscrOptimizer,metrics=['accuracy'],loss_weights=dis_loss_weights)
        self.discriminator.trainable = False
        self.encoder.trainable = False
        
        gen_outputs = self.generator(self.generator.inputs )
        adv_outputs = self.discriminator(gen_outputs) + [self.encoder(gen_outputs)]
        self.adversial = Model(self.generator.inputs,adv_outputs)
        self.adversial.compile(loss=adv_loss,optimizer=GanOptimizer,metrics=['accuracy'],loss_weights=adv_loss_weights)
    def _saveGan(self,path):
        """ Save the generator and the generator """
        if os.path.exists(path):
            i = 0
            path
            while os.path.exists( os.path.join( path, 'GAn{}'.format(i) ) ):
                i += 1         
            pth = os.path.join( path, 'GAn{}'.format(i) )
            os.mkdir( os.path.join( path, 'GAn{}'.format(i) ) )
            self.encoder.save( os.path.join(pth,"encoder.h5") )
            self.discriminator.save( os.path.join(pth,"discriminator.h5") )
            self.generator.save( os.path.join(pth,"generator.h5") )
    
    def trainAdv(self,x,y):
        """train on a single batch the adversial
        """
        mtr = self.adversial.train_on_batch(x,y)
        return mtr
    def trainDis(self,x,y):
        """train on a single batch the discriminator
        """
        mtr = self.discriminator.train_on_batch(x,y)
        return mtr
    
    def generatorOutput(self,x):
        """Generator outputs"""
        return self.generator.predict(x)
    
    def generatorInpt(self,batchsize,ninput):
        """Return a batch of generator input
        this method randomly return a batch of fake z vector
        ninput is the index of the input (input layer's list) """
        inpt_shp = self.generator.input_shape[ninput][1]        
        return np.random.normal(scale=0.5, size=[batchsize, inpt_shp])

    def encoderVect(self,x):
        """Return the value of the encoder for an input x"""
        return self.encoder.predict(x)
     
  
    

class StackedGAN(object):
    """ StackedGAN: GAN with an encoder that transform into latent vector an intput and use it to train GANs """
    def __init__(self,*args):
        """ arguments must be, a list of subStackedGAN object """              
        gen_shp = tuple(map(lambda s: (s[1],), args[-1].generator.input_shape))
        
        self.mdl = args
        self.evaluationInpt = None
        self.InitialEpoch = 0
        self.batchEval = None
        #create encoder
        encoder = self.mdl[0].encoder
        
        for mdl in self.mdl[1:]:
            encoder = mdl.encoder(encoder.outputs)
            
        self.encoder = Model(self.mdl[0].encoder.inputs,encoder,name="encoder")
        self.encoder.trainable = True
        self.encoder.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        generator = self.mdl[-1].generator
        inpt = [Input(shape=i) for i in gen_shp]
        generator = generator(inpt)

        for mdl in self.mdl[:-1][::-1]:
            #Create a full generator (only used for inference)
            shp = (mdl.generator.input_shape[0][1],)
            inpt.append(Input(shape=shp))
            generator = mdl.generator(   [inpt[-1],generator] )
        
        self.generator = Model(inpt, generator, name="generator"  )
        
    def generateLabel(self,batchsize):
        """ Generate a batch of fake label (last generator; second input)"""
        inpt_shp = self.mdl[-1].generator.input_shape[1][1]
        return np.eye(inpt_shp)[np.random.choice(inpt_shp,batchsize)].reshape(batchsize,inpt_shp) 

        
    def trainencoder(self,xtr,ytr,epochs,batchsize,validation_data=None):
        '''Train the encoder'''
        print("Encoder Trainig")
        self.encoder.fit(xtr,ytr,epochs=epochs,batch_size=batchsize,validation_data=validation_data)
       
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
    
    def generateBatchEval(self,batchsize=16):
        """ Generate evaluation batch """               
        inpt = self.generatorInputs(batchsize)
        while np.unique(np.argmax(inpt[1],axis=1)).shape[0] != inpt[1].shape[1]:
            inpt = self.generatorInputs(batchsize)       
        self.evaluationInpt = inpt
    
    
    def generatorInputs(self,batchsize):
        """ This function return a full generator input """        
        out = [self.mdl[-1].generatorInpt(batchsize=batchsize,ninput=0),self.generateLabel(batchsize=batchsize)]
        for mdl in self.mdl[:-1]:
            out += [mdl.generatorInpt(batchsize=batchsize,ninput=0)]        
        return out
    
    def GenerateOutputs(self,batchsize=None,xtest=None,returnArray=False,dataViewer=None,save=False,view=False,epoch=None):
        """ Generate an output and transform it using a dataViewer """        
        if xtest is None and not batchsize is None:
            #Case no input and batchsize
            xtest = self.generatorInputs(batchsize)
        
        if not xtest is None:
            prediction = self.generator.predict(xtest)
            if not dataViewer is None  :
                for mtr in prediction:
                    if view==True:
                        dataViewer.view(mtr=mtr)
                    if save == True:
                        dataViewer.save(mtr=mtr,epoch=epoch)
            if returnArray == True:
                return prediction        

    def saveGan(self,path):
        """ Save the GAN in order to load and train it again """
        if os.path.exists(path):
            pth = os.path.join(path,datetime.now().strftime("%Y%m%d_%H%M%S")+ "_epoch_" + str(self.InitialEpoch))
            os.mkdir( pth )
            for gan in self.mdl:
                gan._saveGan(pth)
            
            #save the evaluation batch
            if not self.evaluationInpt is None:
                i = 0
                for arr in self.evaluationInpt:
                    np.save( os.path.join(pth,"eval{}".format(i)),  arr)
                    i += 1
        else:
            print('{} is not a valid path!'.format(path))
    def exportGenerator(self,path):
        """ Once the generator is trained, save the full generator """   
        if os.path.exists(path):
            p = os.path.join(path,'fullGenerator.h5')
            if os.path.exists(p):
                i = 0
                while os.path.exists(p):
                    i += 1
                    p = os.path.exists(os.path.join(path,'fullGenerator_{}.h5'.format(i)))
            self.generator.save(p)
        else:
            print("{} is not a valid path".format(path))
        
    def train(self,x_train,epoch,batch_size=1024,evalStep=10,pathSave=None,dataViewer=None):
        """ train all the gans
        during each training step, the discriminators are successfully trained (started from the n's gan to the GAN0)
        then adversial are trained
        """       
        try:
            evalGen, evalSave = evalStep
        except:
            evalGen, evalSave = evalStep,evalStep
        
        bSize = batch_size//2     
        if isinstance(x_train, list) or isinstance(x_train, tuple) :
            trainingStep = x_train[0].shape[0]//bSize
        else:
            trainingStep = x_train.shape[0]//bSize
        #Generate a single batch that will be use for evaluation.
        #The generator will be evaluate on the same batch.
        if self.evaluationInpt is None:
            self.generateBatchEval()
        batchEval = self.evaluationInpt      
        
        if not dataViewer is None :
            self.GenerateOutputs(xtest=batchEval,returnArray=False,dataViewer=dataViewer,save=True,view=False,epoch=self.InitialEpoch)
        end = self.InitialEpoch + epoch 
        
        for epc in range(1+ self.InitialEpoch,self.InitialEpoch + epoch + 1):
            #loop through epoch           
            progress_bar = Progbar(target=trainingStep)
            print("Epoch {}/{}".format(epc,end))
            x_train = self._shuffleData(x_train)#Shuffle data            
            
            mtrDis = {}
            mtrAdv = {}
            for i,mdl in enumerate(self.mdl):
                #create the dict for the discriminator and generator
                mtrDis['Gan{}'.format(i)] = {mtr:0 for mtr in mdl.discriminator.metrics_names}
                mtrAdv['Gan{}'.format(i)] = {mtr:0 for mtr in mdl.adversial.metrics_names}
            
            
            for step in range(trainingStep):                
                realDigit = x_train[0][step*bSize:(step+1)*bSize]
                realLabel = x_train[1][step*bSize:(step+1)*bSize]
                
                realFeatures = [realDigit]               
                for mdl in self.mdl:
                    realFeatures.append( mdl.encoderVect( realFeatures[-1] ) )
                 
                realFeatures[-1] = realLabel# The replace the output of the last encoder by the real label; not the estimation

                #------------------ discriminator training ------------------           
                for mdl,realFeature in zip(self.mdl[::-1],realFeatures[:-1][::-1])  :
                    mtr = list()
                    #Loop through the GAN in reverse order
                    if mdl == self.mdl[-1] :
                        #Case of the last GAN
                        #fake input is a tuple (z,label)
                        fakeInput = (mdl.generatorInpt(batchsize=bSize,ninput=0),self.generateLabel(batchsize=bSize))                     
                    else:
                        #other case, 
                        fakeInput = (mdl.generatorInpt(batchsize=bSize,ninput=0),fakeFeature)
                    
                    
                    fakeFeature = mdl.generatorOutput(x=fakeInput)
                    Feature = np.concatenate((realFeature,fakeFeature))
                    
                    #the discriminator output is a tuple (trueFalse,z)                    
                    trueFalse = self._discOutput(batchsize=2*bSize,true=1,false=0)
                    z = np.concatenate((mdl.generatorInpt(batchsize=bSize,ninput=0),fakeInput[0]))
                    
                    mtr.append(mdl.trainDis(Feature,(trueFalse,z)))# Train the discriminator and store in a list the loss/ac               
                for i, m in enumerate(mtr[::-1]):
                    z = mtrDis['Gan{}'.format(i)]
                    for key,val in zip(z.keys(),m):
                        z[key] += val/trainingStep
                                
                #------------------ adversial training ------------------ 
                trueFalse = self._discOutput(batchsize=bSize,true=1,false=None)             
                for mdl,realFeature in zip(self.mdl[::-1] ,realFeatures[::-1]):
                    mtr = list()
                    fakeInput = [mdl.generatorInpt(batchsize=bSize,ninput=0),realFeature]
                                        
                    #Adversial outputs (true/false,,)
                    fakeOutput = [trueFalse] + fakeInput
                    
                    mtr.append(mdl.trainAdv(fakeInput,fakeOutput))# Train the generator and store in a list the loss/acc
                    
                for i, m in enumerate(mtr[::-1]):
                    z = mtrAdv['Gan{}'.format(i)]
                    for key,val in zip(z.keys(),m):
                        z[key] += val/trainingStep                
                                
                progress_bar.update(step + 1)
            
            print(mtrDis)
            print(mtrAdv)

            self.InitialEpoch += 1#Increment the epoch counter 
            if (epc + 1) % evalGen == 0 and not dataViewer is None :
                #Generate the output
                self.GenerateOutputs(xtest=batchEval,returnArray=False,dataViewer=dataViewer,save=True,view=False,epoch=epc)
            if not pathSave  is None and (epc + 1) % evalSave == 0 :
                    #Save the GAN
                    self.saveGan(pathSave)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
        






