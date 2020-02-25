from tensorflow.keras.models import Model
from GAN.GAN._GAN import GAN as _GAN
from GAN.GAN._GAN import Label_GAN as _Label_GAN
from GAN.Loss import mi_loss as _mi_loss
from tensorflow.keras.utils import Progbar
import numpy as np
      
class cGAN(_Label_GAN,_GAN):
    '''Conditional GAN'''
    def __init__(self,generator,discriminator,DiscrOptimizer,GanOptimizer):
        _Label_GAN.__init__(self,generator,discriminator)
        _GAN.__init__(self,generator,discriminator)  
        assert generator.output_shape == discriminator.input_shape[0], 'generator output shape {} and discriminator input shape {} must be equal!'.format(generator.output_shape,discriminator.input_shape[0])
        lblSize = list(generator.input_shape[1])[1:]        
        inptVect = self.generator.input[0]
        inptLbl = self.generator.input[1]
        self.discriminator.compile(loss='binary_crossentropy',optimizer=DiscrOptimizer,metrics=['accuracy'])
        self.discriminator.trainable = False            
        self.adversial = Model([inptVect,inptLbl], self.discriminator([self.generator([inptVect,inptLbl]), inptLbl] ),name='Adversial' )
        self.adversial.compile(loss='binary_crossentropy',optimizer=GanOptimizer,metrics=['accuracy'])                
        
    def DisOutput(self,batchsize,true=1,false=None,lbl=None,**args):
        return self._discOutput(batchsize,true,false)
        
    def rdmGenInput(self,batchSize=16):
        '''Return a batch of vector and label in a list'''
        lblSize = self.generator.input_shape[1][1]
        return [self._rdmGenInput(batchSize),self._rdmGenInput2(batchSize)]
        
        
class ACGAN(_Label_GAN,_GAN):
    def __init__(self,generator,discriminator,DiscrOptimizer,GanOptimizer):
        _Label_GAN.__init__(self,generator,discriminator)
        _GAN.__init__(self,generator,discriminator)
        
        assert discriminator.output_shape[1] == generator.input_shape[1],"Discriminator output label shape {} and Generator input label shape must be the same".format(discriminator.output_shape[1],generator.input_shape)        
        lblSize = list(generator.input_shape[1])[1:]
        rdmVectSize = list(generator.input_shape[0])[1:]
        inptVect =  self.generator.input[0]
        inptLbl = self.generator.input[1]
        self.discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],optimizer=DiscrOptimizer,metrics=['accuracy'])
        self.discriminator.trainable = False            
        self.adversial = Model([inptVect,inptLbl], self.discriminator(self.generator([inptVect,inptLbl])),name='Adversial' )
        self.adversial.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],optimizer=GanOptimizer,metrics=['accuracy'])
        
    def rdmGenInput(self,batchSize=16):
        '''Return a batch of vector and label in a list'''
        lblSize = self.generator.input_shape[1][1]
        return [self._rdmGenInput(batchSize),self._rdmGenInput2(batchSize)]
                
    def DisOutput(self,batchsize,lbl,true=1,false=None,**args):
        return [self._discOutput(batchsize,true,false),lbl]
