'''Module utils'''

import tensorflow.keras.backend as K
import os
#import gan2
from json import dump, loads


def load_GAN(configFile,GAN,DiscOptimize,GanOptimizer):
    ''''Load a GAN'''    
    if os.path.isfile(configFile):
        with open(configFile,'r') as config:
            conf = loads(config.read())              
        if 'generator' in conf.keys() and 'discriminator' in conf.keys() and 'evalLbl' in conf.keys()  and 'InitialEpoch'  in conf.keys():
            if 'evalVect' in conf.keys():
                generator = load_model( conf['generator']  )
                discriminator=load_model( conf['discriminator'] )
                evaluationInpt = [np.load(conf['evalVect']),np.load(conf['evalLbl'])]
                InitialEpoch = conf['InitialEpoch']       
            elif 'eval' in conf.keys():
                generator.load_weights( conf['generator']  )
                discriminator.load_weights( conf['discriminator'] )
                evaluationInpt = np.load(conf['eval'])
                InitialEpoch = conf['InitialEpoch']
            Gan = GAN(generator,discriminator,DiscOptimize,GanOptimizer)
            Gan.evaluationInpt = evaluationInpt
            Gan.InitialEpoch = InitialEpoch
            return Gan
    else:
        print("{} is not a valid path".format(configFile))

        
class dataViewer(object):
    ''' Class data viewer: this object visualize and save the predicted value '''
    def __init__(self,functionView,functionSave,path=None):
        self.functionView = functionView
        self.functionSave = functionSave
        if path is None or not os.path.isdir(path):
            self.path = os.path.expanduser('~')
        else :
            self.path = path  
            
    def view(self,mtr):
        return self.functionView(mtr)
    
    def save(self,mtr,epoch=None):
        self.makePath(epoch=epoch)
        i = 0
        while  os.path.exists( os.path.join(self.path,"Epoch_{}".format(epoch,i),"Epoch_{}_{}.jpg".format(epoch,i) )):
            i += 1         
        self.functionSave(mtr, os.path.join(self.path,"Epoch_{}".format(epoch,i),"Epoch_{}_{}.jpg".format(epoch,i) ))
    
    def makePath(self,epoch=None):
        path = os.path.join(self.path,"Epoch_{}".format(epoch))
        if not os.path.isdir(path):
            os.mkdir(path)
        
