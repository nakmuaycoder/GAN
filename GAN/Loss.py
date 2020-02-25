

""" Special losses use for training GANS """


import tensorflow.keras.backend as K

def mi_loss(c, q_of_c_given_x):
    # mi_loss = -c * log(Q(c|x))
    return K.mean(-K.sum(K.log(q_of_c_given_x + K.epsilon()) * c, axis=1))

def wasserstein_loss(y_label, y_pred):
    return -K.mean(y_label * y_pred)