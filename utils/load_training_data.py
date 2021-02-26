import scipy.io as sio
import time

def load_train(train_dir,mat_file):
    print('Loading Training Data ...')
    time_load_data_begin = time.time()
    Training_labels = []
    
    data_name = [train_dir + mat_file][0]
    data = sio.loadmat(data_name)
    labels = data['labels']
    Training_labels = labels
    
    nrtrain = Training_labels.shape[0]
    time_load_data = time.time() - time_load_data_begin
    print('Loading Training Data use {0:<.4f} sec'.format(time_load_data))

    return [nrtrain, Training_labels]