import numpy as np
# import tensorflow_federated as tff
#emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def restore_data_all(tff_dataset,num_clients):
  dataset = [] 

  total_data_size = 0
  ####################
  cnt = 0
  for id in range (num_clients):
      client_data = tff_dataset.create_tf_dataset_for_client(
        tff_dataset.client_ids[id])
    
      for i in iter(client_data):
        cnt = cnt+1
  total_size = cnt
  x = np.ones((total_size,32,32))  
  y = np.zeros((total_size))
  cnt = 0
  for id in range (num_clients):
    client_data = tff_dataset.create_tf_dataset_for_client(
      tff_dataset.client_ids[id])
    
    for i in iter(client_data):
      x[cnt,2:30,2:30] = i['pixels'].numpy()
      y[cnt] = i['label'].numpy()
      cnt = cnt +1
  dataset.append([x,y])
  return dataset

def restore_data_all_exclude(tff_dataset,num_clients):
  dataset = [] 

  total_data_size = 0
  ####################
  
  for idd in range (num_clients):
    cnt = 0
    for id in range (num_clients):
      if idd != id:  
        client_data = tff_dataset.create_tf_dataset_for_client(
          tff_dataset.client_ids[id])
      
        for i in iter(client_data):
          cnt = cnt+1
    total_size = cnt
    x = np.ones((total_size,32,32))  
    y = np.zeros((total_size))
    cnt = 0
    for id in range (num_clients):
      if idd != id:
        client_data = tff_dataset.create_tf_dataset_for_client(
          tff_dataset.client_ids[id])
        
        for i in iter(client_data):
          x[cnt,2:30,2:30] = i['pixels'].numpy()
          y[cnt] = i['label'].numpy()
          cnt = cnt +1
    dataset.append([x,y])
  return dataset

import random
def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

def create_extreme_hetero_data(dataset_train,dataset_test,No_clients,n_class,n_train,n_test):
  dataset_train_par = init_list_of_objects(No_clients)
  dataset_test_par = init_list_of_objects(No_clients)
  for label in range(10):
    for sample in range(dataset_train[0][0].shape[0]):
      if dataset_train[0][1][sample] ==label:
        dataset_train_par[label].append(dataset_train[0][0][sample])

  for label in range(10):
    for sample in range(dataset_test[0][0].shape[0]):
      if dataset_test[0][1][sample] ==label:
        dataset_test_par[label].append(dataset_test[0][0][sample])

  for label in range(10):
    dataset_train_par[label] = np.asarray(dataset_train_par[label])

  for label in range(10):
    dataset_test_par[label] = np.asarray(dataset_test_par[label])

  data_train_new = init_list_of_objects(No_clients)
  label_train_new = init_list_of_objects(No_clients)

  data_test_new = init_list_of_objects(No_clients)
  label_test_new = init_list_of_objects(No_clients)
  for client in range(No_clients):
    
    xx = range(10)
    xx = sorted(xx, key = lambda x: random.random() )
    label_indx = xx[0:n_class]
    cnt = 0
    for indx in label_indx:
      if cnt ==0:
        data_train_new[client] = dataset_train_par[indx][np.random.randint(0,dataset_train_par[indx].shape[0],n_train)]
        label_train_new[client] = indx*np.ones((n_train))

        data_test_new[client] = dataset_test_par[indx][np.random.randint(0,dataset_test_par[indx].shape[0],n_test)]
        label_test_new[client] = indx*np.ones((n_test))
      else:
        data_train_new[client] = np.concatenate([data_train_new[client],dataset_train_par[indx][np.random.randint(0,dataset_train_par[indx].shape[0],n_train)]])
        label_train_new[client] = np.concatenate([label_train_new[client], indx*np.ones((n_train))])

        data_test_new[client] = np.concatenate([data_test_new[client],dataset_test_par[indx][np.random.randint(0,dataset_test_par[indx].shape[0],n_test)]])
        label_test_new[client] = np.concatenate([label_test_new[client], indx*np.ones((n_test))])
      cnt +=1
  return data_train_new, label_train_new, data_test_new, label_test_new


import numpy as np
import os
from scipy.io import savemat, loadmat
# import tensorflow_federated as tff
#emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()



def load_HARboX_data_v2(N_clients = 60,tr_ratio = 0.8,flag_store_data_index=0):
  root = os.getcwd()
  data_train = []
  label_train = []
  data_test = []
  label_test = []
  dataset_test = []
  classe_set = ['Call_train', 'Hop_train', 'Walk_train','Wave_train','typing_train']
  ID_list_list = []
  for client_id in range(10,N_clients+10):
    c = 0

    
    for class_id in classe_set:
      read_path = r'C:\NotWin\University\PhD\HAR\FL-Datasets-for-HAR-main\datasets\HARBox\large_scale_HARBox/'+str(client_id)+'/'+class_id+'.txt'
#       read_path = 'H:\Arash\HAR\large_scale_HARBox/'+str(client_id)+'/'+class_id+'.txt' ## This for office PC

      if os.path.exists(read_path):
       temp_original_data = np.loadtxt(read_path,delimiter=' ')[:,1:] # the first column seems to be not a feature set
      else:
        continue
      temp_coll = temp_original_data.reshape(-1, 900)
      count_img = temp_coll.shape[0]
      temp_label = c * np.ones(count_img)
      if c ==0:
        coll_class = temp_coll
        coll_label = temp_label
      else:
        coll_class = np.concatenate([coll_class,temp_coll])
        coll_label = np.concatenate([coll_label,temp_label])
      c += 1
    N_data = coll_class.shape[0]
    d_list = range(N_data)
    

    if flag_store_data_index==0:
      ID_list_list = loadmat(root+'/data_index.mat')
      ID_list_list = ID_list_list["index"]
      indx = ID_list_list[0][client_id-1][0].tolist()
    else:
      ID_list = sorted(d_list, key = lambda x: np.random.random() )
      ID_list_list.append(ID_list)
      indx = list(ID_list_list[client_id-1])
      
    
    coll_class = np.array(coll_class)
    coll_label = np.array(coll_label)
    data_train.append(coll_class[indx[:round(tr_ratio*N_data)],...])
    label_train.append(coll_label[indx[:round(tr_ratio*N_data)],...])
    data_test.append(coll_class[indx[round(tr_ratio*N_data):],...])
    label_test.append(coll_label[indx[round(tr_ratio*N_data):],...])
  if flag_store_data_index==1:
      savemat(root+'/saved_models/data_index.mat',{"index":np.asarray(ID_list_list)})
  ## gathering all test data for the generalization evaluation
  c = 0
  for i in range(N_clients):
    if c==0:
      data_all = data_test[i]
      label_all = label_test[i]
    else:
      data_all = np.concatenate([data_all, data_test[i]])
      label_all = np.concatenate([label_all, label_test[i]])
    c += 1
  dataset_test = [data_all , label_all]
  return data_train, label_train, data_test, label_test, dataset_test
