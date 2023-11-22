import argparse
# import tensorflow_federated as tff
import sys
import os
from IPython import display
import random
root = os.getcwd()
sys.path.append(root+'/utils')
sys.path.append(root+'/network')
from scipy.io import savemat
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import graph_tools as Gr
from eval import evaluate_manual_data, evaluate
from net import ANN
from dataset import restore_data_all, create_extreme_hetero_data
from dataset import load_HARboX_data_v2
# def main():
  ## Credit: ARASH RASTI MEYMANDI
    #################
    #   # ######    #
    #   #      #    #
    #   #    #      #
    #   #  #        #
    #   #  #        #
    #   #    #      #
    #   #      #    #
    #               #
    #################
if True:
  parser = argparse.ArgumentParser(description='Parameter Processing')
  parser.add_argument('--model', type=str, default='ANN', help='model') ## the only current model is ANN
  parser.add_argument('--batch_size', type=int, default=128, help='batch size for local training')
  parser.add_argument('--E', type=int, default=4, help='Local training rounds')
  parser.add_argument('--R', type=int, default=120, help='Total number of communication rounds in the framework')
  parser.add_argument('--N_clients', type=int, default=10, help='number of clients in the framework') ## for the paper graph this must be set to 20
  parser.add_argument('--save_path', type=str, default='report_snr-5-nr-100-alpha-5_testrun', help='path to save results')
  parser.add_argument('--SNR', type=int, default=5, help='The SNR')
  parser.add_argument('--N_r', type=int, default=100, help='Number of receiver')
  parser.add_argument('--alpha', type=float, default=0.5, help='the personalization degree') ## Personalization parameter.

  ##################################################
  parser.add_argument("-f", "--file", required=False) # this particular code is essential for the parser to work in google colab
  ##################################################
  args = parser.parse_args()
  if not os.path.exists(args.save_path):
      os.mkdir(args.save_path)

  N_clients = args.N_clients

  data_train_new, label_train_new, data_test_new, label_test_new, dataset_test = load_HARboX_data_v2(N_clients
                                                                                                       ,flag_store_data_index=0) 


  ## preparing the settings
  e_max = args.R
  accuracy_FedAvg_general_true = np.zeros((e_max))
  accuracy_FedAvg_local_true = np.zeros((e_max))

  accuracy_FedAvg_general_QR = np.zeros((e_max))
  accuracy_FedAvg_local_QR = np.zeros((e_max))

  accuracy_FedAvg_general_conv = np.zeros((e_max))
  accuracy_FedAvg_local_conv = np.zeros((e_max))

  accuracy_FedAvg_general_QR_pers = np.zeros((e_max))
  accuracy_FedAvg_local_QR_pers = np.zeros((e_max))

  accuracy_Fedfilt_general = [np.zeros((e_max)),np.zeros((e_max)),np.zeros((e_max)),np.zeros((e_max))]
  accuracy_Fedfilt_local = [np.zeros((e_max)),np.zeros((e_max)),np.zeros((e_max)),np.zeros((e_max))]
  rounds = e_max
  E =args.E # local rounds
  N_active_clients = N_clients
  N_total_clients = N_clients
  Batch_size = args.batch_size
  G_list = [[],[],[],[],[]] #[id, model data1, model data2, etc]
  client_list = range(N_total_clients)
  client_list = sorted(client_list, key = lambda x: random.random() )

  ## Building the models
  model_FedAvg  = ANN()
  model_Fedfilt = ANN()
  optimizer_clients = tf.keras.optimizers.legacy.SGD(learning_rate=0.02)
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model_FedAvg.compile(loss=loss,
                optimizer=optimizer_clients,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  model_Fedfilt.compile(loss=loss,
                optimizer=optimizer_clients,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  global_var = model_FedAvg.get_weights()
  global_var_FedAvg = global_var
  global_var_flat = Gr.flatten(global_var)
  global_var_flat = np.array(global_var_flat)
  G = np.matlib.repmat(global_var_flat,N_active_clients,1)
  G_list[0] = client_list
  G_list[1] = G
  G_list[2] = G
  G_list[3] = G
  G_list[4] = G
  total_data_size = 0
  for i in client_list:
    client_data_size = label_test_new[i].shape[0]
    total_data_size = total_data_size + client_data_size

  ## CHANNEL MODEL AND ESTIMATION FUNCTIONS
  def db2pow(ydb):
    return np.power(10, ydb/10)

  N_r = args.N_r
  N_clients = N_clients
  k = N_clients
  Ns = len(global_var_flat)

  H = (np.random.randn(N_r,k)+1j*np.random.randn(N_r,k))/np.sqrt(2);  # N_r*K, unit variance entries
  n = (np.random.randn(N_r,Ns)+1j*np.random.randn(N_r,Ns))/np.sqrt(2); # N*Ns, unit var entries
  def channel_model(S_user,H,n):
    s = S_user
    X_user = H@s + n
    return X_user

  def Estimate_QR(X_user,H):
    q1, r1 = np.linalg.qr(H, mode='reduced')
    X_tilda = q1.conj().T @ X_user
    x_hat_qr = np.linalg.inv(r1) @ X_tilda
    return x_hat_qr
  def Estimate_conventional(X_user,H):
    x_hat = np.linalg.inv(H.T @ H) @ H.T @ X_user
    return x_hat
  #######################
  ################################################################################
  ####################### Federated Learning  ####################################
  ################################################################################
  ## INITIALIZATION OF THE MODELS
  global_var_flat_conv = global_var_flat
  global_var_flat_true = global_var_flat
  global_var_flat_QR = global_var_flat
  global_var_flat_pers = global_var_flat
  ## TRAINING PROCEDURE
  for round in range(0,rounds):
    Delta_sum = None
    print("round ",round)
    client_num = 0
    client_list = range(N_total_clients)
    client_list = sorted(client_list, key = lambda x: random.random() )

    for client in client_list:
      print("Client ID ",client)

      ## shuffling data manually
      client_data_size = data_train_new[client].shape[0]

      client_xdata = np.expand_dims(data_train_new[client],axis =-1)
      client_ydata = label_train_new[client]

  ######################## TRAINING FOR THE MODEL WITH CONVENTIONAL ESTIMATION #######################################
      g_conv = Gr.unflatten(global_var_flat_conv,global_var)
      model_FedAvg.set_weights(g_conv)
      # model_FedAvg.set_weights(global_var_FedAvg)
      for epoch in range(E):
        info = model_FedAvg.fit(client_xdata, client_ydata ,batch_size=Batch_size,
    epochs=1,
    verbose=0,
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,)

      client_var_FedAvg = model_FedAvg.get_weights()
      client_var_FedAvg = Gr.flatten(client_var_FedAvg)
      client_var_FedAvg = np.array(client_var_FedAvg)
      client_var_FedAvg = np.expand_dims(client_var_FedAvg,axis=-1)
      if client_num==0:
        client_var_matrix_conv = client_var_FedAvg
      else:
        client_var_matrix_conv = np.concatenate([client_var_matrix_conv,client_var_FedAvg],axis=1)
      # client_num += 1

  ######################## TRAINING FOR THE MODEL WITH QR ESTIMATION #######################################
      g_QR = Gr.unflatten(global_var_flat_QR,global_var)
      model_FedAvg.set_weights(g_QR)
      # model_FedAvg.set_weights(global_var_FedAvg)
      for epoch in range(E):
        info = model_FedAvg.fit(client_xdata, client_ydata ,batch_size=Batch_size,
          epochs=1,
          verbose=0,
          callbacks=None,
          validation_split=0.0,
          validation_data=None,
          shuffle=True,)

      client_var_FedAvg = model_FedAvg.get_weights()
      client_var_FedAvg = Gr.flatten(client_var_FedAvg)
      client_var_FedAvg = np.array(client_var_FedAvg)
      client_var_FedAvg = np.expand_dims(client_var_FedAvg,axis=-1)
      if client_num==0:
        client_var_matrix_QR = client_var_FedAvg
      else:
        client_var_matrix_QR = np.concatenate([client_var_matrix_QR,client_var_FedAvg],axis=1)
      # client_num += 1
  ######################## TRAINING FOR THE MODEL WITH personalized CONVENTIONAL ESTIMATION #######################################
      # g_pers = alpha*G[client] + (1- alpha)*global_var_flat_pers
      alpha = args.alpha
      g_pers = Gr.unflatten(alpha*G[client] + (1-alpha)*global_var_flat_pers,global_var)
      model_FedAvg.set_weights(g_pers)
      # model_FedAvg.set_weights(global_var_FedAvg)
      for epoch in range(E):
        info = model_FedAvg.fit(client_xdata, client_ydata ,batch_size=Batch_size,
    epochs=1,
    verbose=0,
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,)

      client_var_FedAvg = model_FedAvg.get_weights()
      client_var_FedAvg = Gr.flatten(client_var_FedAvg)
      client_var_FedAvg = np.array(client_var_FedAvg)
      # client_var_FedAvg = alpha*client_var_FedAvg + (1-alpha)*global_var_flat_pers
      G[client,:] = client_var_FedAvg
      client_var_FedAvg = np.expand_dims(client_var_FedAvg,axis=-1)

      if client_num==0:
        client_var_matrix_pers = client_var_FedAvg
      else:
        client_var_matrix_pers = np.concatenate([client_var_matrix_pers,client_var_FedAvg],axis=1)

      # client_num += 1

  ######################## TRAINING THE MODEL WITH TRUE PARAMETERS (NO CHANNEL) #######################################
      g_true = Gr.unflatten(global_var_flat_true,global_var)
      model_FedAvg.set_weights(g_true)
      # model_FedAvg.set_weights(global_var_FedAvg)
      for epoch in range(E):
        info = model_FedAvg.fit(client_xdata, client_ydata ,batch_size=Batch_size,
          epochs=1,
          verbose=0,
          callbacks=None,
          validation_split=0.0,
          validation_data=None,
          shuffle=True,)

      client_var_FedAvg = model_FedAvg.get_weights()
      client_var_FedAvg = Gr.flatten(client_var_FedAvg)
      client_var_FedAvg = np.array(client_var_FedAvg)
      client_var_FedAvg = np.expand_dims(client_var_FedAvg,axis=-1)
      if client_num==0:
        client_var_matrix_true = client_var_FedAvg
      else:
        client_var_matrix_true = np.concatenate([client_var_matrix_true,client_var_FedAvg],axis=1)
      client_num += 1
    client_var_matrix_true = client_var_matrix_true.T
    client_var_matrix_conv = client_var_matrix_conv.T
    client_var_matrix_QR = client_var_matrix_QR.T
    client_var_matrix_pers = client_var_matrix_pers.T

    ########### Channel Effect #######################
    SNR = args.SNR
    n = (np.random.randn(N_r,Ns)+1j*np.random.randn(N_r,Ns))/np.sqrt(2); # N*Ns, unit var entries

    s_true  = db2pow(SNR)*client_var_matrix_true
    x_true  = channel_model(s_true,H,n)

    s_QR = db2pow(SNR)*client_var_matrix_QR
    x_QR = channel_model(s_QR,H,n)
    s_QR_pers = db2pow(SNR)*client_var_matrix_pers
    x_QR_pers = channel_model(s_QR_pers,H,n)

    s_conv = db2pow(SNR)*client_var_matrix_conv
    x_conv = channel_model(s_conv,H,n)
    ############## recover the signals ###############
    s_hat_conv = Estimate_conventional(x_conv,H)/db2pow(SNR)
    s_hat_QR = Estimate_QR(x_QR,H)/db2pow(SNR)
    s_pers = Estimate_QR(x_QR_pers,H)/db2pow(SNR)

    s_true = client_var_matrix_true
    ########### average over the estimated signals ############
    # global_var_flat_conv = np.mean(client_var_matrix,axis=0)
    global_var_flat_conv = np.mean(s_hat_conv.real,axis=0)

    global_var_flat_QR = np.mean(s_hat_QR.real,axis=0)
    global_var_flat_true = np.mean(s_true.real,axis=0)
    global_var_flat_pers = np.mean(s_pers.real,axis=0) ## the true aggregated models (no channel or noise effect. This is the ground truth)


  ########## FedAvg Evaluation without channel ###################################################
    g_true = Gr.unflatten(global_var_flat_true,global_var)

    acc_FedAvg = np.zeros(N_active_clients)
    for client in range(N_active_clients):
      _,acc_FedAvg[client] = evaluate_manual_data(g_true,data_test_new,label_test_new,ANN(),client)
    accuracy_FedAvg_local_true[round] = np.mean(acc_FedAvg)

    acc_FedAvg = np.zeros(1)
    for client in range(1):
      _,acc_FedAvg[client] = evaluate(g_true,dataset_test,ANN(),client)
    accuracy_FedAvg_general_true[round] = np.mean(acc_FedAvg)

  ########## FedAvg Evaluation with channel and conventional estimation ###################################################
    g_conv = Gr.unflatten(global_var_flat_conv ,global_var)

    acc_FedAvg = np.zeros(N_active_clients)
    for client in range(N_active_clients):
      _,acc_FedAvg[client] = evaluate_manual_data(g_conv ,data_test_new,label_test_new,ANN(),client)
    accuracy_FedAvg_local_conv[round] = np.mean(acc_FedAvg)

    acc_FedAvg = np.zeros(1)
    for client in range(1):
      _,acc_FedAvg[client] = evaluate(g_conv,dataset_test,ANN(),client)
    accuracy_FedAvg_general_conv[round] = np.mean(acc_FedAvg)

  ########## FedAvg Evaluation with channel and QR estimation ###################################################
    g_QR = Gr.unflatten(global_var_flat_QR,global_var)

    acc_FedAvg = np.zeros(N_active_clients)
    for client in range(N_active_clients):
      _,acc_FedAvg[client] = evaluate_manual_data(g_QR,data_test_new,label_test_new,ANN(),client)
    accuracy_FedAvg_local_QR[round] = np.mean(acc_FedAvg)

    acc_FedAvg = np.zeros(1)
    for client in range(1):
      _,acc_FedAvg[client] = evaluate(g_QR,dataset_test,ANN(),client)
    accuracy_FedAvg_general_QR[round] = np.mean(acc_FedAvg)
    ########## FedAvg Evaluation with channel and QR estimation and personalization###################################################


    acc_FedAvg = np.zeros(N_active_clients)
    for client in range(N_active_clients):
      g_QR_pers = Gr.unflatten(G[client],global_var)
      _,acc_FedAvg[client] = evaluate_manual_data(g_QR_pers,data_test_new,label_test_new,ANN(),client)
    accuracy_FedAvg_local_QR_pers[round] = np.mean(acc_FedAvg)

    acc_FedAvg = np.zeros(1)
    for client in range(1):
      g_QR_pers = Gr.unflatten(G[client],global_var)
      _,acc_FedAvg[client] = evaluate(g_QR_pers,dataset_test,ANN(),client)
    accuracy_FedAvg_general_QR_pers[round] = np.mean(acc_FedAvg)
  ################################################################################

  ################################################################################
    # print("FedAvg on general test data "+str(accuracy_FedAvg_general[round]))
    # print("FedAvg on local test data "+str(accuracy_FedAvg_local[round]))

    if round % 1==0:
      savemat(root+"/" +args.save_path+"/accuracy_FedAvg_general_true.mat", {"accuracy_FedAvg_general_true": accuracy_FedAvg_general_true})
      savemat(root+"/" +args.save_path+"/accuracy_FedAvg_general_conv.mat", {"accuracy_FedAvg_general_conv": accuracy_FedAvg_general_conv})
      savemat(root+"/" +args.save_path+"/accuracy_FedAvg_general_QR.mat", {"accuracy_FedAvg_general_QR": accuracy_FedAvg_general_QR})
      savemat(root+"/" +args.save_path+"/accuracy_FedAvg_general_QR_pers.mat", {"accuracy_FedAvg_general_QR_pers": accuracy_FedAvg_general_QR_pers})

      plt.plot(accuracy_FedAvg_general_true[:round],linewidth = 1.5)
      plt.plot(accuracy_FedAvg_general_conv[:round],linewidth = 1.5)
      plt.plot(accuracy_FedAvg_general_QR[:round],linewidth = 1.5)
      plt.plot(accuracy_FedAvg_general_QR_pers[:round],linewidth = 1.5)
      plt.title(str(N_active_clients)+" devices, "+str(E)+" local update")
      plt.xlabel("Rounds")
      plt.ylabel("General Test Accuracy")
      plt.legend(['FedAvg_true','OTA FL (conv.)',r'PerSOTA FL ($\alpha=0$)',r'PerSOTA FL ($\alpha=0.8$)'])
      plt.grid(True)

      plt.savefig(root+"/" +args.save_path+"/Acc_VS_rounds_general.jpg", bbox_inches='tight', dpi=120)
#       plt.savefig("Acc_VS_rounds_general.jpg", bbox_inches='tight', dpi=120)
      plt.close()

    if round % 1==0:
      savemat(root+"/" +args.save_path+"/accuracy_FedAvg_local_true.mat", {"accuracy_FedAvg_local_true": accuracy_FedAvg_local_true})
      savemat(root+"/" +args.save_path+"/accuracy_FedAvg_local_conv.mat", {"accuracy_FedAvg_local_conv": accuracy_FedAvg_local_conv})
      savemat(root+"/" +args.save_path+"/accuracy_FedAvg_local_QR.mat", {"accuracy_FedAvg_local_QR": accuracy_FedAvg_local_QR})
      savemat(root+"/" +args.save_path+"/accuracy_FedAvg_local_QR_pers.mat", {"accuracy_FedAvg_local_QR_pers": accuracy_FedAvg_local_QR_pers})


      plt.plot(accuracy_FedAvg_local_true[:round],linewidth = 1.5)
      plt.plot(accuracy_FedAvg_local_conv[:round],linewidth = 1.5)
      plt.plot(accuracy_FedAvg_local_QR[:round],linewidth = 1.5)
      plt.plot(accuracy_FedAvg_local_QR_pers[:round],linewidth = 1.5)

      plt.title(str(N_active_clients)+" devices, "+str(E)+" local update")
      plt.xlabel("Rounds")
      plt.ylabel("Local Test Accuracy")
      plt.grid(True)
      plt.legend(['FedAvg_true','OTA FL (conv.)',r'PerSOTA FL ($\alpha=0$)',r'PerSOTA FL ($\alpha=0.8$)'])
      plt.savefig(root+"/" +args.save_path+"/Acc_VS_rounds_local.jpg", bbox_inches='tight', dpi=120)
#       plt.savefig("Acc_VS_rounds_local.jpg", bbox_inches='tight', dpi=120)
      plt.close()

    if rounds % 1==0:
      display.clear_output(wait=True)

# if __name__ == '__main__':
#   main()
