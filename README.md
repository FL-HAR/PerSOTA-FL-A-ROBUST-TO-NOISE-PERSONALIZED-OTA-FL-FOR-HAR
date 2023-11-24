# PerSOTA: A Robust-To-Noise Personalized Over The Air Federated Learning for Human Activity Recognition

This is the official code for the submitted paper PerSOTA FL: A ROBUST-TO-NOISE PERSONALIZED OVER THE AIR FEDERATED LEARNING FOR HUMAN ACTIVITY RECOGNITION. For now, the code only works on HARBAX dataset. we will keep updating the code in the upcoming days.


## Requirements
Please install the following packages before running ``` main.py``` .
```
!pip install ??
```
## How to run
First, download the HARBOX dataset from [Here]([https://pages.github.com/](https://github.com/xmouyang/FL-Datasets-for-HAR)). Click on the dropbox list and download ```Large_Scale_HARBox.zip```.

Second, go to ```\utils\dataset.py``` and find the function ``` load_HARboX_data_v2() ``` and modify ``` read_path = r'YOUR_PATH\large_scale_HARBox/'+str(client_id)+'/'+class_id+'.txt'``` to address the dataset properly.

Finally, run the following code that specifies the report path, number of local updates, number of rounds, the SNR, The number of receivers and the personalization parameter.
```
python main.py --save_path 'report' --E 3  --R 400 --SNR 5 --N_r 100 --alpha 0.5
```
Or, ```PerSOTA_FL_notebook``` in google colab.

## Some Results
<p align="center">
  <img src="imgs/Acc_VS_rounds_local.jpg" width="400">
  
</p>

<p align="center">
   <img src="imgs/Acc_VS_rounds_local_2.jpg" width="800">
</p>

# Citation
```
Pending
```
