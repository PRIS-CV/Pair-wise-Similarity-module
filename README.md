# Pair-wise-Similarity-module

Code release for the paper "Learning Calibrated Class Centers for Few-shot Classification by Pair-wise Similarity"

Requirements
-------  
* python=3.6
* PyTorch=1.2+
* torchvision=0.4.2
* pillow=6.2.1
* numpy=1.18.1
* h5py=1.10.2

Dataset
------- 
* CUB-200-2011
   Change directory to ./filelists/CUB
   run source ./download_CUB.sh
   
Train
------- 

* method: relationnet|relationnet_PSM|protonet|protonet_PSM.
* n_shot: number of labeled data in each class （1|5）.
* train_aug: perform data augmentation or not during training.

python ./train.py --dataset CUB  --model Conv4 --method protonet --n_shot 1 <br> 
python ./train.py --dataset CUB  --model Conv4 --method protonet_PSM --n_shot 1 

Save features
------- 
python ./save_features.py --dataset CUB  --model Conv4 --method protonet --n_shot 1 <br> 
python ./save_features.py --dataset CUB  --model Conv4 --method protonet_PSM --n_shot 1

Test
------- 
python ./test.py --dataset CUB  --model Conv4 --method protonet --n_shot 1 <br> 
python ./test.py --dataset CUB  --model Conv4 --method protonet_PSM --n_shot 1

References
------- 
Our code is based on Chen's contribution. Specifically, except for our core design, protonet_PSM and relationnet_PSM, everything else （e.g. backbone, dataset, relation network, evaluation standards, hyper-parameters）are built on and integrated in https://github.com/wyharveychen/CloserLookFewShot.

Contact
------- 
Thanks for your attention! If you have any suggestion or question, you can leave a message here or contact us directly:<br> 

guoyurong@bupt.edu.cn<br> 
mazhanyu@bupt.edu.cn

