# ***MLIA-Final-Project***

# **Dataset That needs to be Used**

Used Dataset: SYSU-Shape Dataset
Assume the dataset is saved in the root directory of the Github repository provided by OpenAI listed here: https://github.com/openai/improved-diffusion 

For example, the 64x64 model use only motorbikes images, which can be downloaded here: https://github.com/bearpaw/sysu-shape-dataset/tree/master/motorbike/images 

The location of where you stored the data is important for the   ``` - - data_dir flag ```, where ```- - data_dir /path/to/images ``` is sent to the training scripts. 

# Setting up
Before starting make sure you are in the ```/paper_code``` directory of this repository. 

Set up the ``` OPENAI_LOGDIR``` environment variable, where the logging directory has the logs and saved models. To do this, the following command is needed: 
``` export OPENAI_LOGDIR=/path/to/location/of/training/logs/folder```


## Pre-trained Models
Note: Open up this Google Drive [https://drive.google.com/drive/folders/1w2ChLLpXP7RsLk5bmOS0ul4Rh6qlwCkF?usp=drive_link] to access the pre-trained models for: 
* 64x64 diffusion
* 64x64 classifier
* 128x128 diffusion
* 128x128 classifier
* LSUN (no classifier guidance)

Download each of these ```.pt``` files in the ```/our_models``` folder.

# Diffusion Model

## Training

To train the model, the following hyperparameters were used: 

* 64x64 Model
  
```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 3e-4 --batch_size 128"
NUM_GPUS="2"
```

Once the hyperparameters are set, you can run the terminal command: 

```
mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir path/to/datasets/sysu-shape-dataset/motorbike/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

* 128x128 Model

```
MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr .0005 --batch_size 32"
```

Once the hyperparameters are set, you can run the terminal command: 

```
mpiexec -n 2 python ./image_train.py --data_dir path/to/dataset/sysu_dataset_diffusion/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
## Sampling

* 64x64 Model
  
Set the hyperparameters: 

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 200 --timestep_respacing 250"
```

Once the hyperparameters are set, you can run the terminal command: 

```
python scripts/image_sample.py --model_path /path/to/saved/models/64x64_diffusion.pt $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
```

* 128x128 Model
  
Run the terminal command: 

```
python ./image_sample.py --model_path checkpoints_and_results/diffusion_class_cond.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```

# Classifier Guidance

## Training

* 64x64 Model

Set the hyperparameters:

```
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 128 --lr 5e-4 --save_interval 500 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 64 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
```

Where batch size in TRAIN_FLAGS is divided by the number of MPI processes you are using (e.g., 2). 

Once the parameters are set, you can run the terminal command: 

```
mpiexec -n N python scripts/classifier_train.py --data_dir path/to/dataset/sysu-shape-dataset/motorbike/images $TRAIN_FLAGS $CLASSIFIER_FLAGS
```
* 128x128 Model
  
Set the hyperparameters:

```
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 32 --lr .003 --save_interval 500 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
```

## Sampling

* 64x64 Model
  
Set the hyperparameters:

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 200 --timestep_respacing 250"
64 --num_channels 128 --num_res_blocks 3"
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 64 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
```

After the flags are set, insert this terminal command: 

```
python classifier_sample.py $MODEL_FLAGS --classifier_path path/to/64x64/classifier/model --model_path path/to/saved/diffusion/model 
$SAMPLE_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS
```

* 128x128 Model

Set the hyperparameters: 

```
## Needs to match MODEL_FLAGS of diffusion model
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 25 --timestep_respacing ddim25 --use_ddim True"
```

After the flags are set, insert this terminal command: 

```
mpiexec -n N python scripts/classifier_sample.py \
    --model_path /path/to/model.pt \
    --classifier_path path/to/classifier.pt \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS
```

# LSUN

## Training

Set the hyperparameters: 

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 128 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
```

After the flags are set, insert this terminal command: 

```
mpiexec -n 2 python ./image_train.py --data_dir path/to/dataset/sysu_airplanes/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

## Sampling

```
python ./image_sample.py --model_path path/to/diffusion/model $MODEL_FLAGS $DIFFUSION_FLAGS
```

# Evaluations
There requirements of the evaluator.py script can be found in requirements.txt. Two arguments are passed in: reference batch and the sample batch. 
These batches are stored in the ```/training_logs``` folder. 

To evaluate, run the command 

```
python evaluations/evaluator.py training_logs/[reference_batch].npz training_logs/[sample_batch].npz
```

The following ```.npz``` files are reference batches: 
* ```reference_batch_64x64.npz```
* ```sample_batch.npz```
* ```airplanes_batch.npz```


The following ```.npz``` files are sample patches: 
* ```diffusion64_samples_200x64x64x3.npz```
* ```classifier64_samples_200x64x64x3.npz```
* ```diffusion_class_cond_no_guidance.npz```
* ```diffusion_class_cond_guidance.npz```
* ```LSUN_airplane_20000.npz```

To evaluate the 64x64 model, use the reference batch ```reference_batch_64x64.npz```. For the sample batch use: 
* ```diffusion64_samples_200x64x64x3.npz``` (diffusion)
* ```classifier64_samples_200x64x64x3.npz``` (classifier)
  
To evaluate the 128x128 model, use the reference batch ```sample_batch.npz```. For the sample batch use: 
* ```diffusion_class_cond_no_guidance.npz``` (diffusion)
* ```diffusion_class_cond_guidance.npz``` (classifier)

To evaluate the 128x128 LSUN, use the reference batch ```airplanes_batch.npz```. For the sample batch use: ```LSUN_airplane_20000.npz```. 

  

  



 





