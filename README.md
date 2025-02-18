# Edge AI Foundation Wake Vision Challenge - Model Centric Track Submission

This is the official submission for the data centric track for the Edge AI Foundation Wake Vision Challenge.

## Overview

This repository contains our submission focusing on model-centric improvements to enhance wake word detection performance on microcontrollers. The main idea behind the submission is to use a smaller custom model architecure based on mcunet architecture with training improvments to achieve a better accuracy.

The submission includes:

- Model architecture
- Data processing 
- Model training configuration
- TFLite model export for microcontroller deployment

## Envirenment Setup
we recomend using the official torch from docker hub to avoid any issues with the dependencies.

```bash
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

```bash
# run the container with the current directory mounted to /workspace
docker run -it --gpus all pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel -v $(pwd):/workspace
```

```bash
# install the dependencies
pip install ultralytics albumentations albumentations-pytorch wandb datasets
```


## Training Data Downloader

After exporting the data we download the data from huggingface and save it in image net format.

```bash
# run the script to download the pretrain data
python download.py --dataset Harvard-Edge/Wake-Vision-Train-Large --split train_large --images_per_shard 5760428 --shard_id 0 --false_positive_csv false_positives.csv --false_negative_csv false_negatives.csv --dual_save
```
This will download the data and relabel it on the fly in steaming mode takes about 3 hours (@ 2Gbits internet speed) to finish. this is slow but uses only 200GB of storage.

> **Warning**: Using the `--download_all` flag will download the entire dataset without sharding and extracting. This is **not recommended** unless you have:
> - 5+ Gbps internet speed
> - 256+ threads available for extraction
> - 3TB of available storage for the ImageNet format export
>
> Even with these requirements, the process will still take approximately 45 minutes to complete.

```bash
# run the script to download the fine data
python download.py --dataset Harvard-Edge/Wake-Vision-Train-Large --split train_large --images_per_shard 5760428 --shard_id 0 --false_positive_csv false_positives.csv --false_negative_csv false_negatives.csv --dual_save
```




## Repository initialization.

we use the original weights from of mcunet-vww2 from the hanlab repository. for that we fork the repository and the build the rest of the process on top of it.

to load the pretrained weights on vww dataset we use:

```python
from mcunet.model_zoo import net_id_list, build_model, download_tflite
print(net_id_list)  # the list of models in the model zoo      
model, image_size, description = build_model(net_id="mcunet-vww2", pretrained=True)  # you can replace net_id with any other option from net_id_list
```

## model architecture:
We derive the model architecture from mcunet-vww0 by downsizing it following changes:

* Input Resolution: Uses 48×48, which is lower than vww0 (64×64).

* First Conv Channels: Starts with 12 output channels (vs. 16 in vww0).

* Number of Blocks:
8 blocks total vs 14-block vww0.

* Gradual channel expansion from 12 → 16 → 24 → 32 → 40.

* Classifier: Has 40 input features into the final linear layer, compared to 160 in vww0.
```json

{
  "name": "ProxylessNASNets",
  "bn": {
    "momentum": 0.1,
    "eps": 1e-05,
    "ws_eps": null
  },
  "first_conv": {
    "name": "ConvLayer",
    "kernel_size": 3,
    "stride": 2,
    "dilation": 1,
    "groups": 1,
    "bias": false,
    "has_shuffle": false,
    "in_channels": 3,
    "out_channels": 12,
    "use_bn": true,
    "act_func": "relu6",
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
  },
  "blocks": [
    {
      "name": "MobileInvertedResidualBlock",
      "mobile_inverted_conv": {
        "name": "MBInvertedConvLayer",
        "in_channels": 12,
        "out_channels": 12,
        "kernel_size": 3,
        "stride": 1,
        "expand_ratio": 1,
        "mid_channels": null,
        "act_func": "relu6",
        "use_se": false
      },
      "shortcut": null
    },
    {
      "name": "MobileInvertedResidualBlock",
      "mobile_inverted_conv": {
        "name": "MBInvertedConvLayer",
        "in_channels": 12,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 2,
        "expand_ratio": 3,
        "mid_channels": 36,
        "act_func": "relu6",
        "use_se": false
      },
      "shortcut": null
    },
    {
      "name": "MobileInvertedResidualBlock",
      "mobile_inverted_conv": {
        "name": "MBInvertedConvLayer",
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 1,
        "expand_ratio": 3,
        "mid_channels": 48,
        "act_func": "relu6",
        "use_se": false
      },
      "shortcut": {
        "name": "IdentityLayer",
        "in_channels": [16],
        "out_channels": [16],
        "use_bn": false,
        "act_func": null,
        "dropout_rate": 0,
        "ops_order": "weight_bn_act"
      }
    },
    {
      "name": "MobileInvertedResidualBlock",
      "mobile_inverted_conv": {
        "name": "MBInvertedConvLayer",
        "in_channels": 16,
        "out_channels": 24,
        "kernel_size": 5,
        "stride": 2,
        "expand_ratio": 3,
        "mid_channels": 48,
        "act_func": "relu6",
        "use_se": false
      },
      "shortcut": null
    },
    {
      "name": "MobileInvertedResidualBlock",
      "mobile_inverted_conv": {
        "name": "MBInvertedConvLayer",
        "in_channels": 24,
        "out_channels": 24,
        "kernel_size": 3,
        "stride": 1,
        "expand_ratio": 4,
        "mid_channels": 96,
        "act_func": "relu6",
        "use_se": false
      },
      "shortcut": {
        "name": "IdentityLayer",
        "in_channels": [24],
        "out_channels": [24],
        "use_bn": false,
        "act_func": null,
        "dropout_rate": 0,
        "ops_order": "weight_bn_act"
      }
    },
    {
      "name": "MobileInvertedResidualBlock",
      "mobile_inverted_conv": {
        "name": "MBInvertedConvLayer",
        "in_channels": 24,
        "out_channels": 32,
        "kernel_size": 5,
        "stride": 2,
        "expand_ratio": 4,
        "mid_channels": 96,
        "act_func": "relu6",
        "use_se": false
      },
      "shortcut": null
    },
    {
      "name": "MobileInvertedResidualBlock",
      "mobile_inverted_conv": {
        "name": "MBInvertedConvLayer",
        "in_channels": 32,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "expand_ratio": 3,
        "mid_channels": 96,
        "act_func": "relu6",
        "use_se": false
      },
      "shortcut": {
        "name": "IdentityLayer",
        "in_channels": [32],
        "out_channels": [32],
        "use_bn": false,
        "act_func": null,
        "dropout_rate": 0,
        "ops_order": "weight_bn_act"
      }
    },
    {
      "name": "MobileInvertedResidualBlock",
      "mobile_inverted_conv": {
        "name": "MBInvertedConvLayer",
        "in_channels": 32,
        "out_channels": 40,
        "kernel_size": 3,
        "stride": 1,
        "expand_ratio": 4,
        "mid_channels": 128,
        "act_func": "relu6",
        "use_se": false
      },
      "shortcut": null
    }
  ],
  "feature_mix_layer": null,
  "classifier": {
    "name": "LinearLayer",
    "in_features": 40,
    "out_features": 2,
    "bias": true,
    "use_bn": false,
    "act_func": null,
    "dropout_rate": 0,
    "ops_order": "weight_bn_act"
  },
  "resolution": 48
}

```

## Data Augmetation 

We use the albumentations library to augment the data. We use a strong augmentation pipeline including various standard image ops as well as weather operations as well.

```python
def train_augmentation_pipeline(img_size: int):
    return A.Compose([
        # 1. Always crop & resize to target dimensions
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),

        # 2. Basic horizontal flip
        A.HorizontalFlip(p=0.5),

        # 3. Light color transformations (choose 1 out of the 4)
        A.SomeOf(
            transforms=[
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.ToGray(p=1.0),
                A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=1.0)
            ],
            n=1,              # Pick exactly 1 transform to apply
            p=0.4             # 40% chance to apply any color augmentation
        ),

        # 4. Apply a blur or motion blur with moderate kernel sizes
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.3),

        # 5. Mild geometric transformations
        A.ShiftScaleRotate(
            shift_limit=0.05,     # small shift
            scale_limit=0.1,      # up to +/-10% scale
            rotate_limit=10,      # up to +/-10 degrees
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.2
        ),

        # 6. Occasional random weather condition
        A.OneOf([
            A.RandomRain(
                brightness_coefficient=0.9,
                drop_length=8,       # smaller drop length
                drop_width=1,
                blur_value=5,
                rain_type='drizzle',
                p=1.0
            ),
            A.RandomFog(fog_limit=(10, 20), alpha_coef=0.05, p=1.0),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=1.0
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0.5,
                p=1.0
            ),
        ], p=0.15),  # 15% chance of applying any weather effect

        # 7. Occasional Coarse Dropout
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=4,
            min_height=8,
            min_width=8,
            p=0.2
        ),

        # 8. Normalize & convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
```

This pipeline was crafted through trail and error on a small subset of the data to find the best balance between data augmentation and model performance. MixUP based ops were discarded as they causing training instability.

## Symmetric Cross Entropy Loss

We use the Symmetric Cross Entropy (SCE) loss to improve the performance and robustness of the model. Label smoothing was not used as it did not yield any improvment.

The SCE loss combines regular cross entropy with a reverse cross entropy term:

L = α * CE(p, q) + β * CE(q, p)

where:
- CE(p,q) is the standard cross-entropy loss
- CE(q,p) is the reverse cross-entropy loss
- α and β are weighting parameters (typically α=1.0, β=0.1)
- p are the true labels
- q are the predicted probabilities

The reverse CE term helps prevent the model from being overconfident and makes it more robust to noisy labels. The symmetric nature of the loss provides better regularization compared to standard cross entropy.

Through experimentation, we found α=1.0 and β=0.1 worked best on our validation set. Higher β values decreased accuracy, suggesting our relabeling process produced relatively clean labels.


## Training Loop
*first we prepretrain the model on the vww dataset which can be build using the script: build_vww_dataset.py*

After that to train the model we use the train script which is a very optimized for speed to handle the 5.7 million images in the train-large split.

```bash
# run the script to train the model

CUDA_VISIBLE_DEVICES=0,1  torchrun --nproc_per_node=2 train.py   --batch_size 256 --learning_rate 3e-4 --data_dir full_dataset_human_vs_nohuman_relabeled/ --test_dir shard_0_human_vs_nohuman --val_dir shard_0_human_vs_nohuman --num_workers 256 --net_id mcunet-tiny2 --use_sce True  --label_smoothing 0 --wandb-project mcunet_tiny --scheduler cosine 
```


This will train the model on 2 GPUs in parallel and use 256 workers for data loading. it takes around 14 mins per epoch on two RTX 4090s. 



To train the model on a single GPU please refere to the stable_training_singGPU branch from this repo: https://github.com/benx13/mount_trainer.git


## Results

after 42 epoch we stopped training with best test acc @ 78.51%

we then launched a pretraining on quality dataset we stopped the training at epoch 35 with best test acc @ 80.28%


Results could be improved by training furhter, we stop here due to time constraints.

<a href="https://ibb.co/0R45wWYM"><img src="https://i.ibb.co/Kp4n1QVq/Screenshot-2025-02-12-at-7-32-44-AM.png" alt="Screenshot-2025-02-12-at-7-32-44-AM" border="0"></a>

Note: the image says val acc but it's actually test acc.


## export into tflite 

```bash
# run the script to export the model into tflite
python tflite_exporter.py
```
* please note model weights are stored in folder checkpoint_model_centric
* Please when running the evaluation use a process of data loading simular to eval_tflite.py to avoid data drift

Final model achives results: 
*  Test acc: 80.28@
*  MAC: 1751060
*  Flash:  41.97 KiB
*  RAM: 47.54 KiB

This will export the model into tflite format. please note this works only with the following versions of the modules in requirements_tflite.txt

