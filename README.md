# [TGRS 2026] Two-Step Pansharpening: A High-Frequency-Guided Spatial-Spectral Enhancement Network Based on Mixture of Experts
[Zhuomao Li][zhuomao], 
Ying Li, 
Zhihao Li, 
[Yikun Liu][yikun], 
[Gongping Yang][gongping]

[zhuomao]: https://time.sdu.edu.cn/info/1069/2627.htm
[yikun]: https://scholar.google.com/citations?user=aLjH3NUAAAAJ&hl=zh-CN&oi=ao
[gongping]: https://faculty.sdu.edu.cn/gpyang/zh_CN/index.htm


This repository is the official implementation of our paper:  
[Two-Step Pansharpening: A High-Frequency-Guided Spatial-Spectral Enhancement Network Based on Mixture of Experts][tspan],  
IEEE Transactions on Geoscience and Remote Sensing (TGRS) 2026.

[tspan]: https://ieeexplore.ieee.org/document/11316500


### TS-Pan Framework
![这是图片](/src/net.png)
### High-Frequency-Guided Local-Global Enhancement Block (HFG-LGEB)
![这是图片](/src/hfg.png)

## Before Training
Before training the model, you need to configure the following options in the `option.yaml` file:
  - `log_dir`: the directory to store the training log files.
  - `checkpoint`: the directory to store the trained model parameters.
  - `data_dir_train`: the directory of the training data.
  - `data_dir_eval`: the directory of the evaluation data.

In `main.py`, you must set the right path about the `option.yml`.
Also, you need to configure the right path in `save_config` and `save_net_config` function of `utils.py`

### Preparing the Datasets
We provide the code for dataset in `data\load_train_data.py` and `data\load_test_data.py`. When training your model on the Gaofen-2 dataset, you must normalize the data by dividing by `1023`. The same applies to testing.

## Training the Model
To train the model, you can run the following command:
```
python main.py
```

## Testing the Model
To test the trained pan-sharpening model, you can run the following command and then you will find `.mat` files in `save_dir` path:
```
python test.py
```
Next, you can use Matlab to obtain evaluation metrics (PSNR, Q2n, SAM,...). More testing details please refer to [Test-toolbox-for-traditional-and-DL(Matlab)][test]. And we also provide our testing scripts (`TestRR` and `TestFR`) and `PSNR` metric in our repo.

[test]: https://github.com/liangjiandeng/DLPan-Toolbox/tree/main/02-Test-toolbox-for-traditional-and-DL(Matlab)

## Model Efficiency
To obtain model efficiency metrics (Params, FLOPs, Inference Time, FPS), you can run the following command:
```
python params_flops_inference_time.py
```

## Configuration
The configuration options are stored in the `option.yaml` file. Here is an explanation of each of the options:
### Your Model Name
  - `name`: The model name in python file for training (In this file, we named `Net` in `tspan.py`)
### algorithm
  - `algorithm`: The model for training (your model file, i.e., tspan)
### Logging
  - `log_dir`: The location where the log files will be stored.
### Model Weights
  - `checkpoint`: The location to store the model weights.
### Training Data
  - `data_dir_train`: The location of the training data.
  - `data_dir_eval`: The location of the test data.
### Pretrain
  - `pretrained`: Whether to use a pretrained model. (Bool)
  - `pre_sr`: The location of the pretrained model.
  - `pre_folder`: The location where the pretrained models are stored.
### Testing
  - `algorithm`: The algorithm to use for testing.
  - `data_dir`: The location of the test data.
  - `model`: The location of the model to use for testing.
  - `save_dir`: The location to save the test results.
### Data Processing
  - `batch_size`: The size of each batch.
  - `n_colors`: The number of color channels.
### Training Hyperparameters
  - `schedule.lr`: The learning rate.
  - `schedule.optimizer`: The optimizer to use, either `ADAM`, `SGD`, or `RMSprop`.
  - `schedule.momentum`: The momentum for the `SGD` optimizer.
  - `schedule.loss`: The loss function.

## Citation
  Please kindly cite our work if this work is helpful for your research.
```
@article{li2025two,
  title={Two-Step Pansharpening: A High-Frequency-Guided Spatial--Spectral Enhancement Network Based on Mixture of Experts},
  author={Li, Zhuomao and Li, Ying and Li, Zhihao and Liu, Yikun and Yang, Gongping},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={64},
  pages={1--19},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgement

- We would like to thank Professor [Deng][dengliangjian] for providing pan-sharpening datasets in [PanCollection][pancollection].
- This code repo is based on the [pansharpening code framework][ps], and we are very grateful for the contributions made by [manman1995][zhouman].
- Thanks for the contributions of [RWKV-UNet][rwkv-u] and [Restore-RWKV][res-rwkv] to the [Vision-RWKV][vrwkv] community.

[dengliangjian]: https://liangjiandeng.github.io/
[pancollection]: https://liangjiandeng.github.io/PanCollection.html
[ps]: https://github.com/manman1995/pansharpening
[zhouman]: https://github.com/manman1995
[rwkv-u]: https://github.com/juntaoJianggavin/RWKV-UNet
[res-rwkv]: https://github.com/Yaziwel/Restore-RWKV
[vrwkv]: https://github.com/OpenGVLab/Vision-RWKV


