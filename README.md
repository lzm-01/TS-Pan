# [TGRS 2025] Two-Step Pansharpening: A High-Frequency-Guided Spatial-Spectral Enhancement Network Based on Mixture of Experts
[Zhuomao Li][zhuomao], 
Ying Li, 
Zhihao Li, 
[Yikun Liu][yikun], 
[Gongping Yang][gongping]

[zhuomao]: https://time.sdu.edu.cn/info/1069/2627.htm
[yikun]: https://scholar.google.com/citations?user=aLjH3NUAAAAJ&hl=zh-CN&oi=ao
[gongping]: https://faculty.sdu.edu.cn/gpyang/zh_CN/index.htm


## This repository is the official implementation of our paper. 
### TS-Pan Framework
![这是图片](/src/net.png)
### High-Frequency-Guided Local-Glocal Enhancement Block (HFG-LGEB)
![这是图片](/src/hfg.png)

## Configuration before Training
Before training the model, you need to configure the following options in the `option.yaml` file:
  - `log_dir`: the directory to store the training log files.
  - `checkpoint`: the directory to store the trained model parameters.
  - `data_dir_train`: the directory of the training data.
  - `data_dir_eval`: the directory of the evaluation data.

## Training the Model
To train the model, you can run the following command:
```
python main.py
```

## Testing the Model
To test the trained pan-sharpening model, you can run the following command:
```
python test.py
```

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


