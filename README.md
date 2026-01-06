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
### The complete code and README files will be uploaded before the end of Jan. 2026. 
### TS-Pan Framework
![这是图片](/src/net.png)
### High-Frequency-Guided Local-Glocal Enhancement Block (HFG-LGEB)
![这是图片](/src/hfg.png)

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

