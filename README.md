# SliderYolo
SliderYolo是采用百度飞桨PPYolo训练而来，可以识别易盾，云片，极验，腾讯等各种正方形滑块！识别率99.9999%！

# 使用方式

## 返回滑块坐标

在`slider_infer.py`文件中可以看到下面的函数：

```python
def infer():
    config = Config('./')  # 模型路径
    detector = Detector(config, './', use_gpu=False, run_mode='fluid')
    results = detector.predict('24487f4052354b988f5de1093b6e11c0.jpg', 0.5)  # 0.5 是阈值
    print('*' * 80)
    print(results)
    print('*' * 80)
```

运行该函数即可！

> 这种方式是只会产生坐标值，不会讲结果在原图上画出来！

## 返回滑块在原图的标注

如果需要显示标注结果可以运行`infer.py`，运行示例如下：

```python
python infer.py --model_dir=. --image_file=fc8572b93baa42d689bf4915065b8c7a.jpg --use_gpu=False
```

- --model_dir 代表模型文件路径 
- --image_file 代表图片路径
- --use_gpu 代表是否启用GPU 

> 注意：还有更多参数，可以看infer.py中的源码部分！

识别结果：

```
-----------  Running Arguments -----------
camera_id: -1
image_file: fc8572b93baa42d689bf4915065b8c7a.jpg
model_dir: .
output_dir: output
run_benchmark: False
run_mode: fluid
threshold: 0.5
use_gpu: False
video_file:
------------------------------------------
-----------  Model Configuration -----------
Model Arch: YOLO
Use Paddle Executor: False
Transform Order:
--transform op: Resize
--transform op: Normalize
--transform op: Permute
--------------------------------------------
Inference: 1513.9191150665283 ms per batch image
class_id:0, confidence:0.9938,left_top:[66.65,39.17], right_bottom:[108.25,80.51]
class_id:0, confidence:0.9922,left_top:[165.78,74.90], right_bottom:[207.54,114.06]
******************** {'boxes': array([[  0.        ,   0.9938261 ,  66.645035  ,  39.16696   ,
        108.24666   ,  80.5123    ],
       [  0.        ,   0.99220854, 165.77527   ,  74.898384  ,
        207.54053   , 114.0648    ]], dtype=float32)}
```

![fc8572b93baa42d689bf4915065b8c7a](https://github.com/EnjoyScraping/SliderYolo/blob/main/output/fc8572b93baa42d689bf4915065b8c7a.jpg)
