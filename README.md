## SDM

This is the code for "Video Quality Assessment with Serial Dependence Modeling, TMM, 2021". 

The code was created in 2019 when I had the first attempt with deep learning, so the code seems very disordered, and I have no plan to reorganize them :)

### Usage

The model consists of two parts: 
- feature extraction with MATLAB in the folder `./matlab`;
- recurrent modeling with PyTorch.

1. Feature extraction is based on the previous work ([FAST-TMM-2019](https://github.com/Sissuire/FAST-VQA)), where OpenCV is requested. More details see in the folder. Features from the four VQA databases (i.e., LIVE, CSIQ, IVPL, and IVC-IC) can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/11pO-M93T5Ao0sR17srTnhUBlJwXxB9ov?usp=sharing) or [百度云](https://pan.baidu.com/s/1f_pzrXTBKD5QJQWWuL47ug)(提取码：i8y8)  (~1G in total).

2. Once features are extracted, the major is to set the paths in config files `xxx.yaml` to note where the precomputed feature data stores, and where the dataset information file (`xxx_list_for_VQA.txt`, we have put them in the folder).

Example code would be seen in `demo_loop.py` or `demo_IVC-IC.py`. More details can be determined with a step-by-step running in debug mode. If you are only interested in the implementation of `A-LSTM` or `attention`, please check them in the file `./model/rnn_imp.py`.

*** In current work, we only validate the performance on FR-VQA, but we hope this work could be transferred into broader scenarios (for example, UGC-VQA, or others). It is easy to reserve the second step but substitute the first step with specific modeling. ***

### Cite

If you are interested in the work, or find the code helpful, please cite our work

```
@ARTICLE{sdm,
  author={Liu, Yongxu and Wu, Jinjian and Li, Aobo and Li, Leida and Dong, Weisheng and Shi, Guangming and Lin, Weisi},
  journal={IEEE Transactions on Multimedia}, 
  title={Video Quality Assessment with Serial Dependence Modeling}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2021.3107148}}
```
### Contact

If any question or bug, feel free to contact me via `yongxu.liu@stu.xidian.edu.cn`.
