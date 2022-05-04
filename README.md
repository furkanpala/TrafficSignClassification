# Traffic Sign Classifcation Using VGG
## Model
*K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv 1409.1556, 09 2014. https://arxiv.org/abs/1409.1556*
## Dataset
[GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

| Split | Samples |
| ----------- | ----------- |
| Train | 39209 |
| Validation | 6315 |
| Test | 6315 |

*We halved the testing set using random split to get a validation set (see the seed in config.yaml). Validation set is not considered for the results. This will change in future commits.*
![Class distribution of training set](images/dataset_dist.png)
![Preprocessing pipeline](images/preprocess.png)


## Results
| Model | Micro F1 | Macro F1 | Top-5 Accuracy
| ----------- | ----------- | ----------- | ----------- |
| VGG-19 | 97.07% | 95.25% | 98.86% |

![Loss graph](images/losses.png)
![F1 scores graph](images/f1_scores.png)
![F1 scores graph](images/lr_decay.png)
![F1 scores graph](images/cm.png)
![F1 scores graph](images/samples.png)