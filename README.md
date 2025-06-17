## Introduction
This repository contains the implementation of the proposed method and experimental analysis of the paper “A self-attention TCN-based model for suicidal ideation detection from social media posts”, published in Expert Systems with Applications, 2024. 
In this paper, we created a novel deep learning model that can find suicidal thoughts in social media posts, and a thorough analysis using a combination of different techniques to analyze the posts. The experimental results demonstrate the high accuracy results of our model at identifying when someone might be exhibits signs of suicidal thoughts. This research could help people who are struggling with mental health by providing an early warning system. 
Our experimental results demonstrate that AL-BTCN achieves over **94% accuracy, recall, and F1-score**, making it a highly effective tool for early detection of suicidal risk based on textual content. This research contributes to mental health support systems by providing a potential early warning tool.


## Practical Guideline: Proposed Model
The implementations of the proposed model, as well as the experimented state-of-the-art approaches to which we have compared our presented model, are all included in this repository. 
Executing the codes are pretty straightforward. In order to run the code, you should follow these steps:
1)	Installation and Importing the required libraries
a.	Such as: Tensorflow, Scikit-Learn, NLKT, …
2)	Loading the dataset and preprocessing texts in order to remove any URLs, Email addresses, Hashtags, and Punctuations, and balancing the dataset.
3)	Importing and loading the BERT pre-trained model
4)	Defining the model
5)	Training the model on the training and validation dataset
6)	Evaluating the model on the test dataset

## Data Analysis
The analyses conducted on the suicidal tweets used in our paper includes semantic analysis, sentiment analysis, and topic prediction. These analyses are all included in the Data Analysis folder. You can refer to these implementations for further analyzing the datasets, in particular suicidal tweets. Such analyses can provide a proper insight regarding the differences between a suicidal note and a typical non-suicidal tweet. 

## State-of-the-art Baselines
The recently published baselines are presented in the State of the art Baselines folder. The execution of these models are principally similar to the practical guideline written above. Following the same process of loading, preprocessing, and balancing the datasets, defining the models and training and evaluating these models, the mere difference is related to the model definition which is based on the architectures presented in referenced papers [[1]](#1) [[2]](#2). 

## References
<a id="1">[1]</a> 
Renjith, Shini, et al. "An ensemble deep learning technique for detecting suicidal ideation from posts in social media platforms." Journal of King Saud University-Computer and Information Sciences 34.10 (2022): 9564-9575.

<a id="2">[2]</a> 
Ghosh, Tapotosh, et al. "An attention-based hybrid architecture with explainability for depressive social media text detection in Bangla." Expert Systems with Applications 213 (2023): 119007.


# AL-BTCN: A Self-Attention TCN-Based Model for Suicidal Ideation Detection from Social Media Posts

This repository contains the official implementation and experimental resources for the paper:

> **A self-attention TCN-based model for suicidal ideation detection from social media posts**  
> Seyedeh Leili Mirtaheri, Sergio Greco, Reza Shahbazian  
> *Expert Systems with Applications*, Volume 255, Part D, 2024, 124855  
> [https://doi.org/10.1016/j.eswa.2024.124855](https://doi.org/10.1016/j.eswa.2024.124855)  
> [Read on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417424017226)

---

## 📘 Introduction

This repository contains the implementation of the proposed method and experimental analysis of the paper _“A self-attention TCN-based model for suicidal ideation detection from social media posts,”_ published in *Expert Systems with Applications*, 2024.

In this paper, we created a novel deep learning model that can detect suicidal ideation in social media posts, combined with a thorough semantic and statistical analysis. The model, named **AL-BTCN**, leverages a hybrid **LSTM–TCN architecture** enhanced with a **self-attention mechanism** to effectively capture linguistic and contextual features. 


---

## ⚙️ Practical Guideline: Proposed Model

The repository includes:

- Implementation of the proposed AL-BTCN model
- Baseline implementations of other state-of-the-art methods
- Preprocessing and data loading routines
- Scripts for training, evaluation, and visualization


## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{Mirtaheri2024Suicidal,
  title     = {A self-attention TCN-based model for suicidal ideation detection from social media posts},
  author    = {Seyedeh Leili Mirtaheri and Sergio Greco and Reza Shahbazian},
  journal   = {Expert Systems with Applications},
  volume    = {255},
  number    = {Part D},
  pages     = {124855},
  year      = {2024},
  issn      = {0957-4174},
  doi       = {10.1016/j.eswa.2024.124855},
  url       = {https://www.sciencedirect.com/science/article/pii/S0957417424017226}
}
```
