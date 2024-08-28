Hybrid Graph Reasoning with Dynamic Interaction for Visual Dialog
====================================
Shanshan Du, Hanli Wang, Tengpeng Li, and Chang Wen Chen

### Overview:
As a pivotal branch of intelligent human-computer interaction, visual dialog is a technically challenging task that requires artificial intelligence (AI) agents to answer consecutive questions based on image content and history dialog. Despite considerable progresses, visual dialog still suffers from two major problems: (1) how to design flexible cross-modal interaction patterns instead of over-reliance on expert experience and (2) how to infer underlying semantic dependencies between dialogues effectively. To address these issues, an end-to-end framework employing dynamic interaction and hybrid graph reasoning is proposed in this work. Specifically, three major components are designed and the practical benefits are demonstrated by extensive experiments. First, a dynamic interaction module is developed to automatically determine the optimal modality interaction route for multifarious questions, which consists of three elaborate functional interaction blocks endowed with dynamic routers. Second, a hybrid graph reasoning module is designed to explore adequate semantic associations between dialogues from multiple perspectives, where the hybrid graph is constructed by aggregating a structured coreference graph and a context-aware temporal graph. Third, a unified one-stage visual dialog model with an end-to-end structure is developed to train the dynamic interaction module and the hybrid graph reasoning module in a collaborative manner. Extensive experiments on the benchmark datasets of VisDial v0.9 and VisDial v1.0 demonstrate the effectiveness of the proposed method compared to other state-of-the-art approaches.


### Method:
An overview of the proposed unified one-stage HGDI model is illustrated in Fig. 1. First, the feature encoder encodes visual and textual features into a common vector space to yield higher-level representations suitable for cross-modal interaction. Then, the dynamic interaction module is designed to offer more flexible interaction patterns as described. Meanwhile, the hybrid graph reasoning module combines the proposed structured coreference graph and the context-aware temporal graph to infer more reliable dialog semantic relations. Then, the vision-guided textual features after multi-step graph reasoning are fed into the answer decoder to predict reasonable answers. Finally, multiple loss functions are utilized to simultaneously optimize all modules.


<p align="center">
<image src="source/fig1.jpg" width="650">
<br/><font>Fig. 1. The framework of the proposed HGDI for visual dialog.</font>
</p>

### Results:
Our proposed model HGDI is compared with several state-of-the-art visual dialog models in the discriminative setting and generative setting on two public datasets. The experimental results are shown in Table 1 and Table 2. Moreover, qualitative experiments are conducted on the VisDial v1.0 validation set to verify the effectiveness of the proposed HGDI, as illustrated in Fig. 2 and Fig. 3.

<p align="center">
<font>Table 1. Comparison with the state-of-the-art discriminative models on both VisDial v0.9 validation set and v1.0 test set.</font><br/>
<image src="source/fig2.png" width="450">
</p>
<p align="center">
<font>Table 2. Comparison with the state-of-the-art generative models on both VisDial v0.9 and v1.0 validation sets.</font><br/>
<image src="source/fig3.png" width="450">
</p>
<p align="center">
<image src="source/fig4.jpg" width="450">
<br/><font>Fig. 2. Visualization results of the inferred semantic structures on the validation set of VisDial v1.0. The following abbreviations are used: question (Q), generated answer (A), caption (C), and question-answer pair (D). The darker the color, the higher the relevance score.</font>
</p>
<p align="center">
<image src="source/fig5.jpg" width="450">
<br/><font>Fig. 3. Visualization samples of visual attention maps and object-relational graphs during a progressive multi-round dialog inference. The ground-truth answer (GT) and the predicted answer achieved by HGDI (Ours) are presented.</font>
</p>

### Usage:
#### Setup and Dependencies

This code is implemented using PyTorch v1.7, and provides out of the box support with CUDA 11 and CuDNN 8. Anaconda/Miniconda is the recommended to set up this codebase: <br>

1. Install Anaconda or Miniconda distribution based on Python3+.
2. Clone this repository and create an environment:

```shell
conda create -n HGDI python=3.8
conda activate HGDI
cd HGDI-visdial/
pip install -r requirements.txt
```

#### Download Data

1. Download the image features below, and put each feature under `$PROJECT_ROOT/data/{SPLIT_NAME}_feature` directory.

  * [`train_btmup_f.hdf5`][2]: Bottom-up features of 10 to 100 proposals from images of `train` split (32GB). 
  * [`val_btmup_f.hdf5`][3]: Bottom-up features of 10 to 100 proposals from images of `validation` split (0.5GB).
  * [`test_btmup_f.hdf5`][4]: Bottom-up features of 10 to 100 proposals from images of `test` split (2GB).

2. Download the pre-trained, pre-processed word vectors from [here][5] (`glove840b_init_300d.npy`), and keep them under `$PROJECT_ROOT/data/` directory. You can manually extract the vectors by executing `data/init_glove.py`.

3. Download visual dialog dataset from [here][6] (`visdial_1.0_train.json`, `visdial_1.0_val.json`, `visdial_1.0_test.json`, and `visdial_1.0_val_dense_annotations.json`) under `$PROJECT_ROOT/data/` directory.


#### Training

Train the model provided in this repository as:

```shell
python train.py --gpu-ids 0 1 
```

##### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`. Default path is `$PROJECT_ROOT/checkpoints`.

#### Evaluation

Evaluation of a trained model checkpoint can be done as follows:

```shell
python evaluate.py --load-pthpath /path/to/checkpoint.pth --split val --gpu-ids 0 1
```

### License

MIT License

### Acknowledgements

We use [Visual Dialog Challenge Starter Code][7] and [MCAN-VQA][8] as reference code.   

### Citation:

Please cite the following paper if you find this work useful:

S. Du, H. Wang, T. Li, and C. W. Chen, “Hybrid graph reasoning with dynamic interaction for visual dialog,” IEEE Trans. Multimedia., vol. 26, pp. 9095–9108, Apr. 2024.

[1]: https://conda.io/docs/user-guide/install/download.html
[2]: https://drive.google.com/file/d/1NYlSSikwEAqpJDsNGqOxgc0ZOkpQtom9/view?usp=sharing
[3]: https://drive.google.com/file/d/1NI5TNKKhqm6ggpB2CK4k8yKiYQE3efW6/view?usp=sharing
[4]: https://drive.google.com/file/d/1BXWPV3k-HxlTw_k3-kTV6JhWrdzXsT7W/view?usp=sharing
[5]: https://drive.google.com/file/d/125DXSiMwIH054RsUls6iK3kdZACrYodJ/view?usp=sharing
[6]: https://visualdialog.org/data
[7]: https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
[8]: https://github.com/MILVLG/mcan-vqa

