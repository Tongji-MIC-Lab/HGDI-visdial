HGDI-visdial
====================================

Setup and Dependencies
----------------------
This code is implemented using PyTorch v1.7, and provides out of the box support with CUDA 11 and CuDNN 8. Anaconda/Miniconda is the recommended to set up this codebase: <br>

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site][1].
2. Clone this repository and create an environment:

```shell
conda create -n HGDI python=3.8
conda activate HGDI
cd HGDI-visdial/
pip install -r requirements.txt
```

Download Data
----------------------
1. Download the image features below, and put each feature under `$PROJECT_ROOT/data/{SPLIT_NAME}_feature` directory.

  * [`train_btmup_f.hdf5`][2]: Bottom-up features of 10 to 100 proposals from images of `train` split (32GB). 
  * [`val_btmup_f.hdf5`][3]: Bottom-up features of 10 to 100 proposals from images of `validation` split (0.5GB).
  * [`test_btmup_f.hdf5`][4]: Bottom-up features of 10 to 100 proposals from images of `test` split (2GB).

2. Download the pre-trained, pre-processed word vectors from [here][5] (`glove840b_init_300d.npy`), and keep them under `$PROJECT_ROOT/data/` directory. You can manually extract the vectors by executing `data/init_glove.py`.

3. Download visual dialog dataset from [here][6] (`visdial_1.0_train.json`, `visdial_1.0_val.json`, `visdial_1.0_test.json`, and `visdial_1.0_val_dense_annotations.json`) under `$PROJECT_ROOT/data/` directory.


Training
--------

Train the model provided in this repository as:

```shell
python train.py --gpu-ids 0 1 
```

### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`. Default path is `$PROJECT_ROOT/checkpoints`.

Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```shell
python evaluate.py --load-pthpath /path/to/checkpoint.pth --split val --gpu-ids 0 1
```

License
--------
MIT License

Acknowledgements
--------
We use [Visual Dialog Challenge Starter Code][7] and [MCAN-VQA][8] as reference code.   

[1]: https://conda.io/docs/user-guide/install/download.html
[2]: https://drive.google.com/file/d/1NYlSSikwEAqpJDsNGqOxgc0ZOkpQtom9/view?usp=sharing
[3]: https://drive.google.com/file/d/1NI5TNKKhqm6ggpB2CK4k8yKiYQE3efW6/view?usp=sharing
[4]: https://drive.google.com/file/d/1BXWPV3k-HxlTw_k3-kTV6JhWrdzXsT7W/view?usp=sharing
[5]: https://drive.google.com/file/d/125DXSiMwIH054RsUls6iK3kdZACrYodJ/view?usp=sharing
[6]: https://visualdialog.org/data
[7]: https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
[8]: https://github.com/MILVLG/mcan-vqa
