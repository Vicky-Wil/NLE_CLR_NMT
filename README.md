# cyclic learning rate for NMT
In training deep learning networks, the optimizer and related learning rate are often used without much thought or with minimal tuning, even though it is crucial in ensuring a fast convergence to a good quality minimum of the loss function that can also generalize well on the test dataset. Drawing inspiration from the successful application of cyclical learning rate policy to computer vision tasks, we explore how cyclical learning rate can be applied to train transformer-based neural networks for neural machine translation.

This repository includes all the code extending our initial implementation of cyclic learning rate for NMT, mainly based on the following two open source code
 - [loss-landscape](https://github.com/tomgoldstein/loss-landscape) 
 - [fairSeq](https://github.com/pytorch/fairseq)

**Dependency**: One GPU with the following software/libraries installed:
- [PyTorch 1.3.1](https://pytorch.org/)
- [Fairseq 0.6.2](https://github.com/pytorch/fairseq)
- [openmpi 3.1.2](https://www.open-mpi.org/)    (optional)
- [mpi4py 2.0.0](https://mpi4py.scipy.org/docs/usrman/install.html)       (optional)
- [numpy 1.17.4](https://docs.scipy.org/doc/numpy/user/quickstart.html)  
- [h5py 2.7.0](http://docs.h5py.org/en/stable/build.html#install)
- [matplotlib 3.1.2](https://matplotlib.org/users/installing.html)
- [scipy 0.19](https://www.scipy.org/install.html)
## Data
The experience data can be download from [data](https://drive.google.com/drive/folders/1DYybKED5AOC43I3ce_hPMN5pZXwoW1ut?usp=sharing)
## Script
You can find all experience scripts in **./script** folder

For the range test, you can run range_test_iwslt14-de-en.sh

For the training of IWSLT14-de2en, you can run the train_iwslt14-de-en_transformer.sh, after training you can calculate the BLEU score with running the cal_bleu-iwlst14-de-en.sh

For the training of IWSLT14-fr2en, you can run the train_iwslt14-fr-en_trainsformer.sh, after training you can calculate the BLEU score with running the cal_bleu-iwlst17-de-en.sh

For the training of IWSLT17-de2en, you can run the train_iwslt17-de-en_trainsformer.sh, after training you can calculate the BLEU score with running the cal_bleu-iwlst14-fr-en.

If you want to verify the impact of different batch size on the results, you can run the batch-experiments.sh
## Visualization
The three ipynb files are used for loss/bleu value visualization.

If you want to visualize the loss of valid, you can use the valid_loss_visualisation.ipynb

If you want to visualize the learning rate of range test, you can use the range_test_visualization.ipynb 

If you want to visualize the trend of BLEU score, you can use the bleu_visualization.ipynb after calculating the BLEU score

If you want to visualize the results of different batch size, you can use the batch-exp.ipynb


# Citation
Please cite as:

```
@inproceedings{CLR_NMT,
  author    = {Weixuan Wang and
               Choon Meng Lee and
               Jianfeng Liu and 
               Talha Colakoglu and
               Wei Peng},
  title     = {An empirical study of cyclical learning rate on neural machine translation},
  journal   = {Natural Language Engineering},
  booktitle = {Cambridge University Press},
  pages     = {1-21},
  year      = {2022},
  DOI       = {10.1017/S135132492200002X},
```
