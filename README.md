# Review of semi-amortized Variational Autoencoders methods

## Usage

### Training

```
python run.py --checkpoint_path checkpoints/<CHECKPOINT_NAME> --model <MODEL_NAME> --dataset <DATASET>
```

Available models: `vae`, `savae`, `cdiv`, `cdiv_svgd`

Available datasets: `mnist`, `fashion_mnist`


If you would like to use the OMNIGLOT dataset, please donwload data from https://github.com/harvardnlp/sa-vae and run
```
python data/preprocess_img.py --raw_file data/omniglot/chardata.mat --output data/omniglot/omniglot.pt
```
Then you can also experiment with this dataset by setting
```
--dataset omniglot --data_path data/omniglot/omniglot.pt

```

#### Parameters

To see a list of changeable parameters please run:
```
python run.py --help
```

For example, you can adapt number of refinement iterations: `--svi_steps`, `--num_hmc_iters`, `--num_svgd_iters`

### Evaluation

```
python run.py --test 1 --train_from checkpoints/<CHECKPOINT_NAME>_<TYPE>.pt --model <MODEL_NAME> --test_type <TEST_TYPE>  --dataset <DATASET>
```

Available checkpoint types: "last" (last stored model), "best" (best validation score model)

Available models: `vae`, `savae`, `cdiv`, `cdiv_svgd`

Available test types: `vae`, `savae`, `cdiv`, `cdiv_svgd`

Available datasets: `mnist`, `fashion_mnist`

##### Note
If you would like to change the default number of HMC or SVGD refinement iterations or the number of SVGD particles during the evaluation please modify the `importance sampling.py` file.
The parameters passed during the training do not change the evaluation procedure for the CDIV and SVGD VAE.

### Example

Training of a Contrastive Divergence VAE on the Fashion MNIST dataset:
```
python run.py --checkpoint_path checkpoints/cdiv_checkpoint --model cdiv --dataset fashion_mnist

```
Evaluation of the last stored model using the HMC Gaussian fit importance sampling procedure:
```
python run.py --test 1 --train_from checkpoints/cdiv_checkpoint_last.pt --model cdiv --test_type cdiv  --dataset fashion_mnist
```

### Requirements
`pytorch 1.5`
`torchvision`
`pillow`
`matplotlib`
`scipy`
`wandb`

## Acknowledgements

Semi-Amortized VAE and the code base adapted from https://github.com/harvardnlp/sa-vae 
which provides code for the paper [Semi-Amortized Variational Autoencoders](https://arxiv.org/abs/1802.02550) 
by Yoon Kim, Sam Wiseman, Andrew Miller, David Sontag, Alexander Rush.

Contrastive Divergence VAE adapted from https://github.com/franrruiz/vcd_divergence
which provides code for the paper [A Contrastive Divergence for Combining Variational Inference and MCMC](https://arxiv.org/abs/1905.04062) 
by Francisco J. R. Ruiz and Michalis K. Titsias.

The project uses [Weights and Biases](https://www.wandb.com/) for experiment tracking.
