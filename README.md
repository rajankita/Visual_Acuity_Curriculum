# [ICVGIP 2025] Mimicking Human Visual Development for Learning Robust Image Representations

This repository contains the official implementation of the paper accepted at **ICVGIP 2025**.

<p align="left">
   📖 <a href="https://arxiv.org/" target="_blank">Paper</a>
   ▶️ <a href="https://www.youtube.com/watch?v=Xq7oARw1LEI" target="_blank">Short Video</a>
</p>


<p align="center">
   <img width="800" alt="image" src="imgs/method_diagram_cats.png">
</p>

## Abstract
The human visual system is remarkably adept at adapting to changes in the input distribution — a capability modern convolutional neural networks (CNNs) still struggle to match. Drawing inspiration from the developmental trajectory of human vision, we propose a progressive blurring curriculum to improve the generalization and robustness of CNNs. Human infants are born with poor visual acuity, gradually refining their ability to perceive fine details. Mimicking this process, we begin training CNNs on highly blurred images during the initial epochs and progressively reduce the blur as training advances. This approach encourages the network to prioritize global structures over high-frequency artifacts, improving robustness against distribution shifts and noisy inputs. Challenging prior claims that blurring in the initial training epochs imposes a stimulus deficit and irreversibly harms model performance, we reveal that early-stage blurring enhances generalization with minimal impact on in-domain accuracy. Our experiments demonstrate that the proposed curriculum reduces mean corruption error (mCE) by up to 8.30% on CIFAR-10-C and 4.43% on ImageNet-100-C datasets, compared to standard training without blurring. Unlike static blur-based augmentation, which applies blurred images randomly throughout training, our method follows a structured progression, yielding consistent gains across various datasets. Furthermore, our approach complements other augmentation techniques, such as CutMix and MixUp, and enhances both natural and adversarial robustness against common attack methods.

## Quick Start

We recommend using conda. Create a new virtual environment with conda:

```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate vac
```
The code has been tested with Ubuntu 20.04 and NVIDIA Tesla V100.

## Training Scripts
Train ResNet-18 on CIFAR-10 using vanilla training:
```
python train_vac.py --cfg configs/cifar10_preactresnet18_vanilla.yaml
```

Train ResNet-18 on CIFAR-10 using VAC:
```
python train_vac.py --cfg configs/cifar10_preactresnet18_vac.yaml
```

The script `train_vac.py` also supports 
- Data augmentations like CutMix, MixUp, RandAugment, and AutoAugment
- Other training strategies like [SuperLoss](https://github.com/AlanChou/Super-Loss)
- Variants of the curriculum like Constant Blur, No Replay, Linear Curriculum, and Inverse Curriculum

These can be achieved by setting the appropriate input arguments. Some samples are provided in the `configs/` directory.

Some other supported training scripts are:
- `train_fixres_finetune.py` - Train using [FixRes](https://github.com/facebookresearch/FixRes)
- `train_vac_continuous.py` - Train using a continuous curriculum

## Evaluation Scripts

To evaluate trained models on clean and corrupted dataset, run ```test_corruption.py```. 

```
python test_corruption.py --net_type preactresnet18 --dataset cifar10 --pretrained <model_path>

```

To evaluate on adversarial attacks, run `test_adversarial.py`:
```
python test_adversarial.py --net_type preactresnet18 --dataset cifar10 --pretrained <model_path>
```
> **Note:** Exact numbers may vary slightly due to random seeds and hardware differences.

## Citation
If you find this work useful, please cite us:
<!-- 
```bibtex
@misc{kim2024demonstrationadaptivecollaborationlarge,
      title={Mimicking Human Visual Development for Learning Robust Image Representations}, 
      author={Yubin Kim and Chanwoo Park and Hyewon Jeong and Cristina Grau-Vilchez and Yik Siu Chan and Xuhai Xu and Daniel McDuff and Hyeonhoon Lee and Marzyeh Ghassemi and Cynthia Breazeal and Hae Won Park},
      year={2024},
      eprint={2411.00248},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.00248}, 
}
``` -->

## Contact
Ankita Raj (ankita.raj@cse.iitd.ac.in)

## Acknowledgement
Our implementation is based on the following repositories:
- [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet)
- [CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch)
