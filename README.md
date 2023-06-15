# Adversarial Cheap Talk

Code for Adversarial Cheap Talk (ICML 2023)

Lu, Chris, Timon Willi, Alistair Letcher, and Jakob Foerster. "Adversarial Cheap Talk."

[Paper](https://arxiv.org/abs/2211.11030)

[Webpage](https://sites.google.com/view/adversarial-cheap-talk/home)

Due to the rapid development of JAX's ecosystem it can be difficult for users to precisely set up the environment. We *highly* recommend instead using the [PureJaxRL repository](https://github.com/luchris429/purejaxrl/tree/main) to perform related research. We plan to upload a clean re-implementation of this work there. This repository is for reproducing the original results in the paper.

PureJaxRL is similar to this repository in that it contains end-to-end Jax-vectorised PPO implementations. However, it differs from this repository in many ways -- it uses newer libraries that did not exist at the time that the bulk of this research was performed. In particular, [Gymnax](https://github.com/RobertTLange/gymnax/tree/main) vastly simplifies the 

# Installation

`pip install -r requirements.txt`

To install Jax with cuda, run

`pip install jax==0.3.0 -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"`

`pip install jaxlib==0.3.0 -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"`

# Usage

To train a train-time adversarial cheap talk channel against a CartPole PPO agent with two dimensions, run:

`python3 main_train_time.py --rew_type="ANTI" --normalize-obs --end-only --env="CARTPOLE" --n_dims=2`

To train a train/test-time exploiting adversarial cheap talk channel against a CartPole PPO agent, run:

`python3 main_test_time.py --static --zero-shot --normalize-obs --env="CARTPOLE" --n_dims=2`

#  Citation

```
@article{lu2022adversarial,
  title={Adversarial Cheap Talk},
  author={Lu, Chris and Willi, Timon and Letcher, Alistair and Foerster, Jakob},
  journal={arXiv preprint arXiv:2211.11030},
  year={2022}
}
```