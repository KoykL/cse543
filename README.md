# Optimizing Card Game Dou Di Zhu

We build a Dou Di Zhu game and implement the IS-MCTS deep learning method. The network demonstrates some reasonable thinking and manages to compete with the previous method as proposed by AlphaZero.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

First clone the project:
```
git clone https://github.com/KoykL/cse543.git
```
Then download the requirements and run the setup file.
```
pip3 install Cython
```
Note that you should download the PyTorch package on [their official website](http://pytorch.org).

The C++ code should be compiled with C++11 standard.

Then run the following command to start the game:
```
python3 doudizhu.py
```

Flags:
- `-t` for training.
- `--human [player index]` for human player.

## Authors

* **Jiaye Wu**
* **Han Liu**
* **Zongyi Li**

## Acknowledgments

* This is a course project of [CSE543](https://www.cse.wustl.edu/~ychen/CSE543/) at Washington University in St. Louis.

