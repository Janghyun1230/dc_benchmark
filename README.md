# DCBench

This is the official codebase for paper: DC-BENCH: Dataset Condensation Benchmark.

# How to set up
## Step 1
Run the following command to download the library to local
```
git clone git@github.com:justincui03/dc_benchmark.git
```
## Step 2
Download all(or part) of the data from [this shared link](https://drive.google.com/drive/folders/1trp0MyUoL9QrbsdQ8w7TxgoXcMJecoyH?usp=sharing)

### Step 3
Run the following command to create a conda environment
```
cd dc_benchmark
conda env create -f environment.yml
```

# Run an evaluation
Running an evaluation is very simple with DCBench
```
bash eval.sh --dataset CIFAR10 --ipc 1 --model convnet --aug autoaug
```
Here are the configurable parameters
- dataset: choose between CIFAR10, CIFAR100 and tinyimagenet
- ipc: choose between 1, 10 and 50
- model: choose between convnet, convnet4, mlp, resnet18, resnet152, etc
- aug: choose between autoaug, randaug, imagenet_aug, dsa,etc

# Neural Architecture Search
We incorparete the standard NAS-Bench-201 library into our codebase.
You can start a simple NAS experiment by running the following command
```
cd darts-pt
bash darts-201.sh
```
For the detailed setup, please refer to [DARTS-PT
](https://github.com/ruocwang/darts-pt)
