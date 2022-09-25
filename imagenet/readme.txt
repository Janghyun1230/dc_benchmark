* Before run the code set --imagenet_dir in argument.py

# To run IDC (set -f 3)
python run.py --nclass 10 --ipc 10 -f 3

# To run IDC-I (set -f 1)
python run.py --nclass 10 --ipc 10 -f 1

# Other arguments
* -s random (for random selection)
* --nclass 100 (for ImageNet-100)
* --repeat 3 (repeat train/evaluation 3 times)
* -n (network types: resnet, resnet_ap, efficientnet)
* --depth (network depth)
* set --verbose and --print-freq [int] for printing log