import os
import sys
sys.path.append('..')
from constants import DATA_DIR

from evaluator_utils import EvaluatorUtils
from networks.network_utils import NetworkUtils
import argparse

class Evaluator:

    def __init__(self):
        pass

    def load_data(self, data_dir, data_file, args):
        data_loader = EvaluatorUtils.get_data_loader(args.method)
        train_image, train_label = data_loader.load_data(data_dir, args.dataset, args.ipc, data_file)
        dst_test = EvaluatorUtils.get_testset(args.dataset, True)
        return train_image, train_label, dst_test

    
    def evaluate(self, eval_models):
        '''
        do the acutual evaluation
        '''
        per_model_accuracy = {}
        for model_name in eval_models:
            model = NetworkUtils.create_network(model_name)
            per_model_accuracy[model_name] = EvaluatorUtils.evaluate_synset(0, model, self.input_images, self.input_labels, self.test_dataset, args)
        return per_model_accuracy

    def print(per_model_accuracy):
        '''
        print out the evaluation result.
        '''
        for model, accuracy in per_model_accuracy.items():
            print("model: %s , accuracy %.2f", model, accuracy)
        
def prepare_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_file', type=str, default='data', help='dataset path')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    args = parser.parse_args()
    args.dsa = False
    args.device = 'cuda'
    return args

if __name__ == '__main__':
    args = prepare_args()
    evaluator = Evaluator()
    data_file = EvaluatorUtils.get_data_file_name(args.method, args.dataset, args.ipc)
    evaluator.load_data(DATA_DIR, data_file, args)
    evaluator.evaluate()

