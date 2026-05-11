# -*- coding: utf-8 -*-
""" Parser used by this program. 
"""

import argparse

class ModelParser():
    """ Setup base parser. 
    """

    def __init__(self): 
        self.parser = argparse.ArgumentParser()

        """ base options """
        self.parser.add_argument('--model', type=str, default='resnet50', help='Name of model. ')
        self.parser.add_argument('--output_dir', type=str, default='./output_dir', help='The directory for saving checkpoints and logs. ')
        
        """ dataloader options """
        self.parser.add_argument('--data_root', type=str, default='./data', help='Root of dataset. ')
        self.parser.add_argument('--num_workers', type=int, default=8,  help='Number of workers in dataloader. ')

        self.parser.add_argument('--seed', type=int, default=None, help='Random seed for the main processes. ')

        """ prototypical distance options """
        self.parser.add_argument('--distance_type', type=str, default='squared_euclidean',
                                 help='Distance type: squared_euclidean keeps original FSD; graph uses graph or mixed distance. ')
        self.parser.add_argument('--graph_alpha', type=float, default=1.0,
                                 help='Graph distance weight in graph mode. 1.0 means pure graph; 0.3 means 30% graph + 70% squared Euclidean. ')
        self.parser.add_argument('--graph_edge_weight', type=str, default='squared_euclidean',
                                 help='Graph edge weight type. V1 only supports squared_euclidean. ')
        self.parser.add_argument('--distance_norm', type=str, default='mean',
                                 help='Distance normalization for graph mode. V1 supports mean or none. ')
        self.parser.add_argument('--graph_mode', type=str, default='label_aware_global',
                                 help='Graph construction mode. V1 only supports label_aware_global. ')
        self.parser.add_argument('--graph_k', type=int, default=3,
                                 help='Number of same-class support neighbors for support-support edges. ')
        self.parser.add_argument('--graph_query_k_global', type=int, default=3,
                                 help='Number of globally nearest support nodes connected to each query. ')
        self.parser.add_argument('--graph_query_min_per_class', type=int, default=1,
                                 help='Minimum query-support edges kept for each class. ')
        self.parser.add_argument('--graph_fallback', type=str, default='squared_euclidean',
                                 help='Fallback distance when graph path is unreachable. V1 only supports squared_euclidean. ')
        self.parser.add_argument('--transductive', type=self._str2bool, default=False,
                                 help='Whether to put multiple queries into one graph. V1 only supports False. ')
    
    @property
    def args(self): 
        return self.parser.parse_args()
    
    @staticmethod
    def _str2bool(value): 
        if value.lower() in ('yes', 'true', 't', 'y', '1'): 
            return True
        elif value.lower() in ('no', 'false', 't', 'y', '0'): 
            return False
        else: 
            raise argparse.ArgumentTypeError('Unsupported value encountered. ')


class TrainParser(ModelParser): 
    def __init__(self): 
        super().__init__()
        """ optimizer options """
        self.parser.add_argument('--use_fp16', type=self._str2bool, default=True, help='Use fp16 when training. ')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate for training. ')
        self.parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5, help='Gamma in StepLR. ')
        self.parser.add_argument('--lr_scheduler_step', type=int, default=15000, help='Scheduler step in StepLR. ')

        """ training options """
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size of tasks. ')
        self.parser.add_argument('--num_class_train', type=int, default=3, help='Number of classes per iteration during training. ')
        self.parser.add_argument('--num_support_train', type=int, default=5, help='Number of samples in support set of each class during training. ')
        self.parser.add_argument('--num_query_train', type=int, default=5, help='Number of samples in query set of each class during training. ')

        self.parser.add_argument('--num_class_val', type=int, default=2, help='Number of classes per iteration during training. ')
        self.parser.add_argument('--num_support_val', type=int, default=5, help='Number of samples in support set of each class during validation. ')
        self.parser.add_argument('--num_query_val', type=int, default=15, help='Number of samples in query set of each class during validation. ')

        self.parser.add_argument('--exclude_class', type=str, default="ADM", help='Image folder to be tested. ')
        
        self.parser.add_argument('--total_training_steps', type=int, default=40000, help='Total steps for training. ')
        self.parser.add_argument('--accumulation_steps', type=int, default=1, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints). ')
        self.parser.add_argument('--save_interval', type=int, default=5000, help='Interval between saving model weights. ')
        self.parser.add_argument('--log_interval', type=int, default=500, help='Interval between logs. ')
        self.parser.add_argument('--eval_interval', type=int, default=5000, help='Interval between logs. ')

        # add other options if you want


class TestParser(ModelParser): 
    def __init__(self): 
        super().__init__()
        self.parser.add_argument('--use_fp16', type=self._str2bool, default=True, help='Use fp16 when training. ')
        self.parser.add_argument('--test_class', type=str, default="ADM", help='Image folder to be tested. ')
        self.parser.add_argument('--ckpt_path', type=str, default="./output_dir/ckpt/resnet50_50000.pth", help='Path to the trained checkpoint. ')


        """ testing options """
        self.parser.add_argument('--num_class_test', type=int, default=2, help='Number of classes per iteration during training. ')
        self.parser.add_argument('--num_support_test', type=int, default=5, help='Number of samples in support set of each class during training. ')
        self.parser.add_argument('--num_query_test', type=int, default=15, help='Number of samples in query set of each class during training. ')
        # add other options if you want
