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

        """ DWT dual-branch options """
        self._add_bool_argument('--use_dual_branch', default=False, help='Enable DWT dual-branch FSD. ')
        self.parser.add_argument('--freq_input_type', type=str, default='dwt', choices=('dwt', 'rgb'),
                                 help='Frequency branch input type. V1 implements dwt only; rgb is reserved. ')
        self.parser.add_argument('--freq_stats_path', type=str, default=None,
                                 help='Path to train-set DWT mean/std JSON. Required for freq_input_type=dwt. ')
        self.parser.add_argument('--dwt_input_scale', type=float, default=1.0,
                                 help='Scale raw ToTensor input before DWT. Default keeps [0, 1] scale. ')
        self._add_bool_argument('--dwt_use_abs', default=True, help='Use absolute DWT high-frequency coefficients. ')
        self._add_bool_argument('--dwt_use_log1p', default=True, help='Apply log1p to DWT high-frequency tensor. ')
        self.parser.add_argument('--freq_norm_eps', type=float, default=1e-8,
                                 help='Epsilon for DWT frequency mean/std normalization. ')
        self._add_bool_argument('--use_aux_loss', default=True, help='Use RGB and frequency auxiliary losses. ')
        self.parser.add_argument('--lambda_rgb', type=float, default=0.2, help='RGB auxiliary loss weight. ')
        self.parser.add_argument('--lambda_freq', type=float, default=0.2, help='Frequency auxiliary loss weight. ')
        self._add_bool_argument('--normalize_reliability_variance', default=True,
                                help='Normalize support variance before reliability softmax. ')
        self.parser.add_argument('--reliability_norm_mode', type=str, default='episode_mean',
                                 help='Reliability variance normalization mode. V1 supports episode_mean only. ')
        self.parser.add_argument('--reliability_temperature', type=float, default=1.0,
                                 help='Fixed softmax temperature for reliability fusion. ')
        self.parser.add_argument('--reliability_eps', type=float, default=1e-8,
                                 help='Epsilon for reliability variance normalization. ')
        self._add_bool_argument('--detach_reliability', default=True,
                                help='Detach reliability weights alpha from the gradient graph. ')
        self._add_bool_argument('--clip_reliability_weight', default=True,
                                help='Clip alpha_rgb and set alpha_freq to 1 - alpha_rgb. ')
        self.parser.add_argument('--alpha_min', type=float, default=0.1, help='Minimum alpha_rgb when clipping. ')
        self.parser.add_argument('--alpha_max', type=float, default=0.9, help='Maximum alpha_rgb when clipping. ')
        self.parser.add_argument('--tb_log_interval', type=int, default=20,
                                 help='TensorBoard interval for dual-branch debug scalars. ')
    
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

    def _add_bool_argument(self, name, default=False, help=None):
        self.parser.add_argument(
            name,
            default=default,
            action=argparse.BooleanOptionalAction,
            help=help,
        )


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
