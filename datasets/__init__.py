# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data

from .mmnist import build as build_mmnist

def build_dataset(split, args):
	if args.dataset_file == 'mmnist':
		return build_mmnist(split, args)
	raise ValueError(f'dataset {args.dataset_file} not supported')
