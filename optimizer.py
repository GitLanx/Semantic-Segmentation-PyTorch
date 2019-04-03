import torch
import torch.nn as nn


def get_optimizer(args, model):
    if args.optim.lower() == 'sgd':
        if args.model.lower() == 'deeplab-largefov':
            optim = torch.optim.SGD(
                [{'params': get_parameters(args, model, bias=False, final=False)},
                 {'params': get_parameters(args, model, bias=True, final=False), 'lr': args.lr * 2, 'weight_decay': 0},
                 {'params': get_parameters(args, model, bias=False, final=True), 'lr': args.lr * 10},
                 {'params': get_parameters(args, model, bias=True, final=True), 'lr': args.lr * 20, 'weight_decay': 0}],
                lr=args.lr,
                momentum=args.beta1,
                weight_decay=args.weight_decay)
        elif args.model.lower() in ['fcn32s', 'fcn8s']:
            optim = fcn_optim(model, args)
        else:
            optim = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.beta1,
                weight_decay=args.weight_decay)

    elif args.optim.lower() == 'adam':
        optim = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, 0.999),
            weight_decay=args.weight_decay)
    
    return optim

def fcn_optim(model, args):
    """optimizer for fcn32s and fcn8s
    """
    optim = torch.optim.SGD(
        [{'params': model.get_parameters(bias=False)},
         {'params': model.get_parameters(bias=True), 'lr': args.lr * 2, 'weight_decay': 0}],
         lr=args.lr,
         momentum=args.beta1,
         weight_decay=args.weight_decay)
    return optim


def get_parameters(args, model, bias=False, final=False):
    """Adapted from:

    https://github.com/BardOfCodes/pytorch_deeplab_large_fov/blob/master/utils.py
    """
    if final:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == args.n_classes:
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if not m.out_channels == args.n_classes: 
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight
