import torch
import torch.nn as nn


def get_optimizer(args, model):
    """Optimizer for different models
    """
    if args.optim.lower() == 'sgd':
        if args.model.lower() in ['fcn32s', 'fcn8s']:
            optim = fcn_optim(model, args)
        elif args.model.lower() in ['deeplab-largefov', 'deeplab-aspp-vgg']:
            optim = deeplab_optim(model, args)
        elif args.model.lower() in ['deeplab-aspp-resnet']:
            optim = deeplabv2_optim(model, args)
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

def deeplab_optim(model, args):
    """optimizer for deeplab-v1 and deeplab-v2-vgg
    """
    optim = torch.optim.SGD(
        [{'params': model.get_parameters(bias=False, score=False)},
         {'params': model.get_parameters(bias=True, score=False), 'lr': args.lr * 2, 'weight_decay': 0},
         {'params': model.get_parameters(bias=False, score=True), 'lr': args.lr * 10},
         {'params': model.get_parameters(bias=True, score=True), 'lr': args.lr * 20, 'weight_decay': 0}],
         lr=args.lr,
         momentum=args.beta1,
         weight_decay=args.weight_decay)
    return optim

def deeplabv2_optim(model, args):
    """optimizer for deeplab-v2-resnet
    """
    optim = torch.optim.SGD(
        [{'params': model.get_parameters(bias=False, score=False)},
         {'params': model.get_parameters(bias=False, score=True), 'lr': args.lr * 10},
         {'params': model.get_parameters(bias=True, score=True), 'lr': args.lr * 20, 'weight_decay': 0}],
         lr=args.lr,
         momentum=args.beta1,
         weight_decay=args.weight_decay)
    return optim