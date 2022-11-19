import os
import time
import matplotlib
matplotlib.use('Agg')
from progress.bar import Bar
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import models
from models.fan_model import  FAN
from datasets import W300LP, VW300, AFLW2000, LS3DW, LS3DW_F, BLD4VR, LFPW , CelebA
from utils.logger import Logger, savefig
from utils.imutils import batch_with_heatmap
from utils.evaluation import accuracy, AverageMeter, final_preds, calc_metrics, calc_dists
from utils.misc import adjust_learning_rate, save_checkpoint, save_pred
import opts

args = opts.argparser()
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
# torch.setdefaulttensortype('torch.FloatTensor')

num_lmks_train = 71
num_lmks_already_trained = 68
best_acc = 0.
best_auc = 0.
idx = range(1, num_lmks_train, 1)


def get_loader(data):
    return {
        '300W_LP': W300LP,
        'LFPW': LFPW,
        '300W_LP_test': W300LP,
        'LS3D-W/300VW-3D': VW300,
        'AFLW2000': AFLW2000,
        'LS3D-W': LS3DW_F,
        'LS3D-W-Test': LS3DW,
        'BALD_4VR': BLD4VR,
        'BALD_4VR_test': BLD4VR,
        'BALD7071': BLD4VR,
        'BALD7071_test': BLD4VR,
        'CelebA' : CelebA,

    }[os.path.basename(data)]


def main(args):
    global best_acc
    global best_auc

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    print("==> Creating model '{}-{}', stacks={}, blocks={}, feats={}".format(
        args.netType, args.pointType, args.nStacks, args.nModules, args.nFeats))

    print("=> Models will be saved at: {}".format(args.checkpoint))

    # model = models.__dict__[args.netType](
    #    num_stacks=args.nStacks,
    #    num_blocks=args.nModules,
    #    num_feats=args.nFeats,
    #    use_se=args.use_se,
    #    use_attention=args.use_attention,
    #    num_classes=68)
    
    model = FAN(2 , 73)

    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(size_average=True).cuda()

    # optimizer = torch.optim.RMSprop(
    #     model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    title = args.checkpoint.split('/')[-1] + ' on ' + args.data.split('/')[-1]


    if args.data_loader != '':
          train_Loader = get_loader(args.data_loader)
    else:
          train_Loader = get_loader(args.data)

    if args.val_loader != '':
          val_Loader = get_loader(args.val_loader)
    else:
          val_Loader = get_loader(args.val_data)

    print(val_Loader)
    print(train_Loader)

    val_loader = torch.utils.data.DataLoader(
        val_Loader(args, 'val'),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.initialize:
        print("initialising")
        if os.path.isfile(args.initialize):
            print("=> Loading checkpoint '{}'".format(args.initialize))
            checkpoint = torch.load(args.initialize)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)

            # own_state = model.state_dict()

            # for k,param in pretrained_dict.items():
            #   if k in ["module.l0.weight" , "module.l0.bias" , "module.al0.weight" , "module.l1.weight" , "module.l1.bias" ] :
            #     continue
            #   else:
            #     v = param.data
            #     own_state[k].copy_(v)


            print("=> Loaded checkpoint '{}' (epoch {})".format(args.initialize, checkpoint['epoch']))
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Epoch', 'LR', 'Train Loss', 'Valid Loss', 'Train Acc', 'Val Acc', 'AUC'])
            print("initialized weights")

        else:
            print("=> no checkpoint found at '{}'".format(args.initialize))        

    if args.resume:
        print("resuming")
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            logger = Logger(os.path.join(os.path.dirname(args.resume), 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Valid Loss', 'Train Acc', 'Val Acc', 'AUC'])

    cudnn.benchmark = True
    print('=> Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / (1024. * 1024)))

    if args.evaluation:
        print('=> Evaluation only')
        D = args.data.split('/')[-1]
        save_dir = os.path.join(args.checkpoint, D)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        loss, acc, predictions, auc = validate(val_loader, model, criterion, args.netType,
                                                        args.debug, args.flip,num_lmks=num_lmks_train)
        save_pred(predictions, checkpoint=save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        train_Loader(args, 'train'),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

        
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('=> Epoch: %d | LR %.8f' % (epoch + 1, lr))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, args.netType,
                                      args.debug, args.flip)
        # do not save predictions in model file
        valid_loss, valid_acc, predictions, valid_auc = validate(val_loader, model, criterion, args.netType,
                                                      args.debug, args.flip,num_lmks=num_lmks_train)

        logger.append([int(epoch + 1), lr, train_loss, valid_loss, train_acc, valid_acc, valid_auc])

        is_best = valid_auc >= best_auc
        best_auc = max(valid_auc, best_auc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'netType': args.netType,
                'state_dict': model.state_dict(),
                'best_acc': best_auc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            predictions,
            checkpoint=args.checkpoint,
            snapshot=args.snapshot)

    logger.close()
    logger.plot(['AUC'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(loader, model, criterion, optimizer, netType, debug=False, flip=False ):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.train()
    end = time.time()

    # rnn = torch.nn.LSTM(10, 20, 2)
    # hidden = torch.autograd.Variable(torch.zeros((args.train_batch)))

    gt_win, pred_win = None, None
    bar = Bar('Training', max=len(loader))
    for i, (inputs, target) in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(target.cuda(non_blocking=True))
        #print("target_var ", target_var.shape)

        #print("input_var ", input_var.shape)
        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            # pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                # plt.subplot(122)
                # pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                # pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        output = model(input_var)
        # last one of output is score_map
        score_map = output[-1].data.cpu()

        #print("score_map ", score_map.shape)

        if flip:
            flip_input_var = torch.autograd.Variable(
                torch.from_numpy(shufflelr(inputs.clone().numpy())).float().cuda())
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output

        # intermediate supervision
        # we have multi score-maps

        loss = 0
        for o in output:
            loss += 10*criterion(o[:,num_lmks_already_trained:num_lmks_train,:,:], target_var[:,num_lmks_already_trained:num_lmks_train,:,:] ) + criterion(o[:,:num_lmks_already_trained,:,:], target_var[:,:num_lmks_already_trained,:,:] )
        # for o in output:
        #     loss += criterion(o[:,:68,:,:], target_var)


        acc, _ = accuracy(score_map[:,:num_lmks_train,:,:], target.cpu(), idx, thr=0.07)

        
        

        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # loss backforward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()
    # print("loss " ,losses.avg)
    # print("acc " ,acces.avg)
    

    return losses.avg, acces.avg


def validate(loader, model, criterion, netType, debug, flip, num_lmks = 68):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    # predictions
    # results of all test images
    predictions = torch.Tensor(loader.dataset.__len__(), num_lmks, 2)

    model.eval()
    gt_win, pred_win = None, None
    bar = Bar('Validating', max=len(loader))
    all_dists = torch.zeros((num_lmks, loader.dataset.__len__()))
    for i, (inputs, target, meta) in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(target.cuda(non_blocking=True))

        output = model(input_var)
        score_map = output[-1].data.cpu()

        if flip:
            flip_input_var = torch.autograd.Variable(
                torch.from_numpy(shufflelr(inputs.clone().numpy())).float().cuda())
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output

        # intermediate supervision
        loss = 0
        for o in output:
            loss += criterion(o[:,:num_lmks,:,:], target_var)

        acc, batch_dists = accuracy(score_map[:,:num_lmks,:,:], target.cpu(), idx, thr=0.07)
        all_dists[:, i * args.val_batch:(i + 1) * args.val_batch] = batch_dists

        preds = final_preds(score_map[:,:num_lmks,:,:], meta['center'], meta['scale'], [64, 64])
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]

        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()
    mean_error = torch.mean(all_dists)
    auc = calc_metrics(all_dists) # this is auc of predicted maps and target.
    print("=> Mean Error: {:.2f}, AUC@0.07: {} based on maps".format(mean_error*100., auc))
    return losses.avg, acces.avg, predictions, auc


if __name__ == '__main__':
    main(args)
