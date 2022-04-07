import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import random
import argparse
import os
import sys
import pickle
import copy
import importlib
import time
import wandb



from CustomSummaryWriter import *
from models import Net, CondNet, SparseNet
from utils import *



if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    #parser.add_argument('--task-id', type=int, help='task id given by the sbatch script')
    #parser.add_argument('--job-name', type=str, help='job name given by the sbatch script')
    parser.add_argument('--seed', type=int, help='random seed', default=42)
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disable CUDA, i.e. use CPU')
    
    #=== Dataset
    parser.add_argument('--dataset-dir', type=str, help='dataset directory', default='./data')
    parser.add_argument('--dataset', type=str, help='dataset to train with', default='MNIST')
    parser.add_argument('--normalize-pixelwise', default=False, action='store_true', help='do pixelwise normalization of the data')
    parser.add_argument('--train-subset-size', type=int, help='Subset of dataset to use, 0 indicates use full dataset', default=0)
    
    #=== Training
    parser.add_argument('--max-epochs', type=int, help='maximum training epochs', default=3)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--lr-decay', type=str, help='learning rate decay type', default='none')
    parser.add_argument('--mbs', type=int, help='mini-batch size', default=128)
    parser.add_argument('--no-es', default=False, action='store_true', help='disable early-stopping')
    
    #=== Model
    parser.add_argument('--model-type', type=str, help='model type', default='Net', choices=['Net', 'SparseNet', 'CondNet'])
    parser.add_argument('--no-bias', default=False, action='store_true', help='don\'t use biases')
    parser.add_argument('--num-layers', type=int, help='total number of layers', default=1)
    parser.add_argument('--num-mid', type=int, help='width of the hidden layer(s)', default=0)
    parser.add_argument('--fan-in', type=int, help='fan-in for condLayer', default=0)
    parser.add_argument('--fan-out-const', type=str, default="True", choices=["True","False"], help='use same fan-out for all units in sparse (condensed) layer')
    #parser.add_argument('--individ-indx-seqs', default=False, action='store_true', help='individual indx seqs for each cond layer')
    parser.add_argument('--make-linear', type=str, default="True", choices=["True","False"], help='disable non-linear (ReLU) activation functions')

    parser.add_argument('--sparsity-type', type=str, default='none', choices=["none","per_neuron", "per_layer"], help='sparsity type for SparseNet')
    parser.add_argument('--connect-type', type=str, default='none', choices=["none","scattered", "block"], help='connectivity type for SparseNet')

    #=== Output saving
    parser.add_argument('--chkpt-epochs', type=int, help='Save checkpoint, frequency in epochs. Do not save if zero.', default=0)
    args= parser.parse_args()
    #=============================



    # ==== device configuration
    use_cuda= not args.no_cuda and torch.cuda.is_available()
    device  = torch.device('cuda' if use_cuda else 'cpu')

    seed= args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    save_checkpoints= args.chkpt_epochs > 0


    # ========== paths and dirs ==========
    # ====================================
    # if args.model_type=='Net':
    #     args.fan_in= args.num_mid
    # else:
    #     args.fan_in= int(args.num_mid/(args.num_layers-2))
    output_dir= make_outputdirname(args)
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir= args.dataset_dir


    # ========== start up wandb ==========
    # ====================================
    wandb.tensorboard.patch(root_logdir=output_dir)
    run= wandb.init(sync_tensorboard=True, config=args)
    # set run name
    run.name= output_dir

    # ========== training and dataset hyper-params ==========
    # =======================================================
    dataset= args.dataset
    normalize_pixelwise= args.normalize_pixelwise
    train_subset_size= args.train_subset_size
    no_ES=args.no_es
    learning_rate = args.lr
    lr_decay= args.lr_decay
    model_type=args.model_type

    ckpt_every=args.chkpt_epochs
    max_epochs= args.max_epochs   # cut-off value for train loop


    # ========== load dataset ==========
    # ==================================
    no_da= True
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    if train_subset_size>0: # training on a subset
        train_batch_size= train_subset_size
    else: # training on original whole train set
        train_batch_size= args.mbs
    test_batch_size = 1000

    train_loader, test_loader, input_size, num_classes, num_channels =\
        load_dataset(dataset, dataset_dir, train_batch_size, test_batch_size, no_da, kwargs)

    batch_size= args.mbs

    # ========== model hyper-params ==========
    # ========================================
    make_linear= args.make_linear=="True"
    add_bias= not args.no_bias
    num_layers= args.num_layers
    num_mid= args.num_mid
    fan_in= args.fan_in
    fan_out_const= args.fan_out_const=="True"

    sparse= model_type=='SparseNet' # args.sparsity_type!='none'

    writer=CustomSummaryWriter(f'{output_dir}/runs')

    # ========== set up model ==========
    # ==================================
    if model_type=='Net':
        model= Net(num_layers=num_layers, num_in=input_size, num_out=num_classes, num_mid=num_mid, make_linear=make_linear, add_bias=add_bias).to(device)
    elif model_type=='SparseNet':
        sparsity_type= args.sparsity_type
        connect_type = args.connect_type
        sparse= fan_in<num_mid
        model= SparseNet(num_layers=num_layers, num_in=input_size, num_out=num_classes, num_mid=num_mid, make_linear=make_linear, add_bias=add_bias, fan_in=fan_in, sparsity_type=sparsity_type, connect_type=connect_type, fan_out_const=fan_out_const).to(device)
    elif model_type=='CondNet':
        model= CondNet(num_layers=num_layers, 
            num_in=input_size, num_out=num_classes, 
            num_mid=num_mid, fan_in=fan_in, fan_out_const=fan_out_const, 
            make_linear=make_linear, add_bias=add_bias).to(device)
    else:
        print(f'Model type {model_type} unknown!')

    print(model)
    
    optimizer= torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 
    criterion= nn.CrossEntropyLoss()
    
    if lr_decay=='cosine':
        scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs+1)
    elif lr_decay=='cosine_2x':
        scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2*max_epochs)

    if sparse:
        # ==== get smask from model ====
        smask={}
        for nr, layer in enumerate(model.midLayers):
            smask[nr]= layer.weight==0


    # ======== save initial model checkpoint
    start_epoch=0
    if save_checkpoints:
        state= {'epoch': start_epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'args': args}
    
        save_name= f'{output_dir}/checkpoints/init'
        print(f'Saving checkpoint at {save_name}')
        save_checkpoint(state, save_name)



    # ============= train ==============
    # ==================================
    train_loss=1        #initialize
    epoch=start_epoch+1 #initialize
    best_test_acc=0     #initialize
    patience= 20
    test_acc_tracker=list(np.zeros(2*patience)) # keep a list of test acc over past some eps
    model.train()


    if train_subset_size>0:
        images_, labels_ = next(iter(train_loader))
        new_train_set= torch.utils.data.TensorDataset(images_, labels_)
        train_loader = torch.utils.data.DataLoader(new_train_set, batch_size=batch_size, shuffle=True)


    step_counter= 0 # counting training steps
    while epoch<=max_epochs:

        loss_sum, total, correct = 0, 0, 0

        for mbi, (images, labels) in enumerate(train_loader):
            step_counter+=1

            if num_channels==1:
                images= images.reshape(-1, input_size).to(device)
            else:
                images= images.to(device)
            if normalize_pixelwise: images= pixelwise_normalization(images) 
            labels= labels.to(device)

            #==== forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += len(images)*loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted==labels).cpu().sum().item()
            total += len(images)

            #==== backward and optimize
            optimizer.zero_grad()
            loss.backward()

            if sparse: # apply smask to gradients
                #G={}
                for nr, layer in enumerate(model.midLayers):
                    layer.weight.grad[ smask[nr] ]= 0
                    #G[nr]= layer.weight.grad
                #torch.save(G, f'{output_dir}/grads_sparse_{epoch}_{mbi}.pt')

            #== computing gradient norm
            grad_sq_sum_mb= 0 # parameter squared summed, for minibatch
            for p in model.parameters():
                grad_sq_sum = p.grad.detach().data.pow(2).sum()
                grad_sq_sum_mb += grad_sq_sum.item()

            writer.add_scalars('gradnorm_mb', {'train': grad_sq_sum_mb**0.5}, global_step=step_counter, walltime=time.time()-start_time )

            optimizer.step()
        #==== epoch completed.

        train_loss= loss_sum/total
        train_acc = correct/total

        # ======== save model checkpoint
        if save_checkpoints:
            if epoch%ckpt_every==0:
                state= {'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'args': args}
                save_name = f'{output_dir}/checkpoints/chkpt_epoch_{epoch}'
                print(f'Saving checkpoint at {save_name}')
                save_checkpoint(state, save_name)

        # ======== evaluate ========
        test_acc, test_loss = evaluate(model, None, test_loader, normalize_pixelwise, input_size, device, criterion)
        
        # ======== write to TB and stats file
        # (saves both to tb event files and a separate dict called "stats") every epoch
        writer.add_scalars('acc', {'test': test_acc, 'train': train_acc}, global_step=step_counter, walltime=time.time()-start_time )
        writer.add_scalars('loss', {'test': test_loss, 'train': train_loss}, global_step=step_counter, walltime=time.time()-start_time )
        
        
        # ======== Early Stopping routine
        if not no_ES:
            test_acc_tracker.append(test_acc)
            _=test_acc_tracker.pop(0)
            prev_avg_acc=np.mean(test_acc_tracker[:patience])
            curr_avg_acc=np.mean(test_acc_tracker[patience:])
            if curr_avg_acc<prev_avg_acc and epoch>(2*patience):
                print(f'>>> Early Stopping: epoch {epoch}')
                print(f'* current avg: {curr_avg_acc}')
                print(f'* previous avg: {prev_avg_acc}')
                print(f'(no improvement over past {patience} epochs)')
                break

        # ==== remember best test acc
        is_best= test_acc > best_test_acc
        best_test_acc= max(test_acc, best_test_acc)

        if save_checkpoints:
            if is_best:
                best_epoch= epoch
                best_model_dict= copy.deepcopy(model.state_dict())
                best_optim_dict= copy.deepcopy(optimizer.state_dict())


        if lr_decay!='none':
            scheduler.step()
        epoch+=1

    writer.close()  # close current event file
    

    # ========== save best and final model checkpoints =============
    # ==============================================================
    if save_checkpoints:
        
        #=== best
        state= {'epoch': best_epoch, 'state_dict': best_model_dict,
                'optimizer': best_optim_dict, 'args': args, 'best_test_acc': best_test_acc}
        save_name = f'{output_dir}/checkpoints/best'
        print(f'Saving checkpoint as {save_name}')
        save_checkpoint(state, save_name)


        #=== final
        state= {'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'args': args, 'fin_test_acc': test_acc}
        save_name = f'{output_dir}/checkpoints/final'
        print(f'Saving checkpoint as {save_name}')
        save_checkpoint(state, save_name)