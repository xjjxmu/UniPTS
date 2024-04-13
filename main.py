import random
import time
import os
import datetime
from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import set_gpu, get_dataset, get_model, \
    get_logger,get_directories,get_optimizer
from utils.lr_scheduler import build_scheduler
from trainer import train, validate
from utils.net_utils import save_checkpoint, LabelSmoothing
from utils.logging import *
from args import args
import models

def sigmoid(x, beta):
    z = np.exp(-x*beta)
    sig = 1 / (1 + z)

    return sig

class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
    def forward(self, y_s, y_t, epoch=None):
        tem =  8 * 0.99 ** epoch
        p_s = F.log_softmax(y_s, dim=1) / tem 
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

def get_sparisity(net, solution):
    solution = softmax(solution)
    sparsity = []
    k = 0
    for n, m in net.named_modules():
        if hasattr(m, 'prune_rate'):
            params = m.weight.numel()
            sparse = available_params * solution[k] / params
            sparsity.append(max(0, args.prune_rate + 0.05 - sparse))
            k += 1
    return sparsity

def fitness_function(solution, net, train_loader, logger, search = True):
    model = copy.deepcopy(net)
    model = set_gpu(args, model)
    if search is True:
        solution = get_sparisity(model, solution)

    k = 0
    sum_sparse = 0
    count = 0
    for n, m in model.named_modules():
        if hasattr(m, 'prune_rate'):
            m.set_prune_rate(solution[k])
            sparsity, total_params = m.getSparsity()
            sum_sparse += int(( sparsity / 100) * total_params)
            count += total_params
            k += 1
    
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
    model.train()
    
    with torch.no_grad():
        for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
            noise = torch.randn_like(images) * 0.000001
            images = images + noise
            if args.gpu is not None:
                images = images.cuda(args.gpu)
            target = target.cuda(args.gpu).long()
            _ = model(images)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)
    acc1, _ = validate(train_loader, model, criterion, args, logger, 0)
    acc1 /=100
    metric = acc1

    sparsity = sum_sparse / count
    ratio = abs(args.prune_rate - sparsity)
    
    
    logger.info("sparsity:{:.3f}, ratio:{:.3f}, metric:{:.3f}".format(sparsity, ratio, metric))
    del model
    return metric

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_values = np.exp(x)
    sum_exp = np.sum(exp_values, axis=-1, keepdims=True)
    softmax_result = exp_values / sum_exp
    return softmax_result

def generate_initial_population(population_size, vector_length, net,logger):
    population =  [np.reshape(np.array([args.prune_rate] * vector_length),(1, vector_length))]
    solution = np.random.rand(population_size-1, vector_length) 
    population.append(solution)
    population = np.concatenate(population, axis = 0)
    return population

def selection(population, fitness_values, num_parents):    
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_population = population[sorted_indices]
    parents = sorted_population[:num_parents]
    return parents



def crossover(parents, crossover_rate, vector_length, crossover_num = 12): 
    childs = [np.reshape(np.array([args.prune_rate] * vector_length), (1, vector_length))]
    num_parents = len(parents)
    iter = 0
    max_iters = 10 * crossover_num 
    while iter < max_iters and len(childs) < crossover_num:
        id1, id2 = np.random.choice(num_parents, 2, replace=False)
        parent1 = parents[id1]
        parent2 = parents[id2]
        if random.random() < crossover_rate:
            mask = np.random.randint(low=0, high=2, size=vector_length).astype(np.float32)
            child = parent1*mask + parent2*(1.0-mask)
        else:
            child = parent1
        iter += 1
        childs.append(np.reshape(child,(1,vector_length)))
        if len(childs) == crossover_num:
            break
    childs = np.concatenate(childs,axis=0)
    return childs


def mutation(parents, mutation_rate, mutation_num = 12):
    print("mutation...")
    childs = []
    iter=0
    max_iters = 10
    num_parents, vector_length = parents.shape
    while len(childs) < mutation_num and iter<max_iters:
        ids = np.random.choice(num_parents, mutation_num)
        select = parents[ids]
        is_m = np.random.choice(np.arange(0,2), (mutation_num, vector_length), p=[1 - mutation_rate, mutation_rate])
        mu_val = (1 + np.random.randn(mutation_num, vector_length)) * select * is_m
        mu_val[mu_val == 0] = select[mu_val == 0]
        iter+=1
        for child in mu_val:
            childs.append(np.reshape(child,(1,vector_length)))
    childs = np.concatenate(childs,axis=0)
    return childs

def get_calib_loaders(args, train_dataset):
    use_cuda = torch.cuda.is_available()
    kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
    indices = np.random.choice(len(train_dataset), args.num_samples_for_bn)
    calib_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    return calib_loader

def init(net, logger):
    global sparse_init
    sparse_init =[]
    total_params = 0
    for n, m in net.named_modules():
        if hasattr(m, "prune_rate"):
            params = m.weight.numel()
            total_params += params
    global available_params
    available_params = total_params * 0.05 
    logger.info("total params: {}, available params {}".format(total_params, available_params))

    for n, m in net.named_modules():
        if hasattr(m, "prune_rate"):
            params = m.weight.numel()
            sparsity = min(0.95, available_params / params)
            sparse_init.append(sparsity)
            logger.info("module: {}, params: {}, max sparsity: {}".format(n, params, sparsity))

def genetic_algorithm(net, logger,val_loader, train_dataset):

    use_cuda = torch.cuda.is_available()

    kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

    indices = np.random.choice(len(train_dataset), args.num_samples_for_bn)
    calib_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    population_size = 25
    generations = 25
    num_parents = 10
    mutation_rate = 0.1
    crossover_rate = 1
    
    vector_length = 0
    print("train dataset len:{}".format(len(train_dataset)))
    for n, m in net.named_modules():
        if hasattr(m, 'prune_rate'):
            vector_length += 1

    init(net, logger)
    population = generate_initial_population(population_size, vector_length, net, logger)
    for generation in range(generations):
        logger.info("======= Generation:{} ========".format(generation))
        fitness_values = [fitness_function(solution, net, calib_loader, logger) for solution in population]

        parents = selection(population, fitness_values, num_parents)

        cross_childs = crossover(parents, crossover_rate, vector_length)
        mutation_childs = mutation(parents, mutation_rate)
        random_num = max(population_size - len(cross_childs) - len(mutation_childs), 0)
        if random_num > 0:
            random_childs = generate_initial_population(random_num, vector_length, net, logger)
            population = np.concatenate((cross_childs, mutation_childs, random_childs),0)
        population = np.concatenate((cross_childs, mutation_childs), 0)
        
    best_solution = max(population, key=lambda solution: fitness_function(solution, net, calib_loader, logger))
    return best_solution

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main_worker(args):
    seed_all(args.seed)
    
    data = get_dataset(args)
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    ensure_path(log_base_dir)
    ensure_path(ckpt_base_dir)
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    logger = get_logger(os.path.join(log_base_dir, 'logger'+now+'.log'))

    net = get_model(args)
    ckpt = torch.load(args.checkpoint)
    if 'ImageNet' in args.set:
        for key in list(ckpt.keys()):
            new_key = key
            if 'fc.weight' in key or 'classifier.weight' in key:
                ckpt[new_key] = ckpt.pop(key).unsqueeze(-1).unsqueeze(-1)
            else:
                ckpt[new_key] = ckpt.pop(key)
    else:
        ckpt = ckpt['net']
        for key in list(ckpt.keys()):
            new_key = key[key.find(".") + 1 :]
            ckpt[new_key] = ckpt.pop(key)
    net.load_state_dict(ckpt)
    net = set_gpu(args, net)

    
    best_solution = genetic_algorithm(net, logger, data.val_loader, data.calib_dataset)
    logger.info(f"searched distribution ratio:{best_solution}")
    best_solution = np.array(best_solution)
    best_solution = get_sparisity(net, best_solution)
    logger.info(f"searched sparsity ratio:{best_solution}")

    k = 0
    save_dict = {}
    for n, m in net.named_modules():
        if hasattr(m, 'prune_rate'):
            m.set_prune_rate(best_solution[k])
            print(f"name{n}, sparsity:{best_solution[k]}")
            k += 1


    criterion = nn.CrossEntropyLoss().cuda()
    args.start_epoch = args.start_epoch or 0
    kd_criterion = DistillKL(args.T)
    optimizer = get_optimizer(args, net)
    lr_scheduler = build_scheduler(args, optimizer, len(data.calib_loader))
    
    name = args.arch+"_dense"
    fp_model = models.__dict__[name]()
    fp_model.load_state_dict(ckpt)
    fp_model.cuda()
    fp_model.train()

    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        net = set_gpu(args, net)
        train_acc1, train_acc5 = train(
            data.calib_loader, net, kd_criterion, optimizer, epoch, args, 
            logger, fp_model, lr_scheduler)

        count = 0
        sum_sparse = 0.0
        for n, m in net.named_modules():
            if hasattr(m, "prune_rate"):
                sparsity, total_params = m.getSparsity()
                logger.info("epoch{} {} sparsity {}% ".format(epoch, n, sparsity))
                sum_sparse += int(((100 - sparsity) / 100) * total_params)
                count += total_params
        total_sparsity = 100 - (100 * sum_sparse / count)
        logger.info("epoch {} sparsitytotal {}".format(epoch, total_sparsity))

        # evaluate on validation set
        acc1, acc5 = validate(data.val_loader, net, criterion, args, logger, 0)
        logger.info(f"Accuracy of the network at epoch {epoch}: {acc1:.3f}%")

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": net.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

    acc1, acc5 = validate(data.val_loader, net, criterion, args, logger, 0)
    logger.info(f"Accuracy of the network at epoch 100: {acc1:.3f}%")
    logger.info(f"best Accuracy of the network: {best_acc1:.3f}%")

if __name__ == "__main__":
    main_worker(args)

    
