#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import torch.nn as nn
import random

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverlocal import Local

from flcore.servers.server_LMWeather import LMWeather
from flcore.servers.server_LMWeather_reg import LMWeather_reg

from utils.param_utils import calculate_model_size
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
# torch.manual_seed(0)

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):

        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "DLinear": # # encoder-only
            from flcore.trainmodel.DLinear import DLinear
            args.model = DLinear(args).to(args.device)
        elif model_str == 'PatchTST':
            from flcore.trainmodel.PatchTST import PatchTST
            args.model = PatchTST(args).to(args.device)

        elif model_str == 'GPT4TS': # encoder-only
            from flcore.trainmodel.GPT4TS import GPT4TS
            args.model = GPT4TS(args).to(args.device)

        elif model_str == 'GPT4TS_Adapter':
            from flcore.trainmodel.GPT4TS_Adapter import GPT4TSPrompt
            args.model = GPT4TSPrompt(args).to(args.device)
            
        elif model_str == 'Bert4TS': # encoder-only
            from flcore.trainmodel.Bert4TS import Bert4TS
            args.model = Bert4TS(args).to(args.device)

        elif model_str == 'Bert4TS_Trend':
            from flcore.trainmodel.Bert4TS_Adapter import Bert4TS_Adapter
            args.model = Bert4TS_Adapter(args).to(args.device)

        elif model_str == 'OpenLlama4TS': # encoder-only
            from flcore.trainmodel.OpenLlama4TS import OpenLlama4TS
            args.model = OpenLlama4TS(args).to(args.device)
        elif model_str == 'Transformer':
            from flcore.trainmodel.Transformer import Transformer
            args.model = Transformer(args).to(args.device)
        elif model_str == 'Informer':
            from flcore.trainmodel.Informer import Informer
            args.model = Informer(args).to(args.device)
        elif model_str == 'iTransformer': # encoder-only
            from flcore.trainmodel.iTransformer import iTransformer
            args.model = iTransformer(args).to(args.device)
        elif model_str == 'LightTS': # encoder-only
            from flcore.trainmodel.LightTS import LightTS
            args.model = LightTS(args).to(args.device)
        elif model_str == 'Pyraformer':
            from flcore.trainmodel.Pyraformer import Pyraformer
            args.model = Pyraformer(args).to(args.device)
        elif model_str == 'Reformer':
            from flcore.trainmodel.Reformer import Reformer
            args.model = Reformer(args).to(args.devicece)
        else:
            raise NotImplementedError

        print(args.model)
        total_size, trainable_size = calculate_model_size(args.model)
        print(f"Total model size: {total_size:.2f} M")
        print(f"Trainable model size: {trainable_size:.2f} M")

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)

        elif args.algorithm == 'LMWeather':
            server = LMWeather(args, i)

        elif args.algorithm == 'LMWeather_reg':
            server = LMWeather_reg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        else:
            raise NotImplementedError
        print(args.model)
        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(configs=args, dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, required=True, default="long_term_forecast", 
                        help='The goal of this experiment, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('-tn', "--task_name", type=str, default='long_term_forecast', choices={'long_term_forecast', 'short_term_forecast', 'imputation'})
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)

    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    
    # For general time-series task
    parser.add_argument("--seq_len", type=int, default=196, 
                        help="The length of the history time series")
    parser.add_argument("--pred_len", type=int, default=192,
                        help="The length of the future time series need to forecast")
    parser.add_argument('--label_len', type=int, default=48)

    # Model definition
    parser.add_argument('--moving_avg', type=int, default=25, 
                        help='window size of moving average')
    parser.add_argument('--top_k', type=int, default=5, 
                        help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, 
                        help='for Inception')
    parser.add_argument("--kernel_size", type=int, default=25,
                        help="The kernel size of moving avrage operation")
    parser.add_argument("--enc_in", type=int, default=20,
                        help="The number of time-series channels (features)")
    parser.add_argument('--dec_in', type=int, default=20, 
                        help='decoder input size')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=2, 
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, 
                        help='num of decoder layers')
    parser.add_argument('--d_model', type=int, default=768, 
                        help='dimension of model')
    parser.add_argument('--c_out', type=int, default=1, 
                        help='output size')
    parser.add_argument('--n_heads', type=int, default=8, 
                        help='num of heads')
    parser.add_argument('--d_ff', type=int, default=512, 
                        help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0, 
                        help='dropout (also adapted to the lora setting)')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--output_attention', action='store_true', 
                        help='whether to output attention in ecoder')
    parser.add_argument('--activation', type=str, default='gelu', 
                        help='activation')
    parser.add_argument('--factor', type=int, default=1, 
                        help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

    # time-series dataset
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--target', type=str, default='rh',
                        help="The target variable need to predict, small for Werther Dataset, big for DroughtED Dataset",
                        choices={'ap', 't', 'mxt', 'mnt', 'dt', 'rh', 'wvp', 'p1', 'p2', 'p3', 'p4', 'p5', 'wd', 'ws', 'mwd', 'mws', 'st', 'hv1', 'hv2', 'vv', 
                                 'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX',	'WS10M_MIN', 'WS10M_RANGE',
                                 'WS50M', 'WS50M_MAX', 'WS50M_MIN',	'WS50M_RANGE',
                                 'humidity', 'temperature', 'wind_direction', 'wind_speed', 'pressure'})

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # For pre-trained model
    parser.add_argument('--base_model', type=str, default='GPT2',
                        help="The base pretrain model, options: [GPT2, Bert, DLinear...]")
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze_part', type=int, default=10000,
                        help="Alternative training setting for partial frozen/unfreeze, default 10000 means NONE")

    # For GPT2 in time-series forecasting
    parser.add_argument('--gpt_layers', type=int, default=1,
                        help="The number of layers from pre-trained GPT2 model")
    parser.add_argument('--is_gpt', type=int, default=1)

    # For Bert in time-series forecasting
    parser.add_argument('--bert_layers', type=int, default=3,
                        help="The number of layers from pre-trained Bert model")

    parser.add_argument('--inverse', type=int, default=0,
                        help='Inverse the forecasting data in test phase')
    parser.add_argument('--lradj', type=str, default='type1', 
                        help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', 
                        help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_reg', action='store_true', default=False,
                        help='Use L2 regularization to optimize federated average processes')
    
    # Parameter-Efficient Fine-Tuning
    parser.add_argument('--is_peft', type=int, default=1)
    parser.add_argument('--peft', type=str, default='lora',
                        help="The manner of parameter-efficient fine-tuning of transformers architectures")
    parser.add_argument('--rank', type=int, default=8,
                        help="The number of rank if use lora")

    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)



    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    # print("Algorithm: {}".format(args.algorithm))
    # print("Local batch size: {}".format(args.batch_size))
    # print("Local steps: {}".format(args.local_epochs))
    # print("Local learing rate: {}".format(args.local_learning_rate))
    # print("Local learing rate decay: {}".format(args.learning_rate_decay))
    # if args.learning_rate_decay:
    #     print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    # print("Total number of clients: {}".format(args.num_clients))
    # print("Clients join in each round: {}".format(args.join_ratio))
    # print("Clients randomly join: {}".format(args.random_join_ratio))
    # print("Client drop rate: {}".format(args.client_drop_rate))
    # print("Client select regarding time: {}".format(args.time_select))
    # if args.time_select:
    #     print("Time threthold: {}".format(args.time_threthold))
    # print("Running times: {}".format(args.times))
    # print("Dataset: {}".format(args.dataset))
    # print("Number of classes: {}".format(args.num_classes))
    # print("Backbone: {}".format(args.model))
    # print("Using device: {}".format(args.device))
    # print("Using DP: {}".format(args.privacy))
    # if args.privacy:
    #     print("Sigma for DP: {}".format(args.dp_sigma))
    # print("Auto break: {}".format(args.auto_break))
    # if not args.auto_break:
    #     print("Global rounds: {}".format(args.global_rounds))
    # if args.device == "cuda":
    #     print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    # print("DLG attack: {}".format(args.dlg_eval))
    # if args.dlg_eval:
    #     print("DLG attack round gap: {}".format(args.dlg_gap))

        # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))
    print("=" * 50)

    run(args)

