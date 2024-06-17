import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, AutoModel
import logging
import time
import pdb
import os
import json
import random
from tqdm import tqdm
from itertools import chain
import sys
sys.path.append("./src/") 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from data_loader import (
    C2_Dataset,
)
from model_wrapper import (
    Model_Wrapper
)
from metric_learning import (
    C2_Metric_Learning,
)


LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='C2 BLI System')
    parser.add_argument('--model_dir', 
                        help='model dir')
    parser.add_argument('--train_dir', type=str, required=True,
                    help='training dict dir')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output dir')
    parser.add_argument('--l1', type=str, required=True,
                        help='Language L1')
    parser.add_argument('--l2', type=str, required=True,
                        help='Language L2')    
    parser.add_argument('--l1_voc', type=str, required=True,
                        help='Directory of L1 Vocabulary')    
    parser.add_argument('--l1_emb', type=str, required=True,
                        help='Directory of Aligned Static Embeddings for L1')
    parser.add_argument('--l2_voc', type=str, required=True,
                        help='Directory of L2 Vocabulary')
    parser.add_argument('--l2_emb', type=str, required=True,
                        help='Directory of Aligned Static Embeddings for L2')
    parser.add_argument('--max_length', default=6, type=int)
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=50, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=5, type=int)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--amp', action="store_true", 
            help="automatic mixed precision training")
    parser.add_argument('--parallel', action="store_true") 
    parser.add_argument('--random_seed',
                        help='random seed',
                        default=None, type=int)
    parser.add_argument('--infoNCE_tau', default=0.1, type=float) 
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean_pool}") 
    parser.add_argument('--dropout_rate', default=0.1, type=float) 
    parser.add_argument('--random_erase', default=0.0, type=float, 
            help="portion of tokens to be randomly erased on one side of the input")
    parser.add_argument('--num_neg',help='num_neg',default=10, type=int)
    parser.add_argument('--template', default=0, type=int)
    args = parser.parse_args()

    ###
    if args.model_dir == "google/mt5-small" or args.model_dir == "google/mt5-base":
        args.amp = False
    ###


    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def train(args, data_loader, model, scaler=None, model_wrapper=None, step_global=0):
    LOGGER.info("train!")

    
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()

        
        batch_x1, batch_x2 = data
        batch_x_cuda1, batch_x_cuda2 = {}, {}
        for k,v in batch_x1.items():
            batch_x_cuda1[k] = v.cuda()
        for k,v in batch_x2.items():
            batch_x_cuda2[k] = v.cuda()              

        if args.amp:
            with autocast():
                bli_loss = model(batch_x_cuda1, batch_x_cuda2)
        else:
            bli_loss = model(batch_x_cuda1, batch_x_cuda2)  
        loss_final = bli_loss
        if args.amp:
            scaler.scale(loss_final).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss_final.backward()
            model.optimizer.step()

        train_loss += loss_final.item()
        train_steps += 1
        step_global += 1
        if step_global % args.checkpoint_step == 0:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_iter_{}".format(str(step_global)))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global
    
def main(args):
    init_logging()
    print(args)

    torch.manual_seed(args.random_seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_wrapper = Model_Wrapper()
    encoder, tokenizer = model_wrapper.load_bert(
        path=args.model_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
    )

    
    mask_token_id = tokenizer.encode("[MASK]")[1]
    print ("[MASK] token ID:", mask_token_id)
 
    if args.model_dir == "bert-base-multilingual-uncased":
        encoder.embeddings.dropout = torch.nn.Dropout(p=args.dropout_rate)
        for i in range(len(encoder.encoder.layer)):
            encoder.encoder.layer[i].attention.self.dropout = torch.nn.Dropout(p=args.dropout_rate)
            encoder.encoder.layer[i].attention.output.dropout = torch.nn.Dropout(p=args.dropout_rate)
            encoder.encoder.layer[i].output.dropout  = torch.nn.Dropout(p=args.dropout_rate)

    if args.model_dir == "google/mt5-small" or args.model_dir == "google/mt5-base":
        args.dropout_rate = 0.1
        encoder.encoder.dropout = torch.nn.Dropout(p=args.dropout_rate)
        encoder.decoder.dropout = torch.nn.Dropout(p=args.dropout_rate)
        for i in range(len(encoder.encoder.block)):
            encoder.encoder.block[i].layer[0].dropout = torch.nn.Dropout(p=args.dropout_rate)
            encoder.encoder.block[i].layer[1].DenseReluDense.dropout = torch.nn.Dropout(p=args.dropout_rate)
            encoder.encoder.block[i].layer[1].dropout = torch.nn.Dropout(p=args.dropout_rate)
        for i in range(len(encoder.decoder.block)):
            encoder.decoder.block[i].layer[0].dropout = torch.nn.Dropout(p=args.dropout_rate)
            encoder.decoder.block[i].layer[1].dropout = torch.nn.Dropout(p=args.dropout_rate)
            encoder.decoder.block[i].layer[2].DenseReluDense.dropout = torch.nn.Dropout(p=args.dropout_rate)
            encoder.decoder.block[i].layer[2].dropout = torch.nn.Dropout(p=args.dropout_rate)
        encoder = encoder.get_encoder()


    # The "xlm-mlm-100-1280" Huggingface model does not have built-in dropout layers. So we do nothing for it here.

    model = C2_Metric_Learning(
        encoder = encoder,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        use_cuda=args.use_cuda,
        infoNCE_tau=args.infoNCE_tau,
        agg_mode=args.agg_mode,
        num_neg = args.num_neg,
        batch_size = args.train_batch_size
        )

    if args.parallel:
        model.model_pairwise = torch.nn.DataParallel(model.model_pairwise)
        LOGGER.info("using nn.DataParallel")
    
    def collate_fn_batch_encoding(batch):
        query1, query2, query_id = zip(*batch)
        query_encodings1 = tokenizer.batch_encode_plus(
                list(chain(*query1)), 
                max_length=args.max_length, 
                padding="max_length", 
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")
        query_encodings2 = tokenizer.batch_encode_plus(
                list(chain(*query2)), 
                max_length=args.max_length, 
                padding="max_length", 
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")


        return  query_encodings1, query_encodings2


    train_set = C2_Dataset(
            path=args.train_dir,
            l1 = args.l1,
            l2 = args.l2,
            l1_voc = args.l1_voc,
            l2_voc = args.l2_voc,
            tokenizer = tokenizer,
            random_erase=args.random_erase,
            template = args.template
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn_batch_encoding
    )
    # mixed precision training 
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    start = time.time()
    step_global = 0
    for epoch in range(1,args.epoch+1):
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))

        # train
        train_loss, step_global = train(args, data_loader=train_loader, model=model, 
                scaler=scaler, model_wrapper=model_wrapper, step_global=step_global)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        
        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
        
        # save model last epoch
        if epoch == args.epoch:
            model_wrapper.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
