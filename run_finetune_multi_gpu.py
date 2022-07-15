
import json
import os
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertForSequenceClassification, DNATokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import argparse

from utils import compute_correct_attention_masks

def train(gpu, args):
    # only one node
    rank = gpu
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)

    tokenizer = DNATokenizer.from_pretrained(args.model_path)
    model = BertForSequenceClassification.from_pretrained(args.model_path, 
                                                        num_labels=5, # types of SV
                                                        output_attentions=False,
                                                        output_hidden_states=False)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    # Prepare optimizer and schedule (linear warmup and decay)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Data load
    train_sampler = DistributedSampler(args.train_dataset, num_replicas=args.world_size, rank=rank)
    train_dataloader = DataLoader(
        dataset=args.train_dataset, 
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=False)

    t_total = len(train_dataloader) * args.num_train_epochs
    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent * t_total)

    tr_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()

    global_step = 0
    epochs_trained = 0

    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()

            batch = tuple(t.to(gpu) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if gpu == 0:
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}

                    # if args.evaluate_during_trainign:
                    #     results = evaluate(dataloader_val=eval_loader, model=model, device=device)

                    #     for key, value in results.items():
                    #         eval_key = "eval_{}".format(key)
                    #         logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    print(f"")

                    with open(f"{output_dir}/log_{global_step}.json", "w") as f:
                        json.dump(logs, f)
                    

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training

                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    
    if gpu == 0:
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

def evaluate(dataloader_val, model, device):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def main():    
    parser = argparse.ArgumentParser()

    os.environ['MASTER_ADDR'] = '10.128.2.151'
    os.environ['MASTER_PORT'] = '8888'

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    # parser.add_argument(
    #     "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    # )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
        
    # Other parameters
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_percent", default=0, type=float, help="Linear warmup over warmup_percent*total_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--n_nodes",
        default=1,
        type=int,
        help="number of nodes used for data process",
    )

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = 1
    
    # Load model and tokenizer
    # Load config, model and tokenizer
    model_path = "dnabert6/"
    args.model_path = model_path

    # Prepare data
    # Read sequences
    d_train = pd.read_csv(f"{args.data_dir}/train.tsv", sep="\t")
    d_eval = pd.read_csv(f"{args.data_dir}/eval.tsv", sep="\t")

    d_train = d_train.iloc[:100]
    d_eval = d_train.iloc[:100]

    # Prepare tokenized sequences
    sequences_train = d_train.sequence.values
    labels_train = d_train.label.values

    sequences_eval = d_eval.sequence.values
    labels_eval = d_eval.label.values

    tokenizer = DNATokenizer.from_pretrained(args.model_path)

    # Encode them
    encoded_train = tokenizer.batch_encode_plus(
                    sequences_train,
                    add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                    padding = 'longest',        # Pad to longest in batch.
                    truncation = True,          # Truncate sentences to `max_length`.
                    max_length = 512,   
                    return_tensors = 'pt',        # Return pytorch tensors.
            )
    encoded_train = compute_correct_attention_masks(tokenizer_output=encoded_train)

    encoded_eval = tokenizer.batch_encode_plus(
                    sequences_eval,
                    add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                    padding = 'longest',        # Pad to longest in batch.
                    truncation = True,          # Truncate sentences to `max_length`.
                    max_length = 512,   
                    return_tensors = 'pt',        # Return pytorch tensors.
            )
    encoded_eval = compute_correct_attention_masks(tokenizer_output=encoded_eval)

    print('finished encoding sequences')

    # Take useful info and prepare Dataset
    input_ids_train = encoded_train['input_ids']
    attention_masks_train = encoded_train['attention_mask']
    labels_train = torch.tensor(labels_train)

    input_ids_eval = encoded_eval['input_ids']
    attention_masks_eval = encoded_eval['attention_mask']
    labels_eval = torch.tensor(labels_eval)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_eval = TensorDataset(input_ids_eval, attention_masks_eval, labels_eval) 
    args.dataset_train = dataset_train

    # Spawn mutiple gpu training
    args.n_gpu = n_gpu
    args.world_size = n_gpu * args.n_nodes
    mp.spawn(train, nprocs=args.n_gpu, args=(args,)) 

if __name__ == "__main__":
    main()