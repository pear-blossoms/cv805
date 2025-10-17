import json
import torch
# from models.retriever import Retriever, Config
from llava.model import LlavaConfig, Retriever
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import argparse
# from dataset.dataset import RetriverDataset, DataCollatorForSupervisedDataset
from llava.retrieval_dataset import RetriverDataset, DataCollatorForSupervisedDataset

import os
# from models.utils import load_config
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def conver_json_to_config(config_json):
    config = LlavaConfig()
    keys = config.__dict__.keys()
    for key, value in config_json.items():
        if key in keys:
            setattr(config, key, value)
    return config

def load_config(path):
    with open(path) as f:
        config_json = json.load(f)
    return conver_json_to_config(config_json)


# def get_task_mask(self, task):
#     mask = torch.zeros(self.config.pool_size, dtype=torch.int, device=self.retriever.keys.device)
#     l_idx, r_idx = self.config.task_pool_index_range[task]
#     mask[l_idx: r_idx] = 1
#     return mask

def train(rank, args):
    current_gpu_index = rank
    torch.cuda.set_device(current_gpu_index)
    dist.init_process_group(
        backend='nccl', 
        world_size=args.world_size, 
        rank=current_gpu_index,
        init_method='env://'
    )
    print("Init Success")
    if args.config_path is not None:
        print("Load config from {}".format(args.config_path))
        config = load_config(args.config_path)
    else:
        config = LlavaConfig()
    model = Retriever(config)
    if args.checkpoint is not None:
        print("Load checkpoint from {}".format(args.checkpoint))
        model.load_state_dict(torch.load(args.checkpoint), strict=False)
    train_dataset = RetriverDataset(data_paths=args.data_paths)
    dist_train_samples = DistributedSampler(
        dataset=train_dataset,
        num_replicas=args.world_size,
        rank=rank,
        seed=17
    )
    collate_batch = DataCollatorForSupervisedDataset(tokenizer=model.get_tokenizer())
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=args.num_workers, sampler=dist_train_samples, pin_memory=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)

    model = model.train().cuda(rank)
    
    # Set task mask
    mask = torch.zeros(config.pool_size, dtype=torch.int, device=model.keys.device)
    l_idx, r_idx = config.task_pool_index_range[args.task]
    mask[l_idx: r_idx] = 1
    if rank == 0:
        print('config task idx',config.task_pool_index_range)
        print("mask: {}".format(mask))

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    loss_sum=0
    for epoch in range(args.epochs):
        if rank == 0:
            print("Epoch:{}/{} ==========>".format(epoch + 1, args.epochs))
        dist_train_samples.set_epoch(epoch)
        iters_sum = len(train_dataloader)
        for idx, inputs in enumerate(train_dataloader):
            # print("inputs: {}".format(inputs))
            optimizer.zero_grad()
            inputs = inputs.to(rank)
            out = model(inputs=inputs, pool_mask=mask, training_keys_only=True)
            loss = out['key_loss']
            loss.backward()
            loss_sum = (loss_sum + loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            if ((idx % args.log_interval == 0 and idx != 0) or idx == (iters_sum - 1)) and rank == 0:
                iters_last = args.log_interval if idx != (iters_sum - 1) else iters_sum % args.log_interval
                print("iter {}/{}, loss: {}, lr:{}".format(idx, iters_sum, loss_sum / (iters_last+1), optimizer.param_groups[0]['lr']))
                loss_sum = 0
        scheduler.step()
    
    if rank == 0:
        print("save model to {}".format(args.save_path))
        model = model.to("cpu")
        state_dict = model.module.state_dict()
        keys = list(state_dict.keys())

        save_dir = os.path.dirname(args.save_path)
        if not os.path.exists(save_dir) and save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for key in keys:
            if "model." in key:
                state_dict.pop(key)
        torch.save(state_dict, args.save_path)



if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default=8, type=int)
    parser.add_argument('--epochs', default=3, type=int, metavar='N')
    parser.add_argument('--lr_step_size', default=1, type=int)
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--save_path', default="/vast/users/xiaodan/haokunlin/Continual_LLaVA/llava/output/prompt-key/MRI.pth", type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--task', default='MRI', type=str)
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('--data_paths', default="/vast/users/xiaodan/haokunlin/data/llava_med_for_cv805/upload_hf/training/MRI_data_4416.json", type=str)
    parser.add_argument('--master_port', default="8889", type=str)
    parser.add_argument('--local-rank', default=-1, type=int)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = args.master_port
    args.world_size = args.gpus
    mp.spawn(
        train,
        nprocs=args.gpus,
        args=(args, )
    )