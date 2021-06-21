import random

import torch
from torch.utils.data.dataset import Dataset
import horovod.torch as hvd
import random
# random.uniform()

class MultipleDatasets(Dataset):
    def __init__(self, dbs):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])

    def __len__(self):
        return self.max_db_data_num * self.db_num

    def __getitem__(self, index):
        db_idx = index // self.max_db_data_num
        data_idx = index % self.max_db_data_num
        if data_idx > len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])):
            data_idx = random.randint(0, len(self.dbs[db_idx]) - 1)
        else:
            data_idx = data_idx % len(self.dbs[db_idx])

        return self.dbs[db_idx][data_idx]


class data_prefetcher():
    def __init__(self, cfg, loader):
        self.cfg = cfg
        self.loader = iter(loader)
        mean = cfg.DATA.PIXEL_MEAN
        std = cfg.DATA.PIXEL_STD
        self.stream = torch.cuda.Stream(priority=1)
        self.mean = torch.tensor([mean[0], mean[1], mean[2]]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([std[0], std[1], std[2]]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
            # if hvd.rank() == 0:
            #     import ipdb
            #     ipdb.set_trace()
        except StopIteration:
            self.next_input = None
            return

        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            # self.next_input = self.next_input.cuda(non_blocking=True)
            # if hvd.rank() == 0:
            #     import ipdb
            #     ipdb.set_trace()
            for k,v in self.next_input.items():
                v = v.cuda(non_blocking=True)
                # self.next_target = self.next_target.cuda(non_blocking=True)
                # more code for the alternative if record_stream() doesn't work:
                # copy_ will record the use of the pinned source tensor in this side stream.
                # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
                # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
                # self.next_input = self.next_input_gpu
                # self.next_target = self.next_target_gpu

                # With Amp, it isn't necessary to manually convert data to half.
                # if args.fp16:
                #     self.next_input = self.next_input.half()
                # else:
                if k in ['rgb', 'ir']:
                    self.next_input[k][:,:3,:,:] = v[:,:3,:,:].float()
                    self.next_input[k][:,:3,:,:] = v[:,:3,:,:].sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is not None:
            for k,v in input.items():
                input[k].cuda().record_stream(torch.cuda.current_stream())
        self.preload()
        return input