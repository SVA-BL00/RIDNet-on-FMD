import sys
import queue
import random

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

def _ms_worker_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, worker_id):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)

    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            idx_scale = 0
            if len(scale) > 1 and dataset.train:
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            result = list(samples) + [idx_scale]
            data_queue.put((idx, result))
        except Exception as e:
            data_queue.put((idx, e))

class _MSDataLoaderIter:
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.noise_g = loader.noise_g
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        
        self.sample_iter = iter(self.batch_sampler)
        self.index_queues = [
            multiprocessing.Queue() for _ in range(self.num_workers)
        ]
        self.worker_result_queue = multiprocessing.Queue()
        self.worker_queue_idx = 0
        self.send_idx = 0
        self.rcvd_idx = 0
        self.reorder_dict = {}
        self.batches_outstanding = 0

        base_seed = torch.randint(0, 2**31, (1,)).item()

        self.workers = []
        for i in range(self.num_workers):
            w = multiprocessing.Process(
                target=_ms_worker_loop,
                args=(
                    self.dataset,
                    self.index_queues[i],
                    self.worker_result_queue,
                    self.collate_fn,
                    self.noise_g,
                    base_seed + i,
                    i
                )
            )
            w.daemon = True
            w.start()
            self.workers.append(w)

        for _ in range(2 * self.num_workers):
            self._put_indices()
    
    def _put_indices(self):
        try:
            indices = next(self.sample_iter)
        except StopIteration:
            return
        
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1

    def __next__(self):
        if self.rcvd_idx >= self.send_idx:
            raise StopIteration

        while True:
            try:
                idx, data = self.worker_result_queue.get(timeout=5.0)
                break
            except queue.Empty:
                continue

        self.batches_outstanding -= 1

        if isinstance(data, Exception):
            raise data

        if idx != self.rcvd_idx:
            self.reorder_dict[idx] = data
            while self.rcvd_idx in self.reorder_dict:
                data = self.reorder_dict.pop(self.rcvd_idx)
                self.rcvd_idx += 1
                self._put_indices()
                return data
            return self.__next__()

        self.rcvd_idx += 1
        self._put_indices()
        return data

    def __iter__(self):
        return self

    def __del__(self):
        try:
            for q in self.index_queues:
                q.put(None)
            for w in self.workers:
                w.join(timeout=5.0)
                if w.is_alive():
                    w.terminate()
        except:
            pass

class MSDataLoader(DataLoader):
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=None, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None):

        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn,
            persistent_workers=False)

        self.noise_g = args.noise_g
        self._custom_iter = args.n_threads > 0

    def __iter__(self):
        if self._custom_iter:
            return _MSDataLoaderIter(self)
        else:
            return super(MSDataLoader, self).__iter__()
