#-*- coding: utf-8 -*-
# Author: HW
# @Time: 2021/12/9 16:59
from typing import Optional,  Callable, Sequence, List, Union

from torch.utils.data import DataLoader, Dataset, Sampler

from .collater import AutoCollator


class _MultiCollator:
    def __init__(self, collate_fns: Union[Callable, List[Callable]]):
        if isinstance(collate_fns, Callable):
            collate_fns = [collate_fns]

        self.collators: list = collate_fns

    def __call__(self, ins_lst):
        results = []
        for _collate_fn in self.collators:
            results.append(_collate_fn(ins_lst))
        out = results[0]
        for idx, res in enumerate(results[1:]):
            if isinstance(res, dict):
                out.update(res)
            else:
                raise TypeError(f"the return type of {idx} collate_fn is {type(res)}, but require is dict")

        return out

    def add_collater(self, collater):
        self.collators.append(collater)


class FDL(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Optional[Sampler[int]] = None,
                 batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                 num_workers: int = 0, collate_fn: Optional[Callable] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[Callable] = None,
                 multiprocessing_context=None, generator=None, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                       batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn,
                       pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                       multiprocessing_context=multiprocessing_context, generator=generator, prefetch_factor=prefetch_factor,
                       persistent_workers=persistent_workers)

        self.collate_fn = _MultiCollator(AutoCollator())
        if collate_fn is not None:
            self.collate_fn.add_collater(collate_fn)


    def set_padding(self, *field_names, val=0):
        self.collate_fn.collators[0].set_padding(field_names, val)

    def set_collator(self, collater: Callable):
        self.collate_fn = collater

    def add_collator(self, collater):
        self.collate_fn.add_collater(collater)

    def __iter__(self):
        return super().__iter__()


