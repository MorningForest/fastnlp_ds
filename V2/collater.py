#-*- coding: utf-8 -*-
# Author: HW
# @Time: 2021/12/9 16:59
from typing import Any, Dict, Optional, List, Callable
from numbers import Number
from collections import defaultdict
import copy

import numpy as np
import torch

class ApplyResultException(Exception):
    def __init__(self, msg, index=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了

class SetInputOrTargetException(Exception):
    def __init__(self, msg, index=None, field_name=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了
        self.field_name = field_name  # 标示当前field的名称

def _get_ele_type_and_dim(cell: Any, dim=0):
    r"""
    识别cell的类别与dimension的数量

    numpy scalar type:https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
    :param cell:
    :param dim:
    :return:
    """
    if isinstance(cell, (str, Number, np.bool_)):
        if hasattr(cell, 'dtype'):
            return cell.dtype.type, dim
        return type(cell), dim
    elif isinstance(cell, list):
        dim += 1
        res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    elif isinstance(cell, torch.Tensor):
        return cell.dtype, cell.dim() + dim  # 如果是torch.mean的结果是0
    elif isinstance(cell, np.ndarray):
        if cell.dtype != np.dtype('O'):  # 如果不是object的话说明是well-formatted的了
            return cell.dtype.type, cell.ndim + dim  # dtype.type返回的会是np.int32, np.float等
        # 否则需要继续往下iterate
        dim += 1
        res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    else:  # 包含tuple, set, dict以及其它的类型
        raise SetInputOrTargetException(f"Cannot process type:{type(cell)}.")

def _get_ds_type_dim(ds: dict):
    ## 获取数据集第一行的field内部函数的类型和维度
    field_dtype, field_dim = {}, {}
    for field_name, field_content in ds.items():
        type_0, dim_0 = _get_ele_type_and_dim(field_content)
        field_dtype[field_name], field_dim[field_name] = type_0, dim_0
    return field_dtype, field_dim


class Collator:
    r"""
        辅助DataLoader管理collate_fn的类

    """

    def __init__(self, collate_fn: Callable = None):
        ##初始化， 传入callta_fn函数
        self.collate_fn = collate_fn

    def __call__(self, ins_lst: List) -> Any:
        raise NotImplementedError

    def set_input(self, *args, **kwargs):
        raise NotImplementedError

    def set_pad_value(self, *args, **kwargs):
        raise NotImplementedError

class AutoCollator(Collator):
    def __init__(self):
        '''
        :param field_dtype: {'field_name':int}
        :param field_cell_ndim: {'field_dim': 2}
        '''
        super(AutoCollator, self).__init__()
        self.pad_field_value = defaultdict(int)  # field padding 自定义的 padding 值, 默认为0
        self.field_dtype = None  # 每列数据单元的dtype类型
        self.field_cell_ndim = None  ##每列数据单元维度

    def __call__(self, ins_lst: List[Dict]) -> dict:
        # 第一种情况，设置了set_input的值
        # 第二种情况， 根据数据的类型的判断是否padding
        if self.field_dtype is None and self.field_cell_ndim is None:
            self.field_dtype, self.field_cell_ndim = _get_ds_type_dim(ins_lst[0])

        pack_ins_lst, pad_ins_lst = {field_name: [] for field_name in ins_lst[0].keys()}, {}
        ## 将list列表内数据按列名打包
        for per_ins in ins_lst:
            for field_name in per_ins.keys():
                pack_ins_lst[field_name].append(per_ins[field_name])
        pad_ins_lst = copy.deepcopy(pack_ins_lst)
        if len(self.pad_field_value.keys()) > 0:
            ## 去掉不需要pad的列，如果set_input的列不存在则忽略
            drop_field_names = list(set(ins_lst[0].keys()) - set(self.pad_field_value.keys()))
            for field_name in drop_field_names:
                pack_ins_lst.pop(field_name)
                pad_ins_lst[field_name] = np.array(pad_ins_lst[field_name])

            for field_name, field_array in pack_ins_lst.items():
                content = pad_content(field_array, field_name, self.field_dtype[field_name],
                                      self.field_cell_ndim[field_name],
                                      self.pad_field_value[field_name])
                pad_ins_lst[field_name] = content

        else:
            ## 取出每列的数据，根据类型判断是否能pad
            for field_name, field_array in pack_ins_lst.items():
                pad_field_array = pad_content(field_array, field_name, self.field_dtype[field_name],
                                              self.field_cell_ndim[field_name],
                                              self.pad_field_value[field_name])
                pad_ins_lst[field_name] = pad_field_array
        return pad_ins_lst


    def set_padding(self, *field_names: str, value=0):
        for field_name in field_names:
            self.pad_field_value[field_name] = value


def pad_content(content, field_name, field_type, field_dim, pad_val):
    pad_val = pad_val if pad_val else 0

    if field_type:
        ## 不处理， 返回np.array类型
        if field_dim > 3:
            return np.array(content)
        ##元素类型为数值类型np.int64, np.float64, int, float等
        if isinstance(field_type, type) and \
                (issubclass(field_type, np.number) or issubclass(field_type, Number)):
            if field_dim == 0:
                array = np.array(content, dtype=field_type)
            elif field_dim == 1:
                max_len = max(map(len, content))
                array = np.full((len(content), max_len), pad_val, dtype=field_type)
                for i, content_i in enumerate(content):
                    array[i, :len(content_i)] = content_i
            elif field_dim == 2:
                max_len = max(map(len, content))
                max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                    content_i in content])
                array = np.full((len(content), max_len, max_word_len), pad_val, dtype=field_type)
                for i, content_i in enumerate(content):
                    for j, content_ii in enumerate(content_i):
                        array[i, j, :len(content_ii)] = content_ii
            else:
                shape = np.shape(content)
                if len(shape) == 4:  # 说明各dimension是相同的大小
                    array = np.array(content, dtype=field_type)
                else:
                    raise RuntimeError(
                        f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
            return array
        ##元素类型为数值类型torch.float等
        elif str(field_type).startswith('torch'):
            if field_dim == 0:
                tensor = torch.tensor(content).to(field_type)
            elif field_dim == 1:
                max_len = max(map(len, content))
                tensor = torch.full((len(content), max_len), fill_value=pad_val, dtype=field_type)
                for i, content_i in enumerate(content):
                    tensor[i, :len(content_i)] = content_i.clone().detach()
            elif field_dim == 2:
                max_len = max(map(len, content))
                max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                    content_i in content])
                tensor = torch.full((len(content), max_len, max_word_len), fill_value=pad_val,
                                    dtype=field_type)
                for i, content_i in enumerate(content):
                    for j, content_ii in enumerate(content_i):
                        tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
            else:
                shapes = set([np.shape(content_i) for content_i in content])
                if len(shapes) > 1:
                    raise RuntimeError(
                        f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                shape = shapes.pop()
                if len(shape) == 3:
                    tensor = torch.full([len(content)] + list(shape), fill_value=pad_val,
                                        dtype=field_type)
                    for i, content_i in enumerate(content):
                        tensor[i] = content_i.clone().detach().to(field_type)
                else:
                    raise RuntimeError(
                        f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
            return tensor
        else:
            return np.array(content)  # 不进行任何操作
    else:
        return np.array(content)
