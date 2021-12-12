#-*- coding: utf-8 -*-
# Author: HW
# @Time: 2021/12/9 16:58
from typing import Dict, List, Union, Optional, Callable, Iterable
import os
from copy import deepcopy
from functools import partial

import pyarrow as pa
from pyarrow import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, RLock

from .table import InMemoryTable

PathLike = Union[str, bytes, os.PathLike]


class Instance(object):
    r"""
    Instance是fastNLP中对应一个sample的类。每个sample在fastNLP中是一个Instance对象。
    Instance一般与 :class:`~fastNLP.DataSet` 一起使用, Instance的初始化如下面的Example所示::

        >>>from fastNLP import Instance
        >>>ins = Instance(field_1=[1, 1, 1], field_2=[2, 2, 2])
        >>>ins["field_1"]
        [1, 1, 1]
        >>>ins.add_field("field_3", [3, 3, 3])
        >>>ins = Instance(**{'x1': 1, 'x2':np.zeros((3, 4))})
    """

    def __init__(self, **fields):

        self.fields = fields

    def add_field(self, field_name, field):
        r"""
        向Instance中增加一个field

        :param str field_name: 新增field的名称
        :param Any field: 新增field的内容
        """
        self.fields[field_name] = field

    def items(self):
        r"""
        返回一个迭代器，迭代器返回两个内容，第一个内容是field_name, 第二个内容是field_value

        :return: 一个迭代器
        """
        return self.fields.items()

    def __contains__(self, item):
        return item in self.fields

    def __getitem__(self, name):
        if name in self.fields:
            return self.fields[name]
        else:
            raise KeyError("{} not found".format(name))

    def __setitem__(self, name, field):
        return self.add_field(name, field)

    def __repr__(self):
        return ''
        # return str(pretty_table_printer(self))

class Fcontainer():

    def __init__(self):
        self.field_array = {}
        ##python dict 模仿实现 table功能

    @property
    def column_names(self):
        pass


def _apply_single(pa_table=None, func: Optional[Callable] = None,
                  _apply_field=None, proc_id=0, desc=None):
    results = []
    class Iter_ptr:
        def __init__(self, dataset, idx):
            self.dataset = dataset
            self.idx = idx

        def __getitem__(self, item):
            assert item in self.dataset.column_names, f"no such field:{item} in datasets"

            assert self.idx < len(self.dataset), "index:{} out of range".format(self.idx)

            return self.dataset.column(item).slice(self.idx, 1).to_pylist()[0]

        def __setitem__(self, key, value):
            raise TypeError("You cannot modify value directly.")

        def items(self):
            ins = self.dataset.slice(self.idx, 1)
            return ins.to_pydict()

        def __repr__(self):
            return self.dataset.slice(self.idx, 1).to_string()

    def iter_fn():
        for i in range(len(pa_table)):
            yield Iter_ptr(pa_table, i)

    desc = desc if desc else "#{} ".format(proc_id)

    try:
        for idx, ins in tqdm(enumerate(iter_fn()), total=len(pa_table),
                             desc=desc, position=proc_id):
            if _apply_field is not None:
                per_out = func(ins[_apply_field])
            else:
                per_out = func(ins)
            if isinstance(per_out, dict):
                if len(results) == 0:
                    results = {k: [v] for k, v in per_out.items()}
                else:
                    for key, value in per_out.items():
                        results[key].append(value)
            elif isinstance(per_out, list):
                results.append(per_out)


    except:
        import traceback
        traceback.print_exc()

    return results

class DataSet:

    def __init__(self, data=None):
        self._data = None

        if data is not None:
            if isinstance(data, dict):
                length_set = set()
                for key, value in data.items():
                    length_set.add(len(value))
                assert len(length_set) == 1, "Arrays must all be same length."
                self._data: InMemoryTable = InMemoryTable.from_pydict(data)

            elif isinstance(data, list):
                for ins in data:
                    assert isinstance(ins, Instance), "Must be Instance type, not {}.".format(type(ins))
                    self.append(ins)

            elif isinstance(data, InMemoryTable):
                self._data = data

            elif isinstance(data, pa.Table):
                self._data = InMemoryTable(data)

            else:
                raise ValueError("data only be dict or list type.")

    @property
    def data(self):
        return self._data

    def _inner_iter(self):
        class Iter_ptr:
            def __init__(self, dataset, idx):
                self.dataset = dataset
                self.idx = idx

            def __getitem__(self, item):
                assert item in self.dataset.data.column_names, f"no such field:{item} in datasets"

                assert self.idx < len(self.dataset), "index:{} out of range".format(self.idx)

                return self.dataset.data.column(item).slice(self.idx, 1).to_pylist()[0]

            def __setitem__(self, key, value):
                raise TypeError("You cannot modify value directly.")

            def items(self):
                ins = self.dataset.data.slice(self.idx, 1)
                return ins.to_pydict()

            def __repr__(self):
                return self.dataset.data.slice(self.idx, 1).to_string()

        def inner_iter_func():
            for idx in range(len(self)):
                yield Iter_ptr(self, idx)

        return inner_iter_func()

    def has_field(self, field_name: str) -> bool:
        r"""
            判断DataSet中是否有名为field_name这个field

            :param str field_name: field的名称
            :return bool: 表示是否有名为field_name这个field
        """
        if field_name in self._data.column_names:
            return True
        return False

    def append(self, instance: Instance):
        r"""
           将一个instance对象append到DataSet后面。

           :param ~fastNLP.Instance instance: 若DataSet不为空，则instance应该拥有和DataSet完全一样的field。

        """
        if len(self) == 0:
            self._data = InMemoryTable.from_pydict(mapping=instance.fields)
        else:
            if self._data.num_columns != len(instance.fields):
                raise ValueError(
                    "DataSet object has {} fields, but attempt to append an Instance object with {} fields."
                        .format(len(self._data.num_columns), len(instance.fields)))
            col_names = sorted(self.get_field_names())
            ins_col_names = sorted([key for key, _ in instance.items()])
            for ind in range(len(col_names)):
                if col_names[ind] != ins_col_names[ind]:
                    raise ValueError(
                        f"DataSet fields name:{col_names[ind]} != Instance field name: {ins_col_names[ind]}"
                    )
            data = self._data.to_pydict()
            for key, value in instance.items():
                data[key].append(value)
            self._data = InMemoryTable.from_pydict(mapping=data)

    def add_field(self, field_name: str, fields: Union[list, np.array]):
        r"""
            新增一个field
            :param str field_name: 新增的field的名称
            :param list fields: 需要新增的field的内容
        """
        if field_name in self._data.column_names:
            raise ValueError(
                f"Original column name {field_name} in the dataset. "
            )
        assert len(fields) == len(self), f"The field to add must have the same size as dataset." \
                                              f"Dataset size {len(self)} != field size {len(fields)}"
        # column_table = InMemoryTable.from_pydict({field_name: fields})
        self._data = self._data.append_column(field_name, pa.array(fields))

    def delete_instance(self, index: int):
        r"""
            删除第index个instance

            :param int index: 需要删除的instance的index，序号从0开始。
        """
        assert isinstance(index, int), "Only integer supported."
        if len(self) <= index:
            raise IndexError("{} is too large for as DataSet with {} instances.".format(index, len(self)))
        mask = [True] * len(self)
        mask[index] = False
        self._data = self._data.filter(mask=mask)
        return self

    def delete_field(self, field_name: Union[str, List[str]]):
        r"""
        删除名为field_name的field

        :param str field_name: 需要删除的field的名称.
        """
        if isinstance(field_name, str):
            field_name = [field_name]

        for column_name in field_name:
            if column_name not in self._data.column_names:
                raise ValueError(
                    f"Column name {column_name} not in the dataset. "
                    f"Current columns in the dataset: {self._data.column_names}"
                )
        self._data = self._data.drop(field_name)

        return self

    def copy_field(self, field_name: str, new_field_name: str):
        r"""
        深度copy名为field_name的field到new_field_name

        :param str field_name: 需要copy的field。
        :param str new_field_name: copy生成的field名称
        :return: self
        """
        if not self.has_field(field_name):
            raise KeyError(f"Field:{field_name} not found in DataSet.")
        field_array = self.get_field(field_name)
        self.add_field(new_field_name, field_array)
        return self

    def get_field(self, field_name: str) -> List:
        r"""
        获取field_name这个field

        :param str field_name: field的名称
        :return:
        """
        if field_name not in self._data.column_names:
            raise ValueError(
                f"Original column name {field_name} not in the dataset. "
            )
        return self._data.column(field_name).to_pylist()

    def get_all_fields(self) -> dict:
        r"""
        返回一个dict，key为field_name, value为对应的 :class:`~fastNLP.FieldArray`

        :return dict: 返回如上所述的字典
        """
        return self._data.to_pydict()

    def get_field_names(self) -> list:
        r"""
        返回一个list，包含所有 field 的名字

        :return list: 返回如上所述的列表
        """
        return self._data.column_names

    def get_length(self):
        r"""
        获取DataSet的元素数量

        :return: int: DataSet中Instance的个数。
        """
        return len(self._data)

    def rename_field(self, field_name, new_field_name):
        r"""
        将某个field重新命名.

        :param str field_name: 原来的field名称。
        :param str new_field_name: 修改为new_name。
        """
        if field_name not in self._data.column_names:
            raise ValueError(
                f"Original column name {field_name} not in the dataset. "
                f"Current columns in the dataset: {self._data.column_names}"
            )
        if new_field_name in self._data.column_names:
            raise ValueError(
                f"New column name {new_field_name} already in the dataset. "
                f"Please choose a column name which is not already in the dataset. "
                f"Current columns in the dataset: {self._data.column_names}"
            )
        if not new_field_name:
            raise ValueError("New column name is empty.")

        def rename(columns):
            return [new_field_name if col == field_name else col for col in columns]

        new_column_names = rename(self._data.column_names)

        self._data = self._data.rename_columns(new_column_names)
        return self

    def _add_apply_field(self, results, new_field_name: str):
        r"""
           将results作为加入到新的field中，field名称为new_field_name

           :param List[str] results: 一般是apply*()之后的结果
           :param str new_field_name: 新加入的field的名称
           :return:
       """
        if new_field_name in self._data.column_names:
            self.delete_field(new_field_name)
        self.add_field(new_field_name, results)

    def apply_field(self, func, field_name, new_field_name=None, num_proc=0, **kwargs):
        r"""
           将DataSet中的每个instance中的名为 `field_name` 的field传给func，并获取它的返回值。

           :param callable func: input是instance中名为 `field_name` 的field的内容。
           :param str field_name: 传入func的是哪个field。
           :param None,str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
               盖之前的field。如果为None则不创建新的field。
        """
        assert len(self) != 0, "Null DataSet cannot use apply_field()."
        if not self.has_field(field_name=field_name):
            raise KeyError("DataSet has no field named `{}`.".format(field_name))
        return self.apply(func, new_field_name, _apply_field=field_name, num_proc=num_proc,**kwargs)

    def apply_field_more(self, func, field_name, modify_fields=True, num_proc=0,**kwargs):
        r"""
        将 ``DataSet`` 中的每个 ``Instance`` 中的名为 `field_name` 的field 传给 func，并获取它的返回值。
        func 可以返回一个或多个 field 上的结果。

        .. note::
            ``apply_field_more`` 与 ``apply_field`` 的区别参考 :meth:`~fastNLP.DataSet.apply_more` 中关于 ``apply_more`` 与
            ``apply`` 区别的介绍。

        :param callable func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param str field_name: 传入func的是哪个field。
        :param bool modify_fields: 是否用结果修改 `DataSet` 中的 `Field`， 默认为 True
        :param optional kwargs:
            1. use_tqdm: bool, 是否使用tqdm显示预处理进度

            2. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称

        :return Dict[str:Field]: 返回一个字典
        """
        assert len(self) != 0, "Null DataSet cannot use apply_field()."
        if not self.has_field(field_name=field_name):
            raise KeyError("DataSet has no field named `{}`.".format(field_name))
        return self.apply_more(func, modify_fields, _apply_field=field_name, num_proc=num_proc, **kwargs)

    def apply_more(self, func: Callable = None, modify_fields=True, num_proc: int = 0, **kwargs):
        r"""
        将 ``DataSet`` 中每个 ``Instance`` 传入到func中，并获取它的返回值。func可以返回一个或多个 field 上的结果。

        .. note::
            ``apply_more`` 与 ``apply`` 的区别：

            1. ``apply_more`` 可以返回多个 field 的结果， ``apply`` 只可以返回一个field 的结果；

            2. ``apply_more`` 的返回值是一个字典，每个 key-value 对中的 key 表示 field 的名字，value 表示计算结果；

            3. ``apply_more`` 默认修改 ``DataSet`` 中的 field ，``apply`` 默认不修改。

        :param callable func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param bool modify_fields: 是否用结果修改 ``DataSet`` 中的 ``Field`` ， 默认为 True
        :param optional kwargs:
            1. use_tqdm: bool, 是否使用tqdm显示预处理进度

            2. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称

        :return Dict[str:Field]: 返回一个字典
        """
        # 返回 dict , 检查是否一直相同
        assert callable(func), "The func you provide is not callable."
        assert len(self) != 0, "Null DataSet cannot use apply()."
        assert num_proc >= 0, "num_proc must >= 0"
        try:
            results = {}
            if num_proc == 0:
                results = _apply_single(self, func, kwargs.get("_apply_field", None), desc=kwargs.get("tqdm_desc", "Main"))
            else:
                if num_proc is not None and num_proc > len(self):
                    num_proc = len(self)
                    ##取出列数据, 分块
                shard_len = len(self) // num_proc
                shard_data = [self._data.table.slice(i * shard_len, shard_len) for i in range(num_proc - 1)]
                shard_data.append(self._data.table.slice((num_proc - 1) * shard_len))
                partial_single_map = partial(_apply_single,
                                             func=func,
                                             _apply_field=kwargs.get('_apply_field', None),
                                             )
                pool = Pool(processes=num_proc, initializer=tqdm.set_lock, initargs=(RLock(),))
                res = [(i, pool.apply_async(partial_single_map, kwds={'pa_table': shard_data[i], 'proc_id': i}))
                       for i in range(num_proc)]
                pool.close()
                pool.join()

                apply_out = []
                for idx, async_result in res:
                    apply_out.append(async_result.get())
                if isinstance(apply_out[0], dict):
                    dict_to_table = []
                    for i in range(num_proc):
                        dict_to_table.append(pa.Table.from_pydict(mapping=apply_out[i]))
                    results = pa.concat_tables(dict_to_table).to_pydict()


        except Exception as e:
            import traceback
            traceback.print_exc()
            # if idx != -1:
            #     if isinstance(e, ApplyResultException):
            #         logger.error(e.msg)
            #     logger.error("Exception happens at the `{}`th instance.".format(idx))
            raise e

        if modify_fields is True:
            for field, result in results.items():
                self._add_apply_field(result, field)

        return results

    def apply(self, func: Optional[Callable] = None, new_field_name: str = None, num_proc: int =0, **kwargs):
        r"""
            将DataSet中每个instance传入到func中，并获取它的返回值.

            :param callable func: 参数是 ``DataSet`` 中的 ``Instance``
            :param None,str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
                盖之前的field。如果为None则不创建新的field。
            :param optional kwargs: 支持输入
                1. use_tqdm: bool, 是否使用tqdm显示预处理进度
                2. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称
            :return
        """
        assert callable(func), "The func you provide is not callable."
        assert len(self) != 0, "Null DataSet cannot use apply()."
        results = []
        try:
            if num_proc == 0:
                results = _apply_single(self._data, func, _apply_field=None, desc=kwargs.get("tqdm_desc", "Main"))
            else:
                if num_proc is not None and num_proc > len(self):
                    num_proc = len(self)
                ##取出列数据, 分块
                shard_len = len(self) // num_proc
                shard_data = [self._data.table.slice(i*shard_len, shard_len) for i in range(num_proc - 1)]
                shard_data.append(self._data.table.slice((num_proc-1)*shard_len))
                partial_single_map = partial(_apply_single,
                                             func=func,
                                             _apply_field=kwargs.get('_apply_field', None),
                                             )
                pool = Pool(processes=num_proc, initializer=tqdm.set_lock, initargs=(RLock(), ))
                res = [(i, pool.apply_async(partial_single_map, kwds={'pa_table': shard_data[i], 'proc_id': i}))
                       for i in range(num_proc)]
                pool.close()
                pool.join()

                for index, async_result in res:
                    data = async_result.get()
                    results.append(data)

                if isinstance(results[0], list):
                    results = sum(results, [])
                if isinstance(results[0], dict):
                    dict_to_table = []
                    for i in range(num_proc):
                        dict_to_table.append(pa.Table.from_pydict(mapping=results[i]))
                    results = pa.concat_tables(dict_to_table).to_pydict()

        except:
            import traceback
            print(traceback.print_exc())

        if new_field_name is not None:
            if isinstance(results, list):
                self._add_apply_field(results, new_field_name)
        if isinstance(results, dict):
            for k, v in results.items():
                self._add_apply_field(v, k)

        return results

    def apply_batch(self):
        pass

    def drop(self, func, inplace=True):
        r"""
            func接受一个Instance，返回bool值。返回值为True时，该Instance会被移除或者不会包含在返回的DataSet中。

            :param callable func: 接受一个Instance作为参数，返回bool值。为True时删除该instance
            :param bool inplace: 是否在当前DataSet中直接删除instance；如果为False，将返回一个新的DataSet。

            :return: DataSet
        """
        if inplace:
            results = [func(ins) for ins in self._inner_iter()]
            self._data = self._data.filter(results)
            return self
        else:
            results = [func(ins) for ins in self._inner_iter()]
            if len(results) != 0:
                new_data = self._data.filter(results)
                return DataSet(new_data)
            else:
                return DataSet()

    def concat(self, dataset, inplace=True, field_mapping=None):
        """
            将当前dataset与输入的dataset结合成一个更大的dataset，需要保证两个dataset都包含了相同的field。
            当dataset中包含的field多于当前的dataset，则多余的field会被忽略；若dataset中未包含所有
            当前dataset含有field，则会报错。

            :param DataSet, dataset: 需要和当前dataset concat的dataset
            :param bool, inplace: 是否直接将dataset组合到当前dataset中
            :param dict, field_mapping: 当dataset中的field名称和当前dataset不一致时，需要通过field_mapping把输入的dataset中的field
                名称映射到当前field. field_mapping为dict类型，key为dataset中的field名称，value是需要映射成的名称

            :return: DataSet
        """
        assert isinstance(dataset, DataSet), "Can only concat two datasets."
        fns_in_this_dataset = set(self.get_field_names())
        fns_in_other_dataset = dataset.get_field_names()
        if field_mapping is not None:
            fns_in_other_dataset = [field_mapping.get(fn, fn) for fn in fns_in_other_dataset]
            fns_in_other_dataset = set(fns_in_other_dataset)
        fns_in_other_dataset = set(fns_in_other_dataset)
        fn_not_seen = list(fns_in_this_dataset - fns_in_other_dataset)
        if fn_not_seen:
            raise RuntimeError(f"The following fields are not provided in the dataset:{fn_not_seen}")
        if inplace:
            ds = self
        else:
            ds = deepcopy(self)

        ds_data = InMemoryTable(pa.concat_tables([ds.data.table, dataset.data.table]))
        return DataSet(ds_data)

    def split(self, ratio, shuffle=True):
        r"""
            将DataSet按照ratio的比例拆分，返回两个DataSet

            :param float ratio: 0<ratio<1, 返回的第一个DataSet拥有 `(1-ratio)` 这么多数据，第二个DataSet拥有`ratio`这么多数据
            :param bool shuffle: 在split前是否shuffle一下
        """
        assert len(self) > 1, f'DataSet with {len(self)} instance cannot be split.'
        assert isinstance(ratio, float)
        assert 0 < ratio < 1
        all_indices = [_ for _ in range(len(self))]
        if shuffle:
            np.random.shuffle(all_indices)
        split = int(ratio * len(self))
        if split == 0:
            error_msg = f'Dev DataSet has {split} instance after split.'
            # logger.error(error_msg)
            raise IndexError(error_msg)
        dev_indices = all_indices[:split]
        train_indices = all_indices[split:]
        # dev_set = DataSet()
        # train_set = DataSet()
        dev_list, train_list = [], []
        for idx in dev_indices:
            dev_list.append(self._data.table.slice(idx, 1))
        dev_set = DataSet(pa.concat_tables(dev_list))
            # dev_set.append(self._data.slice(idx, 1))
        for idx in train_indices:
            train_list.append(self._data.table.slice(idx, 1))
            # train_set.append(self._data.slice(idx, 1))
        train_set = DataSet(pa.concat_tables(train_list))
        return train_set, dev_set

    def to_csv(self, path: PathLike):
        pass

    def to_pandas(self) -> pd.DataFrame:
        return self._data.to_pandas()

    def to_parquet(self, path: PathLike):
        pass

    @classmethod
    def from_dict(cls, data: dict):
        return cls(InMemoryTable.from_pydict(mapping=data))

    @classmethod
    def from_parquet(path_or_paths: Union[PathLike, List[PathLike]]
                     ):
        pass

    @classmethod
    def from_pandas(cls, df: pd.DataFrame):
        return cls(InMemoryTable.from_pandas(df=df))

    @classmethod
    def from_csv(cls, path_or_paths):
        return cls(InMemoryTable(pa.csv.read_csv(path_or_paths)))

    @classmethod
    def from_json(path_or_paths):
        pass

    @classmethod
    def from_text(path_or_paths: Union[str, bytes, os.PathLike],
                  split):
        pass

    def to_fastNLPDataSet(self):
        self._data = Fcontainer.from_table(self._data)

    def __len__(self):
        r"""Fetch the length of the dataset.

        :return length:
        """
        if self._data is not None:
            return len(self._data)
        return 0

    def __contains__(self, item):
        return item in self.get_field_names()

    def __iter__(self):
        def iter_func():
            for idx in range(len(self)):
                yield self[idx]
        return iter_func()

    def __repr__(self):
        return super().__repr__()

    def __getitem__(self, idx):
        r"""给定int的index，返回一个Instance; 给定slice，返回包含这个slice内容的新的DataSet。

        :param idx: can be int or slice.
        :return: If `idx` is int, return an Instance object.
                If `idx` is slice, return a DataSet object.
        """
        if isinstance(idx, int):
            if len(self) <= idx:
                raise ValueError(
                    f"out of DataSet"
                )
            return self._data.fast_slice(idx, 1).to_pydict()
        elif isinstance(idx, slice):
            if idx.start is not None and (idx.start >= len(self) or idx.start <= -len(self)):
                raise RuntimeError(f"Start index {idx.start} out of range 0-{len(self) - 1}")
            idx = range(*idx.indices(self._data.num_rows))
            if idx.start > 0 and idx.step == 1 and idx.stop > idx.start:
                return DataSet(self._data.fast_slice(idx.start, idx.stop-idx.start))
            else:
                pass
        elif isinstance(idx, str):
            if idx not in self.get_field_names():
                raise KeyError("No such field called {} in DataSet.".format(idx))
            return self._data.column(idx).to_pylist()

        elif isinstance(idx, list):
            pass
            # dataset = DataSet()
            # for i in idx:
            #     assert isinstance(i, int), "Only int index allowed."
            #     instance = self[i]
            #     dataset.append(instance)
            # for field_name, field in self.field_arrays.items():
            #     dataset.field_arrays[field_name].to(field)
            # dataset.collater = self.collater.copy_from(self.collater)
            # return dataset
        else:
            raise KeyError("Unrecognized type {} for idx in __getitem__ method".format(type(idx)))

    def __getattr__(self, item):
        pass

    def __getstate__(self):
        return {'table': self._data.table}

    def __setstate__(self, state):
        table = state["table"]
        DataSet.__init__(self, table)

from transformers import BertTokenizerFast
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

def process(instance):
    label2id = {'neg': 0, 'pos': 1}
    res = tok(instance['text'], max_length=128, truncation=True)
    id = label2id[instance['label']]
    return {'input_ids': res['input_ids'], 'label': id}

if __name__ == '__main__':
    # ds = DataSet({'text': ['hello', 'he'], 'label': [1, 2]})
    # print(ds.get_field_names())
    # print(ds.get_length())
    # print(len(ds))
    # print(ds.rename_field('label', '1').get_field_names())
    # print(ds.get_field('text'))
    # ds.apply_field(lambda x: x, field_name='text', new_field_name='text1')
    # print(ds.data)
    # print(ds.get_all_fields())
    # ds = ds.delete_instance(0)
    import pickle
    from fastNLP import DataSet as FDataSet
    import time
    table = pa.csv.read_csv("../data/yelp_train.csv").remove_column(0)
    ds = DataSet(table)
    start = time.time()
    ds.data.to_pydict()
    end = time.time()
    print(end-start)
    # print(ds.data)
    # fp = open("ds.pkl", "wb")
    # fp1 = open("fds.pkl", "wb")
    # pickle.dump(ds.data, fp)
    # fds = FDataSet(table.to_pydict())
    # pickle.dump(fds, fp1)
    # res = ds.apply(process, num_proc=2)
    # print(len(res))
    # print(ds.data)
