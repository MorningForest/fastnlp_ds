#-*- coding: utf-8 -*-
# Author: HW
# @Time: 2021/12/9 16:59
from typing import List, Union
from functools import wraps
import copy

import pyarrow as pa
import numpy as np

def _interpolation_search(arr: List[int], x: int) -> int:
    """
    Return the position i of a sorted array so that arr[i] <= x < arr[i+1]

    Args:
        arr (:obj:`List[int]`): non-empty sorted list of integers
        x (:obj:`int`): query

    Returns:
        `int`: the position i so that arr[i] <= x < arr[i+1]

    Raises:
        `IndexError`: if the array is empty or if the query is outside the array values
    """
    i, j = 0, len(arr) - 1
    while i < j and arr[i] <= x < arr[j]:
        k = i + ((j - i) * (x - arr[i]) // (arr[j] - arr[i]))
        if arr[k] <= x < arr[k + 1]:
            return k
        elif arr[k] < x:
            i, j = k + 1, j
        else:
            i, j = i, k
    raise IndexError(f"Invalid query '{x}' for size {arr[-1] if len(arr) else 'none'}.")

def inject_arrow_table_documentation(arrow_table_method):
    def wrapper(method):
        out = wraps(arrow_table_method)(method)
        out.__doc__ = out.__doc__.replace("pyarrow.Table", "Table")
        return out

    return wrapper

def _deepcopy(x, memo: dict):
    """deepcopy a regular class instance"""
    cls = x.__class__
    result = cls.__new__(cls)
    memo[id(x)] = result
    for k, v in x.__dict__.items():
        setattr(result, k, copy.deepcopy(v, memo))
    return result

def _in_memory_arrow_table_from_file(filename: str) -> pa.Table:
    in_memory_stream = pa.input_stream(filename)
    opened_stream = pa.ipc.open_stream(in_memory_stream)
    pa_table = opened_stream.read_all()
    return pa_table


def _in_memory_arrow_table_from_buffer(buffer: pa.Buffer) -> pa.Table:
    stream = pa.BufferReader(buffer)
    opened_stream = pa.ipc.open_stream(stream)
    table = opened_stream.read_all()
    return table

class IndexedTableMixin:
    def __init__(self, table: pa.Table):
        self._schema = table.schema
        self._batches = [recordbatch for recordbatch in table.to_batches() if len(recordbatch) > 0]
        self._offsets: np.ndarray = np.cumsum([0] + [len(b) for b in self._batches], dtype=np.int64)

    def fast_gather(self, indices: Union[List[int], np.ndarray]) -> pa.Table:
        """
        Create a pa.Table by gathering the records at the records at the specified indices. Should be faster
        than pa.concat_tables(table.fast_slice(int(i) % table.num_rows, 1) for i in indices) since NumPy can compute
        the binary searches in parallel, highly optimized C
        """
        assert len(indices), "Indices must be non-empty"
        batch_indices = np.searchsorted(self._offsets, indices, side="right") - 1
        return pa.Table.from_batches(
            [
                self._batches[batch_idx].slice(i - self._offsets[batch_idx], 1)
                for batch_idx, i in zip(batch_indices, indices)
            ],
            schema=self._schema,
        )

    def fast_slice(self, offset=0, length=None) -> pa.Table:
        """
        Slice the Table using interpolation search.
        The behavior is the same as :obj:`pyarrow.Table.slice` but it's significantly faster.

        Interpolation search is used to find the start and end indexes of the batches we want to keep.
        The batches to keep are then concatenated to form the sliced Table.
        """
        if offset < 0:
            raise IndexError("Offset must be non-negative")
        elif offset >= self._offsets[-1] or (length is not None and length <= 0):
            return pa.Table.from_batches([], schema=self._schema)
        i = _interpolation_search(self._offsets, offset)
        if length is None or length + offset >= self._offsets[-1]:
            batches = self._batches[i:]
            batches[0] = batches[0].slice(offset - self._offsets[i])
        else:
            j = _interpolation_search(self._offsets, offset + length - 1)
            batches = self._batches[i : j + 1]
            batches[-1] = batches[-1].slice(0, offset + length - self._offsets[j])
            batches[0] = batches[0].slice(offset - self._offsets[i])
        return pa.Table.from_batches(batches, schema=self._schema)

class Table(IndexedTableMixin):
    """
    Wraps a pyarrow Table by using composition.
    This is the base class for InMemoryTable, MemoryMappedTable and ConcatenationTable.

    It implements all the basic attributes/methods of the pyarrow Table class except
    the Table transforms: slice, filter, flatten, combine_chunks, cast, add_column,
    append_column, remove_column, set_column, rename_columns and drop.

    The implementation of these methods differs for the subclasses.
    """

    def __init__(self, table: pa.Table):
        super().__init__(table)
        self.table = table

    def __deepcopy__(self, memo: dict):
        # arrow tables are immutable, so there's no need to copy self.table
        # moreover calling deepcopy on a pyarrow table seems to make pa.total_allocated_bytes() decrease for some reason
        # by adding it to the memo, self.table won't be copied
        memo[id(self.table)] = self.table
        # same for the recordbatches used by the index
        memo[id(self._batches)] = list(self._batches)
        return _deepcopy(self, memo)

    def __getstate__(self):
        return {"table": self.table}
        # We can't pickle objects that are bigger than 4GiB, or it causes OverflowError
        # So we write the table on disk instead
        # if self.table.nbytes >= config.MAX_TABLE_NBYTES_FOR_PICKLING:
        #     table = self.table
        #     with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".arrow") as tmp_file:
        #         filename = tmp_file.name
        #         logger.debug(
        #             f"Attempting to pickle a table bigger than 4GiB. Writing it on the disk instead at {filename}"
        #         )
        #         _write_table_to_file(table=table, filename=filename)
        #         return {"path": filename}
        # else:
        #     return {"table": self.table}

    def __setstate__(self, state):
        table = state["table"]
        Table.__init__(self, table)
        # if "path" in state:
        #     filename = state["path"]
        #     logger.debug(f"Unpickling a big table from the disk at {filename}")
        #     table = _in_memory_arrow_table_from_file(filename)
        #     logger.debug(f"Removing temporary table file at {filename}")
        #     os.remove(filename)
        # else:
        #     table = state["table"]
        # Table.__init__(self, table)

    @inject_arrow_table_documentation(pa.Table.validate)
    def validate(self, *args, **kwargs):
        return self.table.validate(*args, **kwargs)

    @inject_arrow_table_documentation(pa.Table.equals)
    def equals(self, *args, **kwargs):
        args = tuple(arg.table if isinstance(arg, Table) else arg for arg in args)
        kwargs = {k: v.table if isinstance(v, Table) else v for k, v in kwargs}
        return self.table.equals(*args, **kwargs)

    @inject_arrow_table_documentation(pa.Table.to_batches)
    def to_batches(self, *args, **kwargs):
        return self.table.to_batches(*args, **kwargs)

    @inject_arrow_table_documentation(pa.Table.to_pydict)
    def to_pydict(self, *args, **kwargs):
        return self.table.to_pydict(*args, **kwargs)

    @inject_arrow_table_documentation(pa.Table.to_pandas)
    def to_pandas(self, *args, **kwargs):
        return self.table.to_pandas(*args, **kwargs)

    def to_string(self, *args, **kwargs):
        return self.table.to_string(*args, **kwargs)

    @inject_arrow_table_documentation(pa.Table.field)
    def field(self, *args, **kwargs):
        return self.table.field(*args, **kwargs)

    @inject_arrow_table_documentation(pa.Table.column)
    def column(self, *args, **kwargs):
        return self.table.column(*args, **kwargs)

    @inject_arrow_table_documentation(pa.Table.itercolumns)
    def itercolumns(self, *args, **kwargs):
        return self.table.itercolumns(*args, **kwargs)

    @property
    def schema(self):
        return self.table.schema

    @property
    def columns(self):
        return self.table.columns

    @property
    def num_columns(self):
        return self.table.num_columns

    @property
    def num_rows(self):
        return self.table.num_rows

    @property
    def shape(self):
        return self.table.shape

    @property
    def nbytes(self):
        return self.table.nbytes

    @property
    def column_names(self):
        return self.table.column_names

    def __eq__(self, other):
        return self.equals(other)

    def __getitem__(self, i):
        return self.table[i]

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        return self.table.__repr__().replace("pyarrow.Table", self.__class__.__name__)

    def __str__(self):
        return self.table.__str__().replace("pyarrow.Table", self.__class__.__name__)

    @inject_arrow_table_documentation(pa.Table.slice)
    def slice(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.filter)
    def filter(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.flatten)
    def flatten(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.combine_chunks)
    def combine_chunks(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.cast)
    def cast(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.replace_schema_metadata)
    def replace_schema_metadata(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.add_column)
    def add_column(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.append_column)
    def append_column(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.remove_column)
    def remove_column(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.set_column)
    def set_column(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.rename_columns)
    def rename_columns(self, *args, **kwargs):
        raise NotImplementedError()

    @inject_arrow_table_documentation(pa.Table.drop)
    def drop(self, *args, **kwargs):
        raise NotImplementedError()

class InMemoryTable(Table):
    """
    The table is said in-memory when it is loaded into the user's RAM.

    Pickling it does copy all the data using memory.
    Its implementation is simple and uses the underlying pyarrow Table methods directly.

    This is different from the MemoryMapped table, for which pickling doesn't copy all the
    data in memory. For a MemoryMapped, unpickling instead reloads the table from the disk.

    InMemoryTable must be used when data fit in memory, while MemoryMapped are reserved for
    data bigger than memory or when you want the memory footprint of your application to
    stay low.
    """

    @classmethod
    def from_file(cls, filename: str):
        table = _in_memory_arrow_table_from_file(filename)
        return cls(table)

    @classmethod
    def from_buffer(cls, buffer: pa.Buffer):
        table = _in_memory_arrow_table_from_buffer(buffer)
        return cls(table)

    @inject_arrow_table_documentation(pa.Table.from_pandas)
    @classmethod
    def from_pandas(cls, *args, **kwargs):
        return cls(pa.Table.from_pandas(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.from_arrays)
    @classmethod
    def from_arrays(cls, *args, **kwargs):
        return cls(pa.Table.from_arrays(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.from_pydict)
    @classmethod
    def from_pydict(cls, *args, **kwargs):
        return cls(pa.Table.from_pydict(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.from_batches)
    @classmethod
    def from_batches(cls, *args, **kwargs):
        return cls(pa.Table.from_batches(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.slice)
    def slice(self, offset=0, length=None):
        # Use fast slicing here
        return InMemoryTable(self.fast_slice(offset=offset, length=length))

    @inject_arrow_table_documentation(pa.Table.filter)
    def filter(self, *args, **kwargs):
        return InMemoryTable(self.table.filter(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.flatten)
    def flatten(self, *args, **kwargs):
        return InMemoryTable(self.table.flatten(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.combine_chunks)
    def combine_chunks(self, *args, **kwargs):
        return InMemoryTable(self.table.combine_chunks(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.cast)
    def cast(self, *args, **kwargs):
        return InMemoryTable(self.table.cast(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.replace_schema_metadata)
    def replace_schema_metadata(self, *args, **kwargs):
        return InMemoryTable(self.table.replace_schema_metadata(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.add_column)
    def add_column(self, *args, **kwargs):
        return InMemoryTable(self.table.add_column(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.append_column)
    def append_column(self, *args, **kwargs):
        return InMemoryTable(self.table.append_column(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.remove_column)
    def remove_column(self, *args, **kwargs):
        return InMemoryTable(self.table.remove_column(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.set_column)
    def set_column(self, *args, **kwargs):
        return InMemoryTable(self.table.set_column(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.rename_columns)
    def rename_columns(self, *args, **kwargs):
        return InMemoryTable(self.table.rename_columns(*args, **kwargs))

    @inject_arrow_table_documentation(pa.Table.drop)
    def drop(self, *args, **kwargs):
        return InMemoryTable(self.table.drop(*args, **kwargs))
