from typing import Any, Dict, Union, List, ClassVar, Tuple, Generic, Callable, Optional
import sys
from typing_extensions import Literal, Final
from io import BytesIO as StringIO
import dill
import pickle
import contextlib
import xxhash

class _CloudPickleTypeHintFix:
    """
    Type hints can't be properly pickled in python < 3.7
    CloudPickle provided a way to make it work in older versions.
    This class provide utilities to fix pickling of type hints in older versions.
    from https://github.com/cloudpipe/cloudpickle/pull/318/files
    """

    def _is_parametrized_type_hint(obj):
        # This is very cheap but might generate false positives.
        origin = getattr(obj, "__origin__", None)  # typing Constructs
        values = getattr(obj, "__values__", None)  # typing_extensions.Literal
        type_ = getattr(obj, "__type__", None)  # typing_extensions.Final
        return origin is not None or values is not None or type_ is not None

    def _create_parametrized_type_hint(origin, args):
        return origin[args]

    def _save_parametrized_type_hint(pickler, obj):
        # The distorted type check sematic for typing construct becomes:
        # ``type(obj) is type(TypeHint)``, which means "obj is a
        # parametrized TypeHint"
        if type(obj) is type(Literal):  # pragma: no branch 文字类型
            initargs = (Literal, obj.__values__)
        elif type(obj) is type(Final):  # pragma: no branch
            initargs = (Final, obj.__type__)
        elif type(obj) is type(ClassVar):
            initargs = (ClassVar, obj.__type__)
        elif type(obj) in [type(Union), type(Tuple), type(Generic)]:
            initargs = (obj.__origin__, obj.__args__)
        elif type(obj) is type(Callable):
            args = obj.__args__
            if args[0] is Ellipsis:
                initargs = (obj.__origin__, args)
            else:
                initargs = (obj.__origin__, (list(args[:-1]), args[-1]))
        else:  # pragma: no cover
            raise pickle.PicklingError("Datasets pickle Error: Unknown type {}".format(type(obj)))
        pickler.save_reduce(_CloudPickleTypeHintFix._create_parametrized_type_hint, initargs, obj=obj)

@contextlib.contextmanager
def temporary_assignment(obj, attr, value):
    """Temporarily assign obj.attr to value."""
    original = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, original)

# @contextlib.contextmanager
# def _no_cache_fields(obj):
#     try:
#         if (
#             "PreTrainedTokenizerBase" in [base_class.__name__ for base_class in type(obj).__mro__]
#             and hasattr(obj, "cache")
#             and isinstance(obj.cache, dict)
#         ):
#             with temporary_assignment(obj, "cache", {}):
#                 yield
#         else:
#             yield
#
#     except ImportError:
#         yield

class Pickler(dill.Pickler):
    """Same Pickler as the one from dill, but improved for notebooks and shells"""

    dispatch = dill._dill.MetaCatchingDict(dill.Pickler.dispatch.copy())

    def save_global(self, obj, name=None):
        if sys.version_info[:2] < (3, 7) and _CloudPickleTypeHintFix._is_parametrized_type_hint(
            obj
        ):  # noqa  # pragma: no branch
            # Parametrized typing constructs in Python < 3.7 are not compatible
            # with type checks and ``isinstance`` semantics. For this reason,
            # it is easier to detect them using a duck-typing-based check
            # (``_is_parametrized_type_hint``) than to populate the Pickler's
            # dispatch with type-specific savers.
            _CloudPickleTypeHintFix._save_parametrized_type_hint(self, obj)
        else:
            dill.Pickler.save_global(self, obj, name=name)

def dumps(obj):
    """pickle an object to a string"""
    file = StringIO()
    dump(obj, file)
    return file.getvalue()

def dump(obj, file):
    """pickle an object to a file"""
    Pickler(file, recurse=True).dump(obj)
    return

class Hasher:
    """Hasher that accepts python objects as inputs."""

    dispatch: Dict = {}

    def __init__(self):
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: Union[bytes, List[bytes]]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash_default(cls, value: Any) -> str:
        return cls.hash_bytes(dumps(value))

    @classmethod
    def hash(cls, value: Any) -> str:
        if type(value) in cls.dispatch:
            return cls.dispatch[type(value)](cls, value)
        else:
            return cls.hash_default(value)

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()

def detect_fn(fn: Optional[Callable] = None, fn_kwargs: Optional[dict] = None):
    if fn_kwargs is None:
        fn_kwargs = {}
    hasher = Hasher()
    try:
        hasher.update(fn)
    except:
        print(f"fn：{fn} can not be hashed.")
        return None
    for key in sorted(fn_kwargs):
        hasher.update(key)
        try:
            hasher.update(fn_kwargs[key])
        except:
            print(
                f"Parameter '{key}'={fn_kwargs[key]} of the fn {fn} couldn't be hashed properly."
                )
            return None
    return hasher.hexdigest()
