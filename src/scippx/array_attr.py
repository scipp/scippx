# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock


def rewrap_result(wrap):

    def decorator(callable):

        def func(*args, **kwargs):
            # args and kwargs passed to `wrap` since it may need to call with same args
            # for extra_cols
            return wrap(callable(*args, **kwargs), *args, **kwargs)

        return func

    return decorator


# Case 1:
# - MultiMaskArray calls Quantity.__array_getattr__('units')
# - It never calls `wrap` so this case is not handled here
# Case 2:
# - MultiMaskArray calls BinEdgeArray.__array_getattr__('left')
# - Wrapper should apply left to all masks
# - Call MultiMaskArray._rewrap_content
def make_wrap(wrap, attr, extra_cols=None):

    def do_wrap(obj, *args, **kwargs):
        # case rewrap_result:
        # obj is result of call with args and kwargs, need those to apply to extra_cols
        if extra_cols is None:
            return wrap(obj)
        cols = []
        nop = lambda x: x
        for col in extra_cols:
            try:
                proto_col = col.__array_property__(attr, wrap=nop, unwrap=nop)
            except AttributeError:
                cols.append(col)
            else:
                if hasattr(proto_col, 'shape'):
                    cols.append(proto_col)
                else:  # callable we got from rewrap_result
                    cols.append(proto_col(*args, **kwargs))
        return wrap((obj, ) + tuple(cols))

    return do_wrap


class ArrayAttrMixin:

    def __getattr__(self, name):
        # Top-level, wrap is no-op
        #TODO __array_getattr_
        return self.__array_property__(name, wrap=lambda x: x, unwrap=lambda x: x)

    def _forward_array_getattr_to_content(self, name, wrap, unwrap):
        wrap_ = lambda x: wrap(self._rewrap_content(x))
        unwrap_ = lambda x: unwrap(self._unwrap_content(x))
        content = self._unwrap_content(self)
        if isinstance(content, tuple):
            content, *extra_cols = content
            wrap_ = make_wrap(wrap_, name, extra_cols)
        if hasattr(content, '__array_property__'):
            return content.__array_property__(name, wrap=wrap_, unwrap=unwrap_)
        raise AttributeError(f"{self.__class__} object has no attribute '{name}'")

    def __array_property__(self, name, wrap, unwrap):
        return self._forward_array_getattr_to_content(name, wrap, unwrap)
