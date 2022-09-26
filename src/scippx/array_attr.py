# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock


def rewrap_result(wrap):

    def decorator(callable):

        def func(*args, **kwargs):
            return wrap(callable(*args, **kwargs))

        return func

    return decorator


class ArrayAttrMixin:

    def __getattr__(self, name):
        # Top-level, wrap is no-op
        #TODO __array_getattr_
        return self.__array_property__(name, wrap=lambda x: x, unwrap=lambda x: x)

    def _forward_array_getattr_to_content(self, name, wrap, unwrap):
        content = self._unwrap_content(self)
        if hasattr(content, '__array_property__'):
            wrap_ = lambda x: wrap(self._rewrap_content(x))
            unwrap_ = lambda x: unwrap(self._unwrap_content(x))
            return content.__array_property__(name, wrap=wrap_, unwrap=unwrap_)
        raise AttributeError(f"{self.__class__} object has no attribute '{name}'")

    def __array_property__(self, name, wrap, unwrap):
        return self._forward_array_getattr_to_content(name, wrap, unwrap)