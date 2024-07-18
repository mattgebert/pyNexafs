import abc
from typing import Callable, Any, TypeAlias, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
WrappedFunctionDecorator: TypeAlias = Callable[[Callable[P, T]], Callable[P, T]]


# Decorator for copying docstrings
def doc_copy(copy_func: Callable[P]) -> WrappedFunctionDecorator[P, T]:
    """
    Copies the doc string of the given function to the decorated function.

    Parameters
    ----------
    copy_func : Callable
        Function whose docstring is to be copied.
    """

    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        # f is the decorated function
        f.__doc__ = copy_func.__doc__
        return f

    return decorator


class staticproperty(property):
    """
    Decorator for a static property.

    Allows a static property call, where class 'cls' or object 'self' arguments
    are not required to access the property. Also checks for abstract signature.
\
    Notes
    -----
    Incompatible with overrides.overrides decorator.

    Examples
    --------
    >>> class A(metaclass=abc.ABCMeta):
    >>>     @staticproperty
    >>>     @abc.abstractmethod
    >>>     def prop():
    >>>         return None
    >>> class B(A):
    >>>     @staticproperty
    >>>     def prop():
    >>>         return 1
    >>> print(A.prop)
    None
    >>> A()
    TypeError: Can't instantiate abstract class A without an implementation for abstract method 'prop'
    >>> print(B.prop, B().prop)
    (1,1)
    """

    def __get__(self, instance, owner):
        if hasattr(self.fget, "__isabstractmethod__") and getattr(
            self.fget, "__isabstractmethod__", True
        ):
            raise TypeError(
                f"Can't call abstract method `{self.fget.__name__}` of {owner.__name__}."
            )
        return self.fget()

    def __set__(self, instance, value):
        raise AttributeError("Can't set attribute")

    def __delete__(self, instance):
        raise AttributeError("Can't delete attribute")

    def getter(self, fget):
        return super().getter(fget)


# class classproperty(property):
#     """
#     Decorator for a class property.

#     Used since the deprecated `@property` `@classmethod` chaining.
#     See discussion here: https://stackoverflow.com/q/76249636/1717003
#     However this doesn't work for class setters https://stackoverflow.com/q/78259605/1717003.
#     Hence commented.

#     """
#     def __init__(self, func):
#         self.fget = func
#         self.owner = None
#     def __get__(self, instance, owner):
#         self.owner = owner
#         return self.fget(owner)

#     def __set__(self, instance, value):
#         self.fset(self.owner, value)


if __name__ == "__main__":
    # Some basic tests

    class A:
        @staticproperty
        def prop():
            return 1

    print(A.prop, A().prop)

    class B:
        @staticproperty
        @abc.abstractmethod
        def prop():
            return 2

    try:
        print(B.prop, B().prop)
    except TypeError as e:
        print("Failed as expected:", e)

    class C(B):

        @prop.getter
        def prop():
            return 3

    print(C.prop, C().prop)
