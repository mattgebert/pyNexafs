import abc
import ast
import inspect
import textwrap
from enum import EnumMeta
from typing import (
    Annotated,
    Callable,
    Any,
    TypeAlias,
    ParamSpec,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

P = ParamSpec("P")
T = TypeVar("T")
EnumType = TypeVar("EnumType", bound=EnumMeta)
WrappedFunctionDecorator: TypeAlias = Callable[[Callable[P, T]], Callable[P, T]]


def enum_member_doc(cls: EnumType) -> EnumType:
    """
    Class decorator that assigns docstrings to enum members.

    Ensures each documented member has an ``Annotated[T, "docstring"]``
    type annotation.

    Docstrings are sourced from (in priority order):

    1. **``Annotated`` metadata** — ``Annotated[T, "docstring", ...]`` on the
       member annotation; the first ``str`` metadata argument is used.
    2. **Trailing string literals** — a string literal immediately following a
       plain member assignment.

    After processing, ``cls.__annotations__`` is updated so every documented
    member carries ``Annotated[type(member), "docstring"]``, meaning
    ``typing.get_type_hints(cls, include_extras=True)`` will always reflect the
    docstring regardless of which source was used.

    Parameters
    ----------
    cls : EnumMeta
        The enum class to process.

    Returns
    -------
    EnumMeta
        The same enum class with ``__doc__`` and ``__annotations__`` updated on
        each qualifying member.

    Examples
    --------
    Trailing string literal:

    >>> from enum import StrEnum
    >>> @enum_member_doc
    ... class Color(StrEnum):
    ...     RED = "red"
    ...     \"\"\"The colour red.\"\"\"
    >>> Color.RED.__doc__
    'The colour red.'
    >>> import typing; typing.get_type_hints(Color, include_extras=True)['RED']
    typing.Annotated[str, 'The colour red.']

    ``Annotated`` metadata:

    >>> from typing import Annotated
    >>> @enum_member_doc
    ... class Color(StrEnum):
    ...     RED: Annotated[str, "The colour red."] = "red"
    >>> Color.RED.__doc__
    'The colour red.'
    """
    # Collect docs: start with trailing string literals via AST
    member_docs: dict[str, str] = {}

    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except OSError:
        source = None

    if source is not None:
        tree = ast.parse(source)
        class_def = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        body = class_def.body
        for i, node in enumerate(body):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if i + 1 >= len(body):
                continue
            next_node = body[i + 1]
            if (
                isinstance(next_node, ast.Expr)
                and isinstance(next_node.value, ast.Constant)
                and isinstance(next_node.value.value, str)
            ):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in cls.__members__:
                        member_docs[target.id] = next_node.value.value

    # Annotated[T, "docstring"] metadata overrides trailing literals
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = {}
    for name, hint in hints.items():
        if name not in cls.__members__:
            continue
        if get_origin(hint) is not Annotated:
            continue
        for meta in get_args(hint)[1:]:
            if isinstance(meta, str):
                member_docs[name] = meta
                break

    # Apply docs and ensure Annotated annotations exist for all documented members
    annotations = cls.__annotations__ if hasattr(cls, "__annotations__") else {}
    for name, doc in member_docs.items():
        cls.__members__[name].__doc__ = doc
        existing = annotations.get(name)
        # Wrap in Annotated if not already, or replace the doc string in existing Annotated
        if existing is not None and get_origin(existing) is Annotated:
            args = get_args(existing)
            base_type = args[0]
            other_meta = [m for m in args[1:] if not isinstance(m, str)]
            annotations[name] = (
                Annotated[base_type, doc, *other_meta]
                if other_meta
                else Annotated[base_type, doc]
            )
        else:
            base_type = type(cls.__members__[name])
            annotations[name] = Annotated[base_type, doc]
    cls.__annotations__ = annotations

    return cls


# Decorator for copying docstrings
def doc_copy(copy_func: Callable[P, T]) -> WrappedFunctionDecorator[P, T]:
    """
    Copy the doc string of the given function to the decorated function.

    Parameters
    ----------
    copy_func : Callable
        Function whose docstring is to be copied.

    Returns
    -------
    WrappedFunctionDecorator
        Decorator that copies the docstring from `copy_func` to the decorated function.
    """

    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        # f is the decorated function
        f.__doc__ = copy_func.__doc__
        return f

    return decorator


if __name__ == "__main__":
    ### Some basic tests

    # 1. Test copying docstrings
    def copy_func():
        """
        Copy this docstring.
        """
        pass

    @doc_copy(copy_func)
    def f():
        pass

    print("#1 |", f.__doc__)


class y_property(property):
    """
    Decorator for the y property, allowing indexing by the y label or index of the data.
    """

    def getter_item(self, fgetitem: Callable[P, T]) -> Callable[P, T]:
        self.fgetitem = fgetitem
        return self

    def __getitem__(self, y: Any) -> Any:
        print("GETITEM")
        return self.fgetitem(y)

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)


if __name__ == "__main__":
    ### Some basic tests
    # 1. Test y property
    class A:
        @y_property
        def myprop(self):
            print("get")
            return 1

        @myprop.setter
        def myprop(self, val):
            print("set", val)
            return

        @myprop.getter_item
        def myprop(self, key: int | str):
            print("get_item", key)

    # Run test
    obj = A()
    print("#1 |", obj.myprop)
    print("#2 |")
    obj.myprop = 2
    print("#3 |", A.myprop[obj, 2])


class staticproperty(property):
    """
    Decorator for a static property.

    Allows a static property call, where class 'cls' or object 'self' arguments
    are not required to access the property. Also checks for abstract signature.
\
    Notes
    -----
    Incompatible with override decorator.

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


if __name__ == "__main__":
    ### Some basic tests

    # 1. Test static property can be called with/without instance.
    class A:
        @staticproperty
        def prop():
            return 1

    print("#1 |", A.prop, A().prop)

    # 2. Test abstract static property.
    class B:
        @staticproperty
        @abc.abstractmethod
        def prop():
            return 2

    try:
        print(B.prop, B().prop)
    except TypeError as e:
        print("#2 | Failed as expected:", e)

    # 3. Test overriding abstract static property.
    class C(B):
        try:

            @B.prop.getter
            def prop():
                return 3

        except TypeError as e:
            print("#3.1 | Failed as expected:", e)

        try:

            @prop.getter
            def prop():
                return 3

        except NameError as e:
            print("#3.2 | Failed as expected:", e)

        @staticproperty
        def prop():
            return 3

    print("#3.3 |", C.prop, C().prop)

    # 4. Test setter can be used.
    class A:
        @staticproperty
        def prop():
            return 1

        @prop.setter
        def prop(value):
            print("#4 Set")

    print("#4.1 |", A.prop, A().prop)
    obj = A()
    obj.prop = 2
    print("#4.2 |", obj.prop)
    A.prop = 2
    print("#4.3 |", A.prop)

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
