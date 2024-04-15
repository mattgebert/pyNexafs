class staticproperty(property):
    """
    Decorator for a static property.

    Allows a static property call, where class 'cls' or object 'self' arguments
    are not required to access the property. Also checks for abstract signature.


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
