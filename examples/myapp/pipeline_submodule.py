# Copyright 2025 ICON Technology & Process Consulting SAS
"""Module to test schema generation from AST."""

from tunablex import tunable


@tunable("arg1", "arg2", namespace="model.preprocess.submodule_fun")
def submodule_fun(arg1: int = 0, arg2: bool = False):
    print("submodule_fun", arg1, arg2)


class SubmoduleClass:
    @tunable("attr1", "attr2", namespace="model.preprocess.submodule_class")
    def __init__(self, attr1: int = 0, attr2: bool = False):
        print("submodule_class", attr1, attr2)

    @tunable("arg1", "arg2", namespace="model.preprocess.submodule_class")
    @staticmethod
    def method(arg1: int = 0, arg2: bool = False):
        print("submodule_class_method", arg1, arg2)
