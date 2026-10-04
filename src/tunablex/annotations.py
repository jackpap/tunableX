"""Read annotations without evaluating unrelated names, including Python 3.14."""

import inspect
import sys


def raw_annotations(obj) -> dict:
    """Return only this object's annotations; resolve selected fields separately."""
    if sys.version_info >= (3, 14):
        from annotationlib import Format, get_annotations

        return get_annotations(obj, format=Format.STRING)
    return inspect.get_annotations(obj, eval_str=False)


def signature(obj) -> inspect.Signature:
    """Inspect a signature without evaluating deferred Python 3.14 annotations."""
    if sys.version_info >= (3, 14):
        from annotationlib import Format

        return inspect.signature(obj, annotation_format=Format.STRING)
    return inspect.signature(obj)
