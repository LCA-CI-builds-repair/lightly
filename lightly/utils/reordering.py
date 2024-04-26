from typing import List, Sequence, TypeVar
isort lightly/utils/reordering.py
        >>> keys = [3, 2, 1]
        >>> items = ['!', 'world', 'hello']
        >>> sorted_keys = [1, 2, 3]
        >>> sorted_items = sort_items_by_keys(
        >>>     keys,
        >>>     items,
        >>>     sorted_keys,
        >>> )
        >>> print(sorted_items)
        >>> > ['hello', 'world', '!']

    """
    if len(keys) != len(items) or len(keys) != len(sorted_keys):
        raise ValueError(
            f"All inputs (keys,  items and sorted_keys) "
            f"must have the same length, "
            f"but their lengths are: ({len(keys)},"
            f"{len(items)} and {len(sorted_keys)})."
        )
    lookup = {key_: item_ for key_, item_ in zip(keys, items)}
    sorted_ = [lookup[key_] for key_ in sorted_keys]
    return sorted_
