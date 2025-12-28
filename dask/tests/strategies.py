"""Hypothesis strategies for testing Dask collections."""

from __future__ import annotations

from typing import Any

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np

import dask.array as da

# Type alias for chunks
Chunks = tuple[tuple[int, ...], ...]


@st.composite  # type: ignore[misc]
def chunks(draw: st.DrawFn, *, shape: tuple[int, ...]) -> Chunks:
    """Generate valid chunk specifications for a given shape.

    Adapted from flox/tests/strategies.py

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the array to generate chunks for

    Returns
    -------
    Chunks
        A tuple of tuples representing chunk sizes along each dimension
    """
    chunks = []
    for size in shape:
        if size > 1:
            nchunks = draw(st.integers(min_value=1, max_value=size - 1))
            dividers = sorted(
                set(
                    draw(st.integers(min_value=1, max_value=size - 1))
                    for _ in range(nchunks - 1)
                )
            )
            chunks.append(
                tuple(a - b for a, b in zip(dividers + [size], [0] + dividers))
            )
        else:
            chunks.append((1,))
    return tuple(chunks)


@st.composite  # type: ignore[misc]
def broadcastable_shape(draw: st.DrawFn, *, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Generate a shape that is broadcastable with the given shape.

    Broadcasting rules:
    - Dimensions are compared from right to left
    - Two dimensions are compatible if they are equal or one of them is 1
    - Can have fewer dimensions (treated as having leading 1s)

    Parameters
    ----------
    shape : tuple[int, ...]
        The target shape to broadcast with

    Returns
    -------
    tuple[int, ...]
        A shape that is broadcastable with the input shape
    """
    ndim = len(shape)

    # Decide how many dimensions the broadcastable shape will have
    # Can be anywhere from 1 to ndim + 2 (allow some extra leading dimensions)
    new_ndim = draw(st.integers(min_value=1, max_value=max(1, ndim + 2)))

    # Generate the shape from right to left (matching broadcasting rules)
    new_shape = []
    for i in range(new_ndim):
        # Index into original shape from the right
        idx = ndim - new_ndim + i
        if idx < 0:
            # This is a leading dimension (doesn't align with original shape)
            # Can be any small size, but often 1
            new_shape.append(draw(st.sampled_from([1, 1, 1, 2, 3])))  # bias toward 1
        else:
            # This dimension aligns with shape[idx]
            # Can be either the same size or 1
            new_shape.append(draw(st.sampled_from([1, shape[idx]])))

    return tuple(new_shape)


@st.composite  # type: ignore[misc]
def broadcastable_array(
    draw: st.DrawFn, *, shape: tuple[int, ...], dtype: np.dtype
) -> np.ndarray:
    """Generate a NumPy array with a shape broadcastable to the given shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        The target shape to broadcast with
    dtype : np.dtype
        The dtype of the array to generate

    Returns
    -------
    np.ndarray
        An array with broadcastable shape
    """
    new_shape = draw(broadcastable_shape(shape=shape))
    array = draw(
        npst.arrays(
            dtype=dtype,
            shape=new_shape,
            elements={
                "allow_nan": False,
                "allow_infinity": False,
                "min_value": -100,
                "max_value": 100,
            },
        )
    )
    return array


@st.composite  # type: ignore[misc]
def broadcast_to_shape(draw: st.DrawFn, *, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Generate a shape that the given shape can be broadcast TO.

    Broadcasting rules: shape A can be broadcast to shape B if:
    - B has at least as many dimensions as A
    - For each dimension (aligned right-to-left):
      - A's dimension is 1, OR
      - A's dimension equals B's dimension

    Parameters
    ----------
    shape : tuple[int, ...]
        The source shape that will be broadcast

    Returns
    -------
    tuple[int, ...]
        A target shape that source can be broadcast to
    """
    # Skip if shape contains 0 (can't meaningfully broadcast empty arrays)
    if 0 in shape:
        return shape

    # Target can have more dimensions (add 0-3 leading dimensions)
    extra_dims = draw(st.integers(min_value=0, max_value=3))

    new_shape = []

    # Leading dimensions (not aligned with source)
    for _ in range(extra_dims):
        new_shape.append(draw(st.integers(min_value=1, max_value=10)))

    # Aligned dimensions (from source)
    for size in shape:
        if size == 1:
            # Can expand dimension of size 1 to any size
            new_shape.append(draw(st.integers(min_value=1, max_value=10)))
        else:
            # Must keep the same size
            new_shape.append(size)

    return tuple(new_shape)


# Common dtypes for testing
numeric_dtypes = (
    npst.integer_dtypes(endianness="=")
    | npst.unsigned_integer_dtypes(endianness="=")
    | npst.floating_dtypes(endianness="=", sizes=(32, 64))
)

all_dtypes = (
    numeric_dtypes
    | npst.boolean_dtypes()
    | npst.datetime64_dtypes(endianness="=")
    | npst.timedelta64_dtypes(endianness="=")
)


@st.composite  # type: ignore[misc]
def axis_strategy(draw: st.DrawFn, *, ndim: int, allow_none: bool = True) -> int | None:
    """Generate a valid axis for an array with the given number of dimensions.

    Parameters
    ----------
    ndim : int
        The number of dimensions
    allow_none : bool, optional
        Whether to allow None (meaning all axes), by default True

    Returns
    -------
    int | None
        A valid axis index or None
    """
    if allow_none:
        return draw(st.none() | st.integers(min_value=0, max_value=ndim - 1))
    else:
        return draw(st.integers(min_value=0, max_value=ndim - 1))


@st.composite  # type: ignore[misc]
def slice_strategy(draw: st.DrawFn, *, size: int) -> slice:
    """Generate a valid slice for an axis of the given size.

    Parameters
    ----------
    size : int
        The size of the dimension to slice

    Returns
    -------
    slice
        A valid slice object
    """
    if size == 0:
        return slice(None)

    # Generate start, stop, step
    start = draw(st.none() | st.integers(min_value=-size, max_value=size - 1))
    stop = draw(st.none() | st.integers(min_value=-size, max_value=size))
    step = draw(st.none() | st.integers(min_value=1, max_value=max(1, size)))

    return slice(start, stop, step)


@st.composite  # type: ignore[misc]
def index_strategy(
    draw: st.DrawFn, *, shape: tuple[int, ...]
) -> tuple[int | slice, ...]:
    """Generate a valid index for an array with the given shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the array to index

    Returns
    -------
    tuple[int | slice, ...]
        A valid index (tuple of ints and slices)
    """
    index = []
    for size in shape:
        # Choose between integer index or slice
        if draw(st.booleans()) and size > 0:
            # Integer index
            index.append(draw(st.integers(min_value=0, max_value=size - 1)))
        else:
            # Slice
            index.append(draw(slice_strategy(size=size)))
    return tuple(index)


@st.composite  # type: ignore[misc]
def chunked_arrays(
    draw: st.DrawFn,
    *,
    dtype: st.SearchStrategy[np.dtype] | None = None,
    shape: st.SearchStrategy[tuple[int, ...]] | None = None,
    elements: dict[str, Any] | None = None,
) -> tuple[np.ndarray, da.Array]:
    """Generate a NumPy array and equivalent chunked Dask array.

    Parameters
    ----------
    dtype : st.SearchStrategy[np.dtype] | None, optional
        Strategy for generating dtypes. If None, uses floating point dtypes.
    shape : st.SearchStrategy[tuple[int, ...]] | None, optional
        Strategy for generating shapes. If None, uses 1-4 dims with sides 1-10.
    elements : dict[str, Any] | None, optional
        Element constraints for array generation. If None, uses sensible defaults.

    Returns
    -------
    tuple[np.ndarray, da.Array]
        A NumPy array and equivalent Dask array with random chunking
    """
    if dtype is None:
        dtype = npst.floating_dtypes(sizes=(32, 64))
    if shape is None:
        shape = npst.array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=10)

    array_shape = draw(shape)
    array_dtype = draw(dtype)

    # Use appropriate element constraints based on dtype
    if elements is None:
        if array_dtype.kind == "f":
            elements = {
                "allow_nan": False,
                "allow_infinity": False,
                "min_value": -100,
                "max_value": 100,
            }
        else:
            elements = {}

    numpy_array = draw(
        npst.arrays(dtype=array_dtype, shape=array_shape, elements=elements)
    )
    chunk_spec = draw(chunks(shape=array_shape))
    dask_array = da.from_array(numpy_array, chunks=chunk_spec)

    return numpy_array, dask_array


@st.composite  # type: ignore[misc]
def axes_strategy(draw: st.DrawFn, *, ndim: int) -> None | int | tuple[int, ...]:
    """Generate valid axes for reductions on an array with the given ndim.

    Can generate:
    - None (reduce over all axes)
    - A single integer axis
    - A tuple of one or more axes (without duplicates)

    Parameters
    ----------
    ndim : int
        The number of dimensions

    Returns
    -------
    None | int | tuple[int, ...]
        A valid axis specification for a reduction
    """
    if ndim == 0:
        # 0-d arrays can only reduce with axis=None
        return None

    choice = draw(st.sampled_from(["single", "tuple", "none"]))

    if choice == "single":
        return draw(st.integers(min_value=0, max_value=ndim - 1))
    elif choice == "tuple":
        # Generate a non-empty subset of axes
        num_axes = draw(st.integers(min_value=1, max_value=ndim))
        axes_list = draw(st.permutations(range(ndim)))[:num_axes]
        return tuple(sorted(axes_list))
    else:  # none
        return None


@st.composite  # type: ignore[misc]
def reductions(draw: st.DrawFn) -> str:
    """Generate reduction operation names, optionally with nan-skipping versions.

    Returns operation names like 'sum', 'nansum', 'mean', 'nanmean', etc.

    Returns
    -------
    str
        A reduction operation name
    """
    # Base operations
    # TODO: "var", "std" skipped for now - they need a better algorithm
    base_ops = ["sum", "mean", "min", "max", "prod", "any", "all"]
    base_op = draw(st.sampled_from(base_ops))

    # Operations that have nan-skipping versions
    nan_ops = {"sum", "mean", "std", "var", "min", "max", "prod"}

    # Randomly choose nan-skipping version if available
    use_nan = draw(st.booleans())
    return f"nan{base_op}" if (use_nan and base_op in nan_ops) else base_op


# Scan operations (cumulative)
scans = st.sampled_from(["cumsum", "cumprod"])
