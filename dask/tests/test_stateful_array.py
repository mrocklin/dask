"""Stateful Hypothesis tests comparing Dask array operations to NumPy."""

from __future__ import annotations

import operator
import os
import warnings

import numpy as np
import pytest

# Enable array expression mode before importing dask.array
os.environ["DASK_ARRAY__QUERY_PLANNING"] = "True"

hypothesis = pytest.importorskip("hypothesis")

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import note, settings
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

import dask.array as da
from dask.array.utils import assert_eq
from dask.tests.strategies import broadcast_to_shape, broadcastable_array, chunks


@settings(max_examples=20, deadline=None, stateful_step_count=20)
class DaskArrayStateMachine(RuleBasedStateMachine):
    """Stateful test comparing Dask array operations to NumPy arrays.

    This test runs with array expression mode enabled (DASK_ARRAY__QUERY_PLANNING=True).

    Invariant: The Dask array should always produce the same result as the NumPy array.
    """

    def __init__(self):
        super().__init__()
        self.numpy_array: np.ndarray
        self.dask_array: da.Array

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the NumPy array."""
        return self.numpy_array.shape

    @initialize(
        arrays=npst.arrays(
            dtype=npst.floating_dtypes(sizes=(32, 64), endianness="="),
            shape=npst.array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=10),
            elements={
                "allow_nan": False,
                "allow_infinity": False,
                "min_value": -100,
                "max_value": 100,
            },
        )
    )
    def init_arrays(self, arrays):
        """Initialize with a NumPy array and create an equivalent Dask array."""
        self.numpy_array = arrays
        # Start with a simple chunking strategy
        self.dask_array = da.from_array(self.numpy_array, chunks=-1)
        note(f"Initialize: shape={self.shape}, dtype={self.numpy_array.dtype}")
        note(f"Array expr enabled: {da._array_expr_enabled()}")

    @rule(
        chunks_spec=st.data(),
    )
    def rechunk(self, chunks_spec):
        """Rechunk the Dask array (NumPy array remains unchanged)."""
        # Skip rechunking if any dimension has size 0
        if 0 in self.shape:
            return
        # Generate valid chunks for the current shape
        new_chunks = chunks_spec.draw(chunks(shape=self.shape))
        note(f"Rechunk: {self.dask_array.chunks} -> {new_chunks}")
        self.dask_array = self.dask_array.rechunk(new_chunks)

    @rule(
        axes=st.data(),
    )
    def transpose(self, axes):
        """Transpose both NumPy and Dask arrays."""
        ndim = len(self.shape)
        # Generate a valid permutation of axes
        axes_perm = axes.draw(st.permutations(range(ndim)))
        note(
            f"Transpose: axes={axes_perm}, shape {self.shape} -> {tuple(self.shape[i] for i in axes_perm)}"
        )
        self.numpy_array = np.transpose(self.numpy_array, axes_perm)
        self.dask_array = da.transpose(self.dask_array, axes_perm)

    @rule(
        other=st.data(),
        op=st.sampled_from(
            [operator.add, operator.mul, operator.sub, operator.truediv]
        ),
    )
    def binary_op(self, other, op):
        """Apply a binary operation with a broadcastable array."""
        # Generate a broadcastable array
        other_array = other.draw(
            broadcastable_array(shape=self.shape, dtype=self.numpy_array.dtype)
        )
        other_dask = da.from_array(other_array, chunks=-1)

        op_name = op.__name__
        note(
            f"Binary op: {op_name} with shape {other_array.shape} (broadcast to {self.shape})"
        )

        # Apply the operation (division by zero results in inf/nan, which is handled naturally)
        # Suppress numpy warnings for operations like divide by zero, overflow, etc.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.numpy_array = op(self.numpy_array, other_array)
            self.dask_array = op(self.dask_array, other_dask)

    @rule(
        index=st.data(),
    )
    def basic_indexing(self, index):
        """Apply basic indexing to both arrays."""
        # Generate a valid basic index for the current shape
        idx = index.draw(npst.basic_indices(shape=self.shape))
        note(
            f"Basic indexing: {idx}, shape {self.shape} -> {self.numpy_array[idx].shape}"
        )
        self.numpy_array = self.numpy_array[idx]
        self.dask_array = self.dask_array[idx]

    @rule(
        new_shape=st.data(),
    )
    def broadcast(self, new_shape):
        """Broadcast both arrays to a compatible shape."""
        # Generate a shape that the current array can be broadcast to
        target_shape = new_shape.draw(broadcast_to_shape(shape=self.shape))
        note(f"Broadcast: shape {self.shape} -> {target_shape}")
        self.numpy_array = np.broadcast_to(self.numpy_array, target_shape)
        self.dask_array = da.broadcast_to(self.dask_array, target_shape)

    @invariant()
    def arrays_are_equal(self):
        """Verify that the Dask array matches the NumPy array."""
        # Suppress warnings during computation (e.g., division by zero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert_eq(self.dask_array, self.numpy_array)


# Create the test
TestDaskArray = DaskArrayStateMachine.TestCase
