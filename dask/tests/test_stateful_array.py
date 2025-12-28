"""Stateful Hypothesis tests comparing Dask array operations to NumPy."""

from __future__ import annotations

import operator
import warnings

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")

import dask.array as da
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from dask.array.utils import assert_eq
from dask.tests.strategies import (
    axes_strategy,
    broadcast_to_shape,
    broadcastable_array,
    chunks,
    reductions,
    scans,
)
from hypothesis import assume, note, settings, seed
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

# Set numpy print options for concise output in notes
np.set_printoptions(precision=3, threshold=10, edgeitems=2, linewidth=60)


# TODO: remove the precondition on the `broadcast` rule to reproduce hang
# @seed(307603228577777922830132553021337866664)
# TODO: below seed reproduces an indexing error I think
# @seed(206283088023844505153171715290298593218)
@settings(max_examples=50, deadline=None, stateful_step_count=30)
class DaskArrayStateMachine(RuleBasedStateMachine):
    """Stateful test comparing Dask array operations to NumPy arrays.

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
        assert da._array_expr_enabled()

    # Skip rechunking if any dimension has size 0
    @precondition(lambda self: 0 not in self.shape)
    @rule(data=st.data())
    def rechunk(self, data):
        """Rechunk the Dask array (NumPy array remains unchanged)."""
        # Generate valid chunks for the current shape
        new_chunks = data.draw(chunks(shape=self.shape))
        note(f"Rechunk: {self.dask_array.chunks} -> {new_chunks}")
        self.dask_array = self.dask_array.rechunk(new_chunks)

    # @rule()
    # def persist(self):
    #     """Persist the Dask array (no-op for NumPy array)."""
    #     note(f"Persist: shape {self.shape}")
    #     # Suppress warnings during computation
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", RuntimeWarning)
    #         self.dask_array = self.dask_array.persist()
    # NumPy array is already in memory, so no-op

    @rule(data=st.data())
    def transpose(self, data):
        """Transpose both NumPy and Dask arrays."""
        ndim = len(self.shape)
        # Generate a valid permutation of axes
        axes_perm = data.draw(st.permutations(range(ndim)))
        note(
            f"Transpose: axes={axes_perm}, shape {self.shape} -> {tuple(self.shape[i] for i in axes_perm)}"
        )
        self.numpy_array = np.transpose(self.numpy_array, axes_perm)
        self.dask_array = da.transpose(self.dask_array, axes_perm)

    @precondition(lambda self: self.numpy_array.dtype.kind != "b")
    @rule(
        data=st.data(),
        op=st.sampled_from(
            [operator.add, operator.mul, operator.sub, operator.truediv]
        ),
    )
    def binary_op(self, data, op):
        """Apply a binary operation with a broadcastable array or scalar."""
        # Randomly choose between scalar and array
        use_scalar = data.draw(st.booleans())
        if use_scalar:
            # Generate a scalar value
            other_value = data.draw(
                st.floats(
                    min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
                )
            )
            note(f"Binary op: {op.__name__} with scalar {other_value:.3f}")
        else:
            # Generate a broadcastable array
            other_value = data.draw(
                broadcastable_array(shape=self.shape, dtype=self.numpy_array.dtype)
            )
            note(
                f"Binary op: {op.__name__} with shape {other_value.shape} (broadcast to {self.shape})"
            )

        # Convert to dask if array
        other_dask = (
            other_value if use_scalar else da.from_array(other_value, chunks=-1)
        )

        # Apply the operation (division by zero results in inf/nan, which is handled naturally)
        # Suppress numpy warnings for operations like divide by zero, overflow, etc.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.numpy_array = op(self.numpy_array, other_value)
            self.dask_array = op(self.dask_array, other_dask)

    @rule(data=st.data())
    def basic_indexing(self, data):
        """Apply basic indexing to both arrays."""
        # Generate a valid basic index for the current shape
        idx = data.draw(npst.basic_indices(shape=self.shape))
        note(
            f"Basic indexing: {idx}, shape {self.shape} -> {self.numpy_array[idx].shape}"
        )
        self.numpy_array = self.numpy_array[idx]
        self.dask_array = self.dask_array[idx]

    @precondition(lambda self: False)
    @rule(data=st.data())
    def broadcast(self, data):
        """Broadcast both arrays to a compatible shape."""
        # Generate a shape that the current array can be broadcast to
        target_shape = data.draw(broadcast_to_shape(shape=self.shape))
        note(f"Broadcast: shape {self.shape} -> {target_shape}")
        self.numpy_array = np.broadcast_to(self.numpy_array, target_shape)
        self.dask_array = da.broadcast_to(self.dask_array, target_shape)

    @rule(data=st.data(), op_name=reductions())
    def reduction(self, data, op_name):
        """Apply a reduction operation along specified axes."""
        # Generate valid axes for the current shape
        axes = data.draw(axes_strategy(ndim=len(self.shape)))

        # Skip if any of the axes being reduced over have size 0
        if axes is None:
            axes_tuple = tuple(range(len(self.shape)))
        else:
            axes_tuple = (axes,) if isinstance(axes, int) else axes
        assume(all(self.shape[ax] != 0 for ax in axes_tuple))

        # Draw reduction operation (already includes nan-version logic)
        note(f"Reduction: {op_name}(axis={axes}), shape {self.shape}")

        # Apply the reduction operation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.numpy_array = getattr(np, op_name)(self.numpy_array, axis=axes)
            self.dask_array = getattr(da, op_name)(self.dask_array, axis=axes)

        note(f"  -> shape {self.shape}")

    @precondition(lambda self: len(self.shape) != 0)
    @rule(
        data=st.data(),
        op=scans,
        use_nan_version=st.booleans(),
    )
    def scan(self, data, op, use_nan_version):
        """Apply a cumulative scan operation along a single axis."""
        # Scans require a single axis (not None or tuple)
        axis = data.draw(st.integers(min_value=0, max_value=len(self.shape) - 1))

        # Both cumsum and cumprod have nan-skipping versions
        op_name = f"nan{op}" if use_nan_version else op
        note(f"Scan: {op_name}(axis={axis}), shape {self.shape}")

        # Apply the scan operation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.numpy_array = getattr(np, op_name)(self.numpy_array, axis=axis)
            self.dask_array = getattr(da, op_name)(self.dask_array, axis=axis)

        note(f"  -> shape {self.shape}")

    @invariant()
    def arrays_are_equal(self):
        """Verify that the Dask array matches the NumPy array."""
        # Suppress warnings during computation (e.g., division by zero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Use relaxed tolerance for float32 to handle precision differences
            # rtol = 1e-5 if self.numpy_array.dtype == np.float32 else 1e-7
            assert_eq(self.dask_array, self.numpy_array)


# Create the test
TestDaskArray = DaskArrayStateMachine.TestCase
