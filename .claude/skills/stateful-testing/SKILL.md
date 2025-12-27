---
name: stateful-testing
description: Create Hypothesis stateful tests comparing Dask operations against reference implementations. Use for writing RuleBasedStateMachine tests that verify Dask arrays, dataframes, or bags match NumPy, pandas, or other reference behavior through sequences of operations.
---

# Hypothesis Stateful Testing for Dask

Write stateful tests that verify Dask operations match reference implementations through arbitrary operation sequences.

## What are Stateful Tests?

Stateful tests use Hypothesis's `RuleBasedStateMachine` to:
1. Initialize state (e.g., create arrays)
2. Apply random sequences of operations (rules)
3. Verify invariants hold after each operation

**Example**: Test that a Dask array equals its NumPy equivalent after any sequence of `rechunk()`, `transpose()`, and arithmetic operations.

## Core Pattern

```python
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
from hypothesis import note, settings
import hypothesis.strategies as st

@settings(max_examples=10, deadline=None, stateful_step_count=10)
class MyStateMachine(RuleBasedStateMachine):
    """Test that dask_obj matches reference_obj through all operations."""

    def __init__(self):
        super().__init__()
        self.reference: ReferenceType
        self.dask_obj: DaskType

    @property
    def shape(self):
        """Derive properties from reference object."""
        return self.reference.shape

    @initialize(...)
    def init_state(self, data):
        """Create initial reference and Dask objects."""
        self.reference = create_reference(data)
        self.dask_obj = create_dask_equivalent(self.reference)
        note(f"Initialize: ...")

    @rule(...)
    def operation(self, params):
        """Apply an operation to both objects."""
        note(f"Operation: ...")
        self.reference = ref_operation(self.reference, params)
        self.dask_obj = dask_operation(self.dask_obj, params)

    @invariant()
    def objects_are_equal(self):
        """Verify Dask matches reference after every operation."""
        assert_eq(self.dask_obj, self.reference)

# Create pytest test
TestMyStateMachine = MyStateMachine.TestCase
```

## Strategy Library

**Location**: `dask/tests/strategies.py`

Use these strategies for generating test data:

### Array Strategies

```python
from dask.tests.strategies import (
    chunks,                 # Generate valid chunk specs for a shape
    broadcastable_shape,    # Generate shape that broadcasts with target
    broadcastable_array,    # Generate broadcastable NumPy array
    chunked_arrays,         # Generate (numpy_arr, dask_arr) pairs
    axis_strategy,          # Generate valid axis for reductions
    slice_strategy,         # Generate valid slice for an axis
    index_strategy,         # Generate valid index tuple for a shape
    numeric_dtypes,         # Common numeric dtypes
    all_dtypes,             # All supported dtypes
    reduction_ops,          # Common reduction operation names
)
```

### Using `chunks` strategy

```python
@rule(chunks_spec=st.data())
def rechunk(self, chunks_spec):
    new_chunks = chunks_spec.draw(chunks(shape=self.shape))
    self.dask_array = self.dask_array.rechunk(new_chunks)
```

### Using `broadcastable_array`

```python
@rule(other=st.data(), op=st.sampled_from([operator.add, operator.mul]))
def binary_op(self, other, op):
    other_arr = other.draw(broadcastable_array(
        shape=self.shape,
        dtype=self.numpy_array.dtype
    ))
    other_dask = da.from_array(other_arr, chunks=-1)
    self.numpy_array = op(self.numpy_array, other_arr)
    self.dask_array = op(self.dask_array, other_dask)
```

### Using `chunked_arrays`

```python
from hypothesis import given
from dask.tests.strategies import chunked_arrays

@given(chunked_arrays())
def test_operation(arrays):
    np_arr, da_arr = arrays
    # Test that operation preserves equality
    assert_eq(operation(da_arr), operation(np_arr))
```

## Building Tests Incrementally

Start simple and add complexity:

### Step 1: Single operation

```python
@settings(max_examples=5, deadline=None, stateful_step_count=3)
class DaskArrayStateMachine(RuleBasedStateMachine):
    @initialize(arrays=npst.arrays(...))
    def init_arrays(self, arrays):
        self.numpy_array = arrays
        self.dask_array = da.from_array(arrays, chunks=-1)

    @rule(chunks_spec=st.data())
    def rechunk(self, chunks_spec):
        new_chunks = chunks_spec.draw(chunks(shape=self.shape))
        self.dask_array = self.dask_array.rechunk(new_chunks)

    @invariant()
    def arrays_are_equal(self):
        assert_eq(self.dask_array, self.numpy_array)
```

### Step 2: Add another operation

```python
    @rule(axes=st.data())
    def transpose(self, axes):
        ndim = len(self.shape)
        axes_perm = axes.draw(st.permutations(range(ndim)))
        self.numpy_array = np.transpose(self.numpy_array, axes_perm)
        self.dask_array = da.transpose(self.dask_array, axes_perm)
```

### Step 3: Add operations that change values

```python
    @rule(other=st.data(), op=st.sampled_from([operator.add, operator.mul]))
    def binary_op(self, other, op):
        other_arr = other.draw(broadcastable_array(
            shape=self.shape,
            dtype=self.numpy_array.dtype
        ))
        other_dask = da.from_array(other_arr, chunks=-1)
        self.numpy_array = op(self.numpy_array, other_arr)
        self.dask_array = op(self.dask_array, other_dask)
```

### Step 4: Add reductions, slicing, etc.

```python
    @rule(idx=st.data())
    def getitem(self, idx):
        index = idx.draw(index_strategy(shape=self.shape))
        self.numpy_array = self.numpy_array[index]
        self.dask_array = self.dask_array[index]

    @rule(axis=st.data())
    def sum_axis(self, axis):
        ax = axis.draw(axis_strategy(ndim=len(self.shape)))
        self.numpy_array = self.numpy_array.sum(axis=ax)
        self.dask_array = self.dask_array.sum(axis=ax)
```

## Best Practices

### Use `note()` for Debugging

Add notes to track operation sequences when tests fail:

```python
@rule(chunks_spec=st.data())
def rechunk(self, chunks_spec):
    new_chunks = chunks_spec.draw(chunks(shape=self.shape))
    note(f"Rechunk: {self.dask_array.chunks} -> {new_chunks}")
    self.dask_array = self.dask_array.rechunk(new_chunks)
```

When a test fails, Hypothesis shows the sequence:
```
Falsifying example:
state = DaskArrayStateMachine()
Initialize: shape=(3, 4), dtype=float64
state.rechunk(chunks_spec=...)
Rechunk: ((3,), (4,)) -> ((1, 2), (2, 2))
state.transpose(axes=...)
Transpose: axes=(1, 0), shape (3, 4) -> (4, 3)
```

### Use Properties Instead of Storing Shape

```python
@property
def shape(self) -> tuple[int, ...]:
    """Derive shape from reference object."""
    return self.numpy_array.shape
```

This keeps shape in sync as operations change it.

### Use `operator` Module for Binary Ops

```python
import operator

@rule(op=st.sampled_from([operator.add, operator.mul, operator.sub, operator.truediv]))
def binary_op(self, other, op):
    # Use op(a, b) instead of if/elif chains
    self.numpy_array = op(self.numpy_array, other_array)
    self.dask_array = op(self.dask_array, other_dask)
```

### Constrain Values to Avoid Overflow

```python
@initialize(
    arrays=npst.arrays(
        dtype=npst.floating_dtypes(),
        shape=npst.array_shapes(...),
        elements={
            "allow_nan": False,
            "allow_infinity": False,
            "min_value": -100,
            "max_value": 100,
        },
    )
)
```

For integer dtypes, don't specify min/max values (let Hypothesis choose appropriately).

### Handle Division by Zero Naturally

Division by zero produces `inf` or `nan`, which both NumPy and Dask handle consistently:

```python
# No special handling needed - just apply the operation
self.numpy_array = operator.truediv(self.numpy_array, other)
self.dask_array = operator.truediv(self.dask_array, other_dask)
```

### Tune Performance

```python
@settings(
    max_examples=10,           # Number of test cases to generate
    deadline=None,              # Disable individual test deadline
    stateful_step_count=10,    # Operations per test case
)
```

Start with small values while developing, increase for thorough testing.

## Common Patterns

### Testing Array Operations

See `dask/tests/test_stateful_array.py` for a complete example testing:
- `rechunk()` - shape-preserving, chunk-changing operation
- `transpose()` - shape-changing operation
- Binary ops (`+`, `*`, `-`, `/`) - value-changing operations with broadcasting

### Testing DataFrame Operations

```python
class DataFrameStateMachine(RuleBasedStateMachine):
    @initialize(...)
    def init_frames(self, data):
        self.pandas_df = pd.DataFrame(data)
        self.dask_df = dd.from_pandas(self.pandas_df, npartitions=2)

    @rule(...)
    def operation(self, ...):
        self.pandas_df = self.pandas_df.operation(...)
        self.dask_df = self.dask_df.operation(...)

    @invariant()
    def frames_are_equal(self):
        from dask.dataframe.utils import assert_eq
        assert_eq(self.dask_df, self.pandas_df)
```

### Testing Reductions

```python
@rule(axis=st.data(), op=st.sampled_from(["sum", "mean", "std"]))
def reduction(self, axis, op):
    ax = axis.draw(axis_strategy(ndim=len(self.shape)))
    note(f"Reduction: {op} along axis={ax}")
    self.numpy_array = getattr(self.numpy_array, op)(axis=ax)
    self.dask_array = getattr(self.dask_array, op)(axis=ax)
```

## Debugging Failed Tests

When Hypothesis finds a failure:

1. **Look at the operation sequence** in the failure output
2. **Reproduce manually** using the shown steps
3. **Add more `note()` calls** to see intermediate state
4. **Run with `--hypothesis-seed=<N>`** to reproduce exact failure
5. **Use `--hypothesis-verbosity=verbose`** for more detail

```bash
pytest dask/tests/test_stateful_array.py -v --hypothesis-seed=12345
```

## When to Use Stateful Tests

Use stateful tests when:
- Testing that operations preserve an invariant (e.g., Dask == NumPy)
- Operations can be composed in arbitrary order
- Want to find unexpected interaction bugs
- Testing caching, optimization, or other stateful behavior

**Don't use** for:
- Simple single-operation tests (use `@given` property tests)
- Operations that can't be composed
- When reference implementation isn't available

## Examples

See these test files:
- `dask/tests/test_stateful_array.py` - Array operations vs NumPy
- `dask/tests/strategies.py` - Strategy library

## Adding New Strategies

When you need a new strategy:

1. Add to `dask/tests/strategies.py`
2. Use `@st.composite` decorator
3. Accept `draw: st.DrawFn` as first parameter
4. Use `draw()` to sample from sub-strategies
5. Add docstring with Parameters/Returns
6. Export if it's generally useful

```python
@st.composite
def my_strategy(draw: st.DrawFn, *, param: int) -> MyType:
    """Generate a MyType with the given param.

    Parameters
    ----------
    param : int
        Description

    Returns
    -------
    MyType
        Description
    """
    # Use draw() to sample from strategies
    value = draw(st.integers(min_value=0, max_value=param))
    return MyType(value)
```

## Settings Reference

Common Hypothesis settings for stateful tests:

| Setting | Default | Recommended | Purpose |
|---------|---------|-------------|---------|
| `max_examples` | 100 | 5-20 | Number of test cases |
| `deadline` | 200ms | `None` | Per-test timeout |
| `stateful_step_count` | 50 | 5-20 | Operations per case |
| `print_blob` | False | - | Show reproduction blob |

Example:
```python
@settings(
    max_examples=10,
    deadline=None,
    stateful_step_count=10,
)
class MyStateMachine(RuleBasedStateMachine):
    ...
```
