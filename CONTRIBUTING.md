# Contributing to invertmeeg

## Development setup

```bash
# Clone the repository
git clone https://github.com/LukeTheHecker/invertmeeg.git
cd invertmeeg

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install all dependencies
uv venv --python 3.11
uv pip install -e ".[dev]"

# Run the tests
uv run pytest tests/ -v
```

## Adding a new solver

Adding a solver requires changes in **two** files:

### 1. Create the solver class

Create your solver in the appropriate module under `invert/solvers/` (or add to
an existing file if it belongs to the same family). Your class must inherit from
`BaseSolver`:

```python
from .base import BaseSolver, InverseOperator

class SolverMyMethod(BaseSolver):
    name = "MyMethod"

    def make_inverse_operator(self, forward, *args, alpha="auto", **kwargs):
        super().make_inverse_operator(forward, *args, alpha=alpha, **kwargs)
        leadfield = self.leadfield

        inverse_operators = []
        for alpha in self.alphas:
            # ... compute your inverse operator matrix here ...
            inverse_operators.append(inverse_operator)

        self.inverse_operators = [
            InverseOperator(op, self.name) for op in inverse_operators
        ]
        return self
```

Key points:
- Call `super().make_inverse_operator(...)` first -- it sets up `self.leadfield`
  and `self.alphas`.
- Store results as a list of `InverseOperator` objects in
  `self.inverse_operators`.
- If your solver needs the data during `make_inverse_operator` (like
  beamformers), accept `mne_obj` as the second positional argument and call
  `self.unpack_data_obj(mne_obj)` to get the data matrix.
- If your solver computes the solution on-the-fly rather than precomputing an
  operator matrix, override `apply_inverse_operator` instead.

### 2. Register the solver

Open `invert/invert.py` and add your solver to the `_build_registry()` function:

```python
_add(["mymethod", "my-method"], solvers.SolverMyMethod)
```

If your solver needs extra constructor arguments for certain aliases, use
`functools.partial`:

```python
_add("mymethod-variant", partial(solvers.SolverMyMethod, some_param="value"))
```

Then add the user-facing name to `all_solvers` in `invert/config.py`:

```python
all_solvers = [
    ...,
    "MyMethod",
]
```

### 3. Make sure the import works

If your solver lives in a new file, add the import to
`invert/solvers/__init__.py`:

```python
from .my_module import *
```

### 4. Test it

The parametrised smoke tests in `tests/test_solvers.py` will automatically pick
up your solver from `config.all_solvers`. Run:

```bash
uv run pytest tests/test_solvers.py -k "MyMethod" -v
```

If your solver is data-dependent (beamformers, MUSIC, etc.), add its config name
to the `DATA_DEPENDENT_SOLVERS` set in `tests/test_solvers.py`.

## Code style

- Use `logging` instead of `print()` for any output.
- Use specific exception types (`except np.linalg.LinAlgError:`) rather than
  bare `except:`.
- Add type hints to public method signatures.
- Keep solver implementations self-contained -- avoid adding new top-level
  dependencies unless strictly necessary.

## Running tests

```bash
# All tests
uv run pytest tests/ -v

# Math correctness tests only (fast, no MNE forward model needed)
uv run pytest tests/test_math.py -v

# Solver smoke tests (slower, builds a forward model)
uv run pytest tests/test_solvers.py -v
```
