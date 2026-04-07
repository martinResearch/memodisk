# memodisk v2 — Migration, Bug Fixes & New Features

## Overview

This document covers four areas:

1. **Python 3.12 migration** — minimum version bump and the API changes it unlocks
2. **Bug fixes** — resolving the known failure modes without breaking the core auto-detection
3. **New features** — strict-by-default caching, cache transparency, and pipeline-friendly defaults
4. **Provenance & reproducibility** — lightweight experiment tracking that comes free from auto-detection

---

## 1. Python 3.12 Migration

### Why 3.12

Python 3.12 is the critical threshold. It provides two APIs that eliminate memodisk's
most fragile internals:

- **`sys.monitoring` (PEP 669)** — replaces `sys.setprofile` with per-code-object callbacks,
  tool IDs, and `DISABLE` returns for zero-overhead when not needed.
- **`dis.get_instructions()`** — stable, version-independent bytecode introspection that
  replaces raw opcode iteration.

Python 3.13 adds only `f_locals` write-through (PEP 667) — nice but not essential.
Python 3.14 adds nothing relevant and introduces more bytecode churn.
3.12 gives the best benefit-to-compatibility ratio.

### Migration steps

#### Step 1: Update `pyproject.toml`

```toml
# Before
requires-python = "==3.10.*"
classifiers = [
    "Programming Language :: Python :: 3.10",
]

# After
requires-python = ">=3.12"
classifiers = [
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
]
```

#### Step 2: Replace `sys.setprofile` with `sys.monitoring`

**Current code** (`Tracer.tracefunc` using `sys.setprofile`):
- Fires on every Python call/return across the entire interpreter
- No way to scope to specific code objects
- Raises `BaseException` to signal global-variable changes mid-execution — corrupts
  tracer state on failure
- Conflicts with debuggers and profilers that also use `sys.setprofile`

**New approach** using `sys.monitoring`:
```python
import sys

TOOL_ID = sys.monitoring.DEBUGGER_ID  # or a free tool ID (0-5)

def enable_tracing():
    sys.monitoring.use_tool_id(TOOL_ID, "memodisk")
    sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.CALL | sys.monitoring.events.PY_START)
    sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.CALL, on_call)
    sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.PY_START, on_py_start)

def on_call(code, instruction_offset, callable_obj, arg0):
    # Track the callable directly — no gc.get_referrers() needed
    ...
    return sys.monitoring.DISABLE  # disable for code objects we don't care about

def on_py_start(code, instruction_offset):
    # Record dependencies for this code object
    ...

def disable_tracing():
    sys.monitoring.set_events(TOOL_ID, 0)
    sys.monitoring.free_tool_id(TOOL_ID)
```

Key benefits:
- `on_call` receives the callable directly → eliminates `get_function_from_frame()`
  and its `gc.get_referrers()` non-determinism
- `DISABLE` return skips stdlib/third-party code at C level → major performance gain
- Doesn't conflict with debuggers (separate tool IDs)
- No need for `BaseException` hacks — global changes are checked after execution, not during

#### Step 3: Replace raw bytecode parsing with `dis.get_instructions()`

**Current code** (`get_globals_from_code`):
```python
# Broken on 3.11+ (EXTENDED_ARG never accumulated, LOAD_GLOBAL semantics changed)
global_ops = opmap["LOAD_GLOBAL"], opmap["STORE_GLOBAL"]
extended_args = opmap["EXTENDED_ARG"]
names = code.co_names
op = (int(c) for c in code.co_code)
# ... manual opcode iteration
```

**New code**:
```python
import dis

def get_globals_from_code(code: types.CodeType) -> list[str]:
    return sorted({
        instr.argval
        for instr in dis.get_instructions(code)
        if instr.opname in ("LOAD_GLOBAL", "STORE_GLOBAL")
    })
```

This is stable across Python 3.12–3.14+ because `dis.get_instructions()` abstracts
away opcode encoding changes (EXTENDED_ARG handling, operand width, etc.).

#### Step 4: Replace `HAVE_ARGUMENT` usage

Remove `from dis import HAVE_ARGUMENT, opmap`. The `HAVE_ARGUMENT` constant is
meaningless in 3.6+ (all instructions have 2 bytes) and the import is unnecessary
with `dis.get_instructions()`.

#### Step 5: Use modern Python syntax

With 3.12 as minimum:
- `list[str]` instead of `List[str]`, `dict[str, Any]` instead of `Dict[str, Any]`, etc.
- `X | Y` union syntax instead of `Union[X, Y]` and `Optional[X]`
- Remove `from __future__ import annotations` if present
- `typing_extensions` dependency may become optional (evaluate which features still need it)

---

## 2. Bug Fixes

These fix the failure modes identified in the test suite, ordered by impact.

### Bug 1: `get_globals_from_code()` broken on Python 3.11+

**Location**: `memodisk.py` L166–190

**Problem**: Raw `co_code` iteration doesn't handle:
- `EXTENDED_ARG` accumulation (the `extarg` variable is reset to 0 every iteration
  but never accumulated)
- 3.11 `LOAD_GLOBAL` encoding change (now uses `(namei << 1)` with a flag bit)
- 3.12 `PRECALL` removal and `LOAD_ATTR`/`LOAD_METHOD` merge

**Fix**: Replace with `dis.get_instructions()` as shown in Step 3 above.

**Tests**: `test_failure_modes.py` test #7 (aliased imports) should change from
`xfail` to `pass`.

### Bug 2: `BaseException` abuse in tracer corrupts state

**Location**: `memodisk.py` L428 (inside `Tracer.tracefunc`)

**Problem**: When a global variable changes mid-execution, the tracer raises
`BaseException`. This:
- Leaves `sys.setprofile` registered with the tracer still active
- Corrupts `tracer.globals` state for subsequent calls
- `BaseException` bypasses `except Exception` handlers in user code

**Fix**: With `sys.monitoring`, global-change detection moves out of the callback.
Instead, record globals at function entry, check at function exit. If changed,
mark the cache entry as invalid — don't raise during execution.

```python
def on_py_start(code, instruction_offset):
    frame = sys._getframe(1)
    snapshot = {name: get_global_hash(name, frame.f_globals[name], None, None)
                for name in get_globals_from_code(code)
                if name in frame.f_globals and should_track(frame.f_globals[name])}
    tracer.entry_snapshots[id(code)] = snapshot

# After function returns, compare snapshots — invalidate if different
```

### Bug 3: Non-picklable globals crash the tracer

**Location**: `memodisk.py` L226–252 (`get_global_hash`)

**Problem**: `pickle.dumps(variable)` is called inside the tracer callback. If the
global is non-picklable (e.g., a logger, a lock, a socket), the tracer raises
`BaseException` which propagates into user code as an unexpected crash.

**Fix**: Catch `pickle.PicklingError` (and `TypeError`) specifically. Use a fallback
hash based on `type(variable).__qualname__` + `id(variable)`, and flag the dependency
as "best-effort":

```python
try:
    pickled_var = pickle.dumps(variable)
    hash_str = hashlib.sha256(pickled_var).hexdigest()
    reliability = "reliable"
except (pickle.PicklingError, TypeError):
    hash_str = f"unpicklable:{type(variable).__qualname__}:{id(variable)}"
    reliability = "best-effort"
```

### Bug 4: `gc.get_referrers()` non-determinism in `get_function_from_frame()`

**Location**: `memodisk.py` L192–216

**Problem**: `gc.get_referrers(code)` returns all objects referencing a code object.
The order is non-deterministic and can return the wrong function if multiple functions
share the same code object (decorators, wrappers). Falls back to searching
`frame.f_back.f_globals` which may also fail.

**Fix**: With `sys.monitoring`, the `CALL` event provides the callable directly.
Build a `code_object → function` mapping at decoration time:

```python
# At @memoize decoration time:
_code_to_func[func.__code__] = func

# In the monitoring callback:
def on_call(code, instruction_offset, callable_obj, arg0):
    # callable_obj IS the function — no gc lookup needed
    ...
```

### Bug 5: `assert len(dependencies.code) > 0` fails for C-extension-only calls

**Location**: `memodisk.py` L801

**Problem**: If a memoized function only calls C extensions (e.g., `numpy` operations),
the tracer never fires for any Python code, resulting in zero code dependencies.
The assertion fails.

**Fix**: Always include the memoized function itself as a code dependency (its source
hash). The assertion becomes about "at least the decorated function itself":

```python
# Always include the decorated function's own code
own_dep = CodeDependency(
    module=func.__module__,
    qualname=func.__qualname__,
    hash=get_source_hash(func),
)
dependencies.code.insert(0, own_dep)
# assertion is now always satisfied
```

### Bug 6: `except BaseException` too broad in `get_global_hash`

**Location**: `memodisk.py` L243

**Problem**: `except BaseException as e` catches `KeyboardInterrupt`, `SystemExit`,
`GeneratorExit` — all of which should propagate. Only pickling errors should be caught.

**Fix**:
```python
except (pickle.PicklingError, TypeError, AttributeError) as e:
```

### Bug 7: Property decorator not tracked

**Problem**: Python properties use the descriptor protocol. `sys.setprofile` doesn't
fire `call` events for `__get__`/`__set__`, so property methods are invisible to the
tracer.

**Fix**: This is a fundamental Python limitation — properties bypass the call protocol.
`sys.monitoring` has the same limitation. Document as a known limitation and emit a
one-time warning when a property is detected on a `self` argument's class:

```python
# During tracing, if we see 'self' in frame.f_locals:
for name in dir(type(self)):
    if isinstance(getattr(type(self), name, None), property):
        warnings.warn(
            f"memodisk: {func.__name__} accesses property '{name}' which cannot be tracked. "
            f"Changes to the property getter will not invalidate the cache.",
            stacklevel=2,
        )
```

---

## 3. New Features

### Feature 1: Strict-by-default caching

**Principle**: never silently return potentially stale data.

```python
# Default behavior (v2):
on_unverifiable = "recompute"

# User can opt into aggressive caching:
@memoize(on_unverifiable="trust")    # old v1 behavior
@memoize(on_unverifiable="error")    # strictest — raise if uncertain
```

Implementation:

- On cache hit, re-verify all recorded dependencies before returning
- If any dependency cannot be verified (non-picklable, disappeared, etc.) → recompute
- If all dependencies verified and unchanged → return cached result
- Log a one-time warning per function for "best-effort" dependencies

The worst case is "same speed as no caching" — never "wrong answer."

### Feature 2: Code revalidation via `inspect.getsource()` + AST hash

Replace bytecode hashing for code-change detection with source-level hashing:

```python
import ast
import inspect
import hashlib

def get_source_hash(func: Callable) -> str:
    source = inspect.getsource(func)
    tree = ast.parse(source)
    # Normalize: strip docstrings, comments are already excluded by ast.parse
    normalized = ast.dump(tree)
    return hashlib.sha256(normalized.encode()).hexdigest()
```

Benefits:
- Immune to Python version bytecode changes
- Comments and whitespace changes don't invalidate cache
- Docstring-only edits don't invalidate cache
- Works identically across 3.12–3.14+

Cost: milliseconds per cache hit (reading source files already in OS cache).

### Feature 3: Dependency confidence tracking

Each cached result stores metadata about its dependencies:

```json
{
    "arguments_hash": "abc123...",
    "code": [...],
    "data": [...],
    "dependency_reliability": {
        "reliable": ["args", "src/model.py", "src/features.py", "config.yaml"],
        "best_effort": ["GLOBAL_THRESHOLD (non-picklable, hash by id)"],
        "untrackable": ["numpy.linalg.svd (C extension)"]
    }
}
```

On cache hit:
- All "reliable" deps verified → high confidence
- "best_effort" deps present → warn once, recompute in strict mode
- "untrackable" deps present → warn once, recompute in strict mode

### Feature 4: Cache audit CLI

```bash
$ python -m memodisk status [--cache-dir .memodisk]

train_model:
  cached: 2024-12-15 14:32 (5 min ago)
  confidence: HIGH
  deps: 3 reliable, 0 best-effort, 0 untrackable
  size: 142 MB
  status: FRESH (all deps verified unchanged)

extract_features:
  cached: 2024-12-15 10:00 (4h ago)
  confidence: MEDIUM
  deps: 5 reliable, 1 best-effort (GLOBAL_CONFIG)
  size: 89 MB
  status: STALE (src/features.py changed at L45)

$ python -m memodisk clear [--function train_model] [--older-than 1d]
```

### Feature 5: TTL (time-to-live)

```python
@memoize(ttl="4h")  # auto-expire after 4 hours
```

When a cache entry is older than `ttl`, recompute regardless of dependency state.
This guards against the scenario where upstream data changed through a path memodisk
can't detect (e.g., a database update, a file modified outside the tracked set).

Default: no TTL (backward compatible). Recommended for team use: `ttl="4h"` or
`ttl="1d"`.

### Feature 6: `memodisk.disabled()` context manager

```python
import memodisk

# In CI / production pipeline:
with memodisk.disabled():
    result = training_stage()  # all @memoize decorators become pass-through
```

Also configurable via environment variable:
```bash
MEMODISK_DISABLED=1 python run_pipeline.py
```

This makes the intent explicit: memodisk is a **development acceleration tool**, not a
production caching layer. In production, the pipeline runs everything from scratch.
Decorators can stay in committed code safely.

### Feature 7: `watch_files` for pipeline-aware invalidation

```python
@memoize(watch_files=["data/raw/*.csv", "configs/*.yaml"])
def load_and_process():
    ...
```

If any matching file's modification time changed since the cache was written, invalidate.
This bridges the gap between memodisk's function-level tracking and the pipeline's
stage-level inputs — the user declares "these are my stage's input files" and memodisk
respects that, even if the function code and arguments are identical.

Glob patterns are resolved at cache-check time. New files matching the pattern also
trigger invalidation.

### Feature 8: Dependency disappearance detection

On first execution, memodisk records all detected dependencies. On subsequent cache
checks, if a previously recorded dependency can no longer be detected (function no
longer calls it, global no longer exists), treat as a code change and recompute:

```
memodisk WARNING: train() previously depended on 'config.yaml' but no longer does.
  This suggests a code change removed a dependency. Recomputing.
```

---

## 4. Provenance & Reproducibility

memodisk already records a near-complete provenance record per cached result (argument
hashes, code hashes, global values, data file paths, random states). With small
additions, each cache entry becomes a lightweight experiment record — not just a
cache key, but a record of *exactly what produced this result*.

### Design principles

- **Default metadata stays lean** — hashes and short strings only. No embedded code,
  no serialized inputs, no escape-character headaches in JSON.
- **Rich provenance is opt-in** — for key experiments, not every call.
- **Sidecar files for large data** — diffs, freeze files, etc. live alongside the
  cache entry, not inside the JSON.

### 4.1 Git state tracking (default)

On every cache write, record the git state:

```json
{
    "git_commit": "a1b2c3d4e5f6...",
    "git_dirty": true
}
```

If the working tree is dirty, save the full diff as a **separate sidecar file**:

```
.memodisk/
  train_model/
    abc123/
      result.pkl
      dependencies.json      ← lean metadata (commit + dirty flag)
      dirty.patch             ← only exists if working tree was dirty
```

To reproduce: checkout `git_commit`, apply `dirty.patch`. That's the exact code state.
If the tree is clean, the commit alone is sufficient — no patch file written.

Implementation:
```python
import subprocess

def get_git_state() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
    return {"git_commit": commit, "git_dirty": dirty}

def save_dirty_patch(path: str) -> None:
    diff = subprocess.check_output(["git", "diff"], text=True)
    if diff:
        with open(path, "w") as f:
            f.write(diff)
```

Cost: ~10ms per cache write (one `git rev-parse`, one `git diff --quiet`). Negligible
compared to the function execution time that justified caching in the first place.

### 4.2 Package environment fingerprint (default)

Store a hash of the installed packages, plus the full freeze as a shared sidecar:

```json
{
    "packages_hash": "789xyz..."
}
```

The full freeze file is stored **once per unique hash**, not per cache entry:

```
.memodisk/
  _environments/
    789xyz.txt              ← output of `pip freeze` or `uv pip freeze`
  train_model/
    abc123/
      dependencies.json     ← references packages_hash
```

This avoids duplicating a ~2KB freeze file across hundreds of cache entries.

### 4.3 Opt-in rich provenance

For important experiments, store additional detail:

```python
@memoize(provenance="full")
def train_final_model(data):
    ...
```

`provenance="full"` additionally stores:
- **Argument values** (pickled, as a sidecar `inputs.pkl`) — not just their hash
- **Full pip freeze** inline in dependencies (redundant with shared file, but self-contained)
- **Function source code** (as a sidecar `source.py`) — snapshot of all tracked functions

```
.memodisk/
  train_final_model/
    def789/
      result.pkl
      dependencies.json
      dirty.patch
      inputs.pkl            ← only with provenance="full"
      source.py             ← only with provenance="full"
```

Default (`provenance="minimal"`): git state + packages hash. No sidecar files
except `dirty.patch` when needed.

### 4.4 How provenance complements pipeline tools

| Concern | Pipeline tool (DVC/Airflow) | memodisk |
|---|---|---|
| Stage ordering & DAG | ✓ | — |
| Stage-level reproducibility | ✓ (stage inputs/outputs) | — |
| **Function-level provenance** | — | ✓ (auto-detected deps) |
| **"What code produced this result?"** | — | ✓ (git commit + patch) |
| **"What packages were installed?"** | — | ✓ (freeze fingerprint) |
| **"What globals/config was active?"** | — | ✓ (auto-tracked) |

Pipeline tools track *between* stages. memodisk tracks *within* stages. Together
they give full-stack reproducibility from DAG level down to individual function
calls.

### 4.5 Reproducing a result

```bash
# Find out how a result was produced:
$ python -m memodisk inspect train_model --entry abc123

train_model (cached 2026-04-03 14:32):
  git: a1b2c3d (dirty — patch available)
  packages: numpy==1.26.4, torch==2.1.0, ... (12 total)
  args hash: def456...
  deps: 3 code, 1 data (config.yaml), 2 globals (THRESHOLD=0.5, BATCH_SIZE=32)
  confidence: HIGH
  result: 142 MB

# Reproduce:
$ python -m memodisk reproduce train_model --entry abc123
  → git checkout a1b2c3d
  → git apply .memodisk/train_model/abc123/dirty.patch
  → pip install -r .memodisk/_environments/789xyz.txt
  → Ready to re-run.
```

The `reproduce` command generates the steps — it doesn't execute them, to avoid
surprising side effects. The user reviews and runs them.

---

## Implementation Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| **P0** | `dis.get_instructions()` replacement | Small | Fixes 3.11+ compatibility |
| **P0** | `sys.monitoring` migration | Large | Fixes tracer corruption, perf, debugger conflicts |
| **P0** | Strict-by-default (`on_unverifiable="recompute"`) | Medium | Eliminates silent staleness |
| **P0** | Fix `BaseException` abuse | Small | Fixes tracer state corruption |
| **P1** | Non-picklable globals fallback hash | Small | Fixes crashes in common scenarios |
| **P1** | Source-based code hashing | Medium | Version-independent, more correct |
| **P1** | `memodisk.disabled()` | Small | Safe production coexistence |
| **P1** | Remove `assert len(dependencies.code) > 0` | Small | Fixes C-extension-only functions |
| **P2** | Dependency confidence tracking | Medium | Transparency |
| **P2** | Cache audit CLI | Medium | Developer experience |
| **P2** | TTL support | Small | Safety net |
| **P2** | `watch_files` | Medium | Pipeline integration |
| **P2** | Git state tracking (commit + dirty.patch) | Small | Reproducibility foundation |
| **P3** | Dependency disappearance detection | Medium | Edge case safety |
| **P3** | Property limitation warning | Small | Documentation |
| **P3** | Package environment fingerprint | Small | Reproducibility |
| **P3** | Opt-in rich provenance (`provenance="full"`) | Medium | Full reproducibility |

---

## Positioning

**Before (v1):** "memodisk magically caches everything — just add a decorator."

**After (v2):** "memodisk tracks what produced your results and caches them. It
auto-detects dependencies, recomputes when uncertain, records provenance, and
complements pipeline tools. Use it to make local iteration 10x faster — and to
know exactly how every result was produced."
