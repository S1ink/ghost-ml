# AGENTS.md

Guidance for AI coding agents (and humans) working in ghost-ml. Its tools
label erroneous lidar returns ("ghost points") in recorded bags, train a
classifier, and check the C++ that runs it on the robot. The
[README](README.md) is the user-facing guide; this file covers how to work
in the code.

This repository is the `src/` of a workspace. Run commands from the
workspace root (the parent of this directory), where the annotator's project
files live (`annotator.json`, `models/`, `cache/`, `out/`). The project
history is in `context.md` at the workspace root. Read its latest sections
before larger changes, and add a section for each one.

## Layout

| Path | What it is |
|---|---|
| `annotator_app/` | the web app, the trainer and the model export: plain Python scripts, not a ROS package |
| `ghost_tester/` | ROS 2 C++ package: `artifact_filter_node` (example) and `artifact_filter_check` (C++ vs Python) |
| `ghost_annotation/` | the earlier Foxglove and joystick annotator |
| `csm_utils/`, `launch_utils/` | git submodules |
| `multiscan_driver/`, `csm_metrics/` | git submodules: the driver owns the classifier and the default models |

## The classifier lives in multiscan_driver

- The C++ classifier is
  `multiscan_driver/include/multiscan_driver/artifact_filter/`, owned by the
  driver, which runs it on the robot. `ghost_tester` builds against it
  (`find_package(multiscan_driver)`). Don't add a copy here. The dependency
  runs one way, ghost-ml -> driver, and the robot needs nothing from here.
- The workspace builds only with the driver present. Either it is a
  submodule (`src/multiscan_driver`, with `src/csm_metrics`), or a workspace
  that has it is sourced as an underlay
  (`source <lance-ws>/install/setup.bash`).
- The C++ mirrors `annotator_app/scan_grid.py` (points to grid cells) and
  `annotator_app/artifact_features.py` (features). After changing either
  side, make a check file with `export_model.py --check-file` and run
  `artifact_filter_check`. The classifier inputs must be identical and the
  flagged cells the same.
- The model file format is versioned (`lidar_artifact_model 1`, see
  `export_model.py`). Changing it means changing the driver's `model.hpp`
  in step and bumping the version.
- Publishing makes the installed models the driver's defaults by copying
  them into the driver's `models/` (the `multiscan_driver` submodule, or the
  project's `driver` setting). Do it with Files -> "Publish to driver…" in
  the app, or with `annotator_app/publish_model.py`; both use the same
  functions in that script. Only realtime models (earlier scans only) with
  early exit checkpoints can be published: the driver filters live scans on
  one core. Commit the models in the driver, then the submodule pointer.

## Environment

- Python runs from `<workspace>/.venv` (made by `setup_venv.sh`) on the host
  Python with ROS 2 Lyrical: `source /opt/ros/lyrical/setup.bash; source
  .venv/bin/activate`. Don't add pip-only dependencies without updating
  `annotator_app/requirements.txt`.
- Scripts that need the sensor profile load the project settings
  (`config.load()` or `config.from_args()`) from the workspace root.
- The recorded bags live on a removable drive
  (`/run/media/<user>/ARCHIVE/rosbags`). It may not be mounted: say so
  rather than assuming a bag is missing.

## Code style

The reference is the LANCE package (`src/lance/src` of the robot
workspace): code should read like it. These rules apply to all C++ in
this repository and in multiscan_driver.

The same section is in multiscan_driver's `AGENTS.md` and the LANCE
workspace's: keep them in step.

### Formatting

- `ghost_tester/.clang-format` is the same file as the root one; keep them
  identical or remove the inner one.
- Format every new or changed C++ file with `clang-format -i` and the
  repository's `.clang-format`, a copy of the LANCE workspace's:
  Google-based, 4-space indent, 80 columns, Allman braces.
- Always use braces, even for one-line `if` and loop bodies.
- Every C++ file starts with the Cardinal Space Mining copyright banner
  (copy it from `ghost_tester/src/artifact_filter_check.cpp`). A block
  comment after it says what the file is for and anything a reader must
  know first.
- Use `#pragma once`. In a `.cpp`, include its own header first. Then group
  includes, separated by blank lines: standard library, third-party (ROS,
  zlib, ...), then this project's headers.

### Names

- **Spell names out.** Write `num_points`, `receive_time_ns`, `datagram`,
  `offset`, `probability` and `config`, not `n`, `recv_ns`, `dgram`, `off`,
  `prob` and `cfg`. Readability is worth the column space.
- Exceptions:
  - `i`, `j` and `k` for plain loop indices;
  - `x`, `y` and `z` for coordinates;
  - names from a specification, explained where they appear (e.g. `phi` and
    `theta` from the operating instructions);
  - `a` and `b` in a sort comparator.
- Put the unit at the end of the name: `_ns`, `_us`, `_ms`, `_s`, `_m`,
  `_mm`, `_deg`, `_rad`, `_hz`, `_bytes`. Use `num_` for counts and `has_`
  or `is_` for booleans.
- Casing:
  - types: `PascalCase`;
  - functions and methods: `camelCase` (`addSegment`, `cellAt`);
  - variables and members: `snake_case`;
  - constants: `UPPER_SNAKE_CASE`;
  - enums: `enum class` with `PascalCase` values.

### Functions

- Keep functions short and about one thing. Break long ones into named steps
  (e.g. `FrameAssembler::add`: decode, move to the host clock, count lost
  telegrams, place the segment, accept it). A function that reads as a list
  of steps is the goal.
- If a function stays long, label its parts with `// --- Step ---` comments.
- Structure control flow plainly: early returns instead of deep nesting, and
  no `goto`.

### Macros over lambdas

- For a small piece of code repeated inside a function or file, prefer a
  macro to a local lambda: declaring parameters, filling fields, repeated
  checks with an early `return` (`EXIT_IF_INSUFFICIENT_SIZE`), or anything
  that needs a name as a string (`#name`). The LANCE package does this
  (`INIT_PARAM`, `ADD_MARKER`, `SET_BOUNDS`).
- Name macros in `UPPER_SNAKE_CASE`. Define them just above their first
  use, and `#undef` them when they are no longer needed. Wrap a multi-line
  body in braces, and put macro arguments in parentheses inside the body.
- Use a real function when the code is longer, needs its own types or
  locals, or is called from more than one file.
- Lambdas are still right where a callable is required: callbacks, thread
  bodies and sort comparators. Name a stored lambda like a function
  (`readList`, `isFloat32`).

### Classes and headers

- Group a class with repeated access labels, in this order:
  - `public:` types;
  - `public:` constructors;
  - `public:` methods (more than one block if they group naturally);
  - `protected:` types, then `protected:` methods, then `protected:` members.
- Use `protected` rather than `private` (the LANCE convention).
- Use a `struct` with public members for plain data. Give it methods only if
  they are small helpers.
- Initialise members in the declaration with braces: `uint32_t
  num_points{0};`.
- Always access members through `this->`.
- Group related members with `// --- Group ---` comments, and say which
  thread owns or which mutex guards shared state.
- Put two blank lines between top-level classes or structs, and one between
  closely related small structs.
- Put `/* ... */` doc comments above declarations, saying what something
  means, its units and its edge cases. Use `//` for short inline notes.

### Source files

- Mark the sections of a `.cpp` with `// --- Section ---` banners padded to
  80 columns, with three blank lines before each. Keep helpers in an
  anonymous namespace at the top.

### Comments

- Explain why, not what: the sensor behaviour, the manual section, or the
  reason for an odd choice. Keep sentences short and plain.

### Refactoring numeric code

- Renaming and splitting must not change results. Keep the exact order of
  floating-point operations: `(a * b) / c` and `a * (b / c)` round
  differently. That alone has broken the golden-cloud test.
- To check a refactor, compare outputs byte for byte against the previous
  version on the captures, not just with the tests.

### Files that keep their own style

`ghost_tester/src/main.cpp` and `ghost_tester/include/util/` are the
user's own code from cardinal-perception (`d_ary_heap.hpp` is an identical
copy). Leave them alone unless asked. The `csm_utils` and `launch_utils`
submodules are separate repositories.

### Python

- Match the existing scripts:
  - a module docstring with what the script does and how to call it;
  - `argparse`;
  - `from __future__ import annotations`;
  - project settings through `config.py`.
- The naming rule applies here too: spell names out.
- Feature code (`artifact_features.py`, `scan_grid.py`) has a C++ twin.
  Change both in step, keeping the same operation order and rounding, and
  run the check.

## Commits

`src/` is the git repository. Commit subjects are short, lower case and
imperative. The user commits; don't commit unless asked.
