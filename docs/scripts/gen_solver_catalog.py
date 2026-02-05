"""Auto-generate the solver documentation pages for MkDocs.

This script generates:
- solvers/index.md: Overview page with links to all categories
- solvers/{category}/index.md: Category pages with solver tables
- solvers/{category}/{slug}.md: Individual solver pages with mkdocstrings directives
"""

import importlib
import os
import pkgutil
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import mkdocs_gen_files

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("MNE_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "mne-config"))

import invert.solvers as _solvers_pkg
from invert.solvers.base import BaseSolver

# Category descriptions for overview pages
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Minimum Norm": (
        "Minimum norm methods estimate source activity by finding the solution with "
        "the smallest norm (typically L2) that explains the measured data. These methods "
        "are computationally efficient and provide smooth source estimates."
    ),
    "Bayesian": (
        "Bayesian methods use probabilistic frameworks to estimate source activity, "
        "incorporating prior knowledge about source distributions. They can provide "
        "uncertainty estimates and often yield sparser solutions than minimum norm methods."
    ),
    "Beamformers": (
        "Beamformers are spatial filters that estimate source activity at each location "
        "independently by constructing filters that pass signals from the target location "
        "while suppressing contributions from other sources."
    ),
    "Subspace Methods": (
        "Subspace methods (including MUSIC variants) exploit the orthogonality between "
        "signal and noise subspaces to localize sources. They are particularly effective "
        "for localizing a small number of focal sources."
    ),
    "Matching Pursuit": (
        "Matching pursuit and related greedy algorithms iteratively select sources that "
        "best explain the residual signal. They produce sparse solutions and can handle "
        "correlated sources effectively."
    ),
    "Dipole Fitting": (
        "Dipole fitting methods model brain activity as a small number of equivalent "
        "current dipoles and estimate their locations and orientations by minimizing "
        "the difference between measured and predicted signals."
    ),
    "Hybrid": (
        "Hybrid methods combine multiple inverse approaches to leverage their respective "
        "strengths, such as using beamformers for initial localization followed by "
        "sparse reconstruction."
    ),
    "Baseline": (
        "Baseline methods provide reference implementations for benchmarking purposes, "
        "such as random noise generators for null hypothesis testing."
    ),
    "Neural Networks": (
        "Neural network-based methods use deep learning to learn the inverse mapping "
        "from sensor data to source activity. They require training data but can capture "
        "complex nonlinear relationships. **Note:** These solvers require TensorFlow."
    ),
}

FOLDER_TO_CATEGORY: dict[str, str] = {
    "minimum_norm": "Minimum Norm",
    "bayesian": "Bayesian",
    "beamformers": "Beamformers",
    "music": "Subspace Methods",
    "matching_pursuit": "Matching Pursuit",
    "dipoles": "Dipole Fitting",
    "neural_networks": "Neural Networks",
    "hybrids": "Hybrid",
}

CATEGORY_ORDER: list[str] = [
    "Minimum Norm",
    "Bayesian",
    "Beamformers",
    "Subspace Methods",
    "Matching Pursuit",
    "Dipole Fitting",
    "Neural Networks",
    "Hybrid",
    "Baseline",
]


def slugify(value: str) -> str:
    """Convert a string to a URL-friendly slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "solver"


def get_solver_module_path(cls) -> str:
    """Get the full module path for a solver class."""
    return f"{cls.__module__}.{cls.__name__}"


def solver_id_html(solver_id: str) -> str:
    escaped = solver_id.replace('"', "&quot;")
    return (
        f'<span class="solver-id" data-solver-id="{escaped}" '
        f'title="Click to copy solver_id for Solver(&quot;{escaped}&quot;)">'
        f"<code>{solver_id}</code></span>"
    )


def infer_category(cls) -> str:
    parts = cls.__module__.split(".")
    try:
        solvers_idx = parts.index("solvers")
    except ValueError:
        solvers_idx = -1

    folder = (
        parts[solvers_idx + 1]
        if solvers_idx >= 0 and len(parts) > solvers_idx + 1
        else ""
    )
    if folder in FOLDER_TO_CATEGORY:
        return FOLDER_TO_CATEGORY[folder]
    if folder == "random_noise" or cls.__module__.endswith(".random_noise"):
        return "Baseline"
    if folder == "" and cls.__module__.endswith("random_noise"):
        return "Baseline"
    return "Baseline" if folder == "" else folder.replace("_", " ").title()


# Discover all solver modules
for _importer, modname, _ispkg in pkgutil.walk_packages(
    _solvers_pkg.__path__, prefix=_solvers_pkg.__name__ + "."
):
    if "._old" in modname or modname.endswith("._old"):
        continue
    try:
        importlib.import_module(modname)
    except Exception:
        continue

# Collect solvers with metadata, grouped by category
by_category: dict[str, list[tuple]] = defaultdict(list)
for cls in BaseSolver.__subclasses__():
    if "._old." in cls.__module__:
        continue
    meta = getattr(cls, "meta", None)
    if meta is None or meta.internal:
        continue
    by_category[infer_category(cls)].append((meta, cls))

# Sort categories and entries within each category
sorted_categories = [
    *CATEGORY_ORDER,
    *sorted(set(by_category.keys()) - set(CATEGORY_ORDER)),
]

# =============================================================================
# Generate solvers/index.md - Main overview page
# =============================================================================
overview_lines = [
    "# Solvers",
    "",
    f"invertmeeg provides **{sum(len(v) for v in by_category.values())}** inverse solvers organized into **{len(sorted_categories)}** categories.",
    "",
    "Each solver has a **full name** and a stable **solver id**. Instantiate solvers with `Solver(solver_id)`:",
    "",
    "```python",
    "from invert import Solver",
    "",
    'solver = Solver("MNE")',
    "```",
    "",
    "## Categories",
    "",
]

for category in sorted_categories:
    cat_slug = slugify(category)
    count = len(by_category.get(category, []))
    description = CATEGORY_DESCRIPTIONS.get(category, "")
    # Take first sentence of description for the overview
    short_desc = description.split(".")[0] + "." if description else ""
    overview_lines.append(f"### [{category}]({cat_slug}/index.md)")
    overview_lines.append("")
    overview_lines.append(f"*{count} solver{'s' if count != 1 else ''}*")
    overview_lines.append("")
    if short_desc:
        overview_lines.append(short_desc)
        overview_lines.append("")

with mkdocs_gen_files.open("solvers/index.md", "w") as f:
    f.write("\n".join(overview_lines))

# =============================================================================
# Generate category index pages and individual solver pages
# =============================================================================
for category in sorted_categories:
    cat_slug = slugify(category)
    entries = sorted(by_category.get(category, []), key=lambda x: x[0].acronym or "")

    # -------------------------------------------------------------------------
    # Category index page: solvers/{category}/index.md
    # -------------------------------------------------------------------------
    cat_lines = [
        f"# {category}",
        "",
    ]

    # Add category description
    if category in CATEGORY_DESCRIPTIONS:
        cat_lines.append(CATEGORY_DESCRIPTIONS[category])
        cat_lines.append("")

    cat_lines.extend(
        [
            f"This category contains **{len(entries)}** solver{'s' if len(entries) != 1 else ''}.",
            "",
            "## Solvers",
            "",
            "| Full Name | Solver ID | Description |",
            "|----------|-----------|-------------|",
        ]
    )

    for meta, _cls in entries:
        solver_slug = meta.slug or slugify(meta.acronym or meta.full_name)
        # Escape pipe characters in description
        desc = (meta.description or "").replace("|", "\\|")
        # Truncate long descriptions for table
        if len(desc) > 120:
            desc = desc[:117] + "..."
        cat_lines.append(
            f"| [{meta.full_name}]({solver_slug}.md) | {solver_id_html(meta.acronym)} | {desc} |"
        )

    cat_lines.append("")

    with mkdocs_gen_files.open(f"solvers/{cat_slug}/index.md", "w") as f:
        f.write("\n".join(cat_lines))

    # -------------------------------------------------------------------------
    # Individual solver pages: solvers/{category}/{slug}.md
    # -------------------------------------------------------------------------
    for meta, cls in entries:
        solver_slug = meta.slug or slugify(meta.acronym or meta.full_name)
        module_path = get_solver_module_path(cls)

        solver_lines = [
            f"# {meta.full_name}",
            "",
            f"**Solver ID:** {solver_id_html(meta.acronym)}",
            "",
            "## Usage",
            "",
            "```python",
            "from invert import Solver",
            "",
            "# fwd = ...    (mne.Forward object)",
            "# evoked = ... (mne.Evoked object)",
            "",
            f'solver = Solver("{meta.acronym}")',
            "solver.make_inverse_operator(fwd)",
            "stc = solver.apply_inverse_operator(evoked)",
            "stc.plot()",
            "```",
            "",
        ]

        # Add description
        if meta.description:
            solver_lines.append("## Overview")
            solver_lines.append("")
            solver_lines.append(meta.description)
            solver_lines.append("")

        # Add references
        if meta.references:
            solver_lines.append("## References")
            solver_lines.append("")
            for i, ref in enumerate(meta.references, 1):
                solver_lines.append(f"{i}. {ref}")
            solver_lines.append("")

        # Add API documentation via mkdocstrings
        solver_lines.extend(
            [
                "## API Reference",
                "",
                f"::: {module_path}",
                "    options:",
                "      show_root_heading: false",
                "      show_source: true",
                "      members:",
                "        - __init__",
                "        - make_inverse_operator",
                "        - apply_inverse_operator",
                "",
            ]
        )

        with mkdocs_gen_files.open(f"solvers/{cat_slug}/{solver_slug}.md", "w") as f:
            f.write("\n".join(solver_lines))
