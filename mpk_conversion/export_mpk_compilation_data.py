#!/usr/bin/env python3
"""Validate or regenerate the browser-ready matter P(k) compilation JSON.

Regeneration requires Docker and the upstream marius311/mpk_compilation image.
The script instruments the upstream notebook's final plotting cell and records
the exact arrays passed to the notebook's labeled errorbar(...) calls.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


UPSTREAM_REPO = "https://github.com/marius311/mpk_compilation"
UPSTREAM_COMMIT = "4bfa131ac7a1063c7fd4c8dd8859f16dfb48e293"
TARGET_DATASETS = {
    "Planck 2018 TT",
    "Planck 2018 EE",
    "Planck 2018 \u03c6\u03c6",
    "DES Y1 cosmic shear",
    "SDSS DR7 LRG",
    "eBOSS DR14 Ly-\u03b1 forest",
}


INSTRUMENTATION = r'''
# === CosmoSliderWeb export instrumentation ===
mpk_target_colors = Dict(
    "Planck 2018 TT" => "#1f77b4",
    "Planck 2018 EE" => "#ff7f0e",
    "Planck 2018 ϕϕ" => "#2ca02c",
    "DES Y1 cosmic shear" => "#bcbd22",
    "SDSS DR7 LRG" => "#d62728",
    "eBOSS DR14 Ly-α forest" => "#9467bd",
)
mpk_target_ids = Dict(
    "Planck 2018 TT" => "planck_tt",
    "Planck 2018 EE" => "planck_ee",
    "Planck 2018 ϕϕ" => "planck_phiphi",
    "DES Y1 cosmic shear" => "des_y1",
    "SDSS DR7 LRG" => "sdss_dr7_lrg",
    "eBOSS DR14 Ly-α forest" => "eboss_dr14_lya",
)
mpk_seen_labels = Set{String}()
mpk_export = Dict{String,Any}(
    "source" => Dict{String,Any}(
        "repository" => "https://github.com/marius311/mpk_compilation",
        "commit" => "4bfa131ac7a1063c7fd4c8dd8859f16dfb48e293",
        "notebook" => "notebooks/mpk_compilation.ipynb",
        "figure" => "mpk_compilation.png",
        "citations" => [
            "Chabanier et al. 2019, arXiv:1905.08103",
            "Planck 2018: I, arXiv:1807.06205",
        ],
    ),
    "note" => "Fixed browser-ready matter power spectrum compilation reference from the upstream notebook figure. Values are shown at every website redshift as a visual observational/reference overlay.",
    "units" => Dict("k" => "h/Mpc", "pk" => "(Mpc/h)^3"),
    "datasets" => Any[],
)

function mpk_float_vector(values)
    return [Float64(v) for v in vec(values)]
end

function mpk_add_dataset(label, x, y; yerr=nothing, xerr=nothing, kwargs...)
    label_string = string(label)
    if !haskey(mpk_target_colors, label_string) || label_string in mpk_seen_labels
        return
    end

    push!(mpk_seen_labels, label_string)
    xs = mpk_float_vector(x)
    ys = mpk_float_vector(y)
    yerrs = yerr === nothing ? zeros(length(xs)) : mpk_float_vector(yerr)

    k_low = copy(xs)
    k_high = copy(xs)
    if xerr !== nothing
        xe = Array(xerr)
        if ndims(xe) == 2
            for i in eachindex(xs)
                k_low[i] = xs[i] - Float64(xe[1, i])
                k_high[i] = xs[i] + Float64(xe[2, i])
            end
        else
            xerrs = mpk_float_vector(xerr)
            for i in eachindex(xs)
                k_low[i] = xs[i] - xerrs[i]
                k_high[i] = xs[i] + xerrs[i]
            end
        end
    end

    points = Any[]
    for i in eachindex(xs)
        push!(points, Dict(
            "k" => xs[i],
            "pk" => ys[i],
            "kLow" => max(k_low[i], 1e-12),
            "kHigh" => max(k_high[i], 1e-12),
            "pkLow" => max(ys[i] - yerrs[i], 1e-12),
            "pkHigh" => max(ys[i] + yerrs[i], 1e-12),
        ))
    end

    push!(mpk_export["datasets"], Dict(
        "id" => mpk_target_ids[label_string],
        "label" => label_string,
        "color" => mpk_target_colors[label_string],
        "points" => points,
    ))
end

mpk_original_errorbar = errorbar
function errorbar(x, y, args...; label=nothing, yerr=nothing, xerr=nothing, kwargs...)
    if label !== nothing
        mpk_add_dataset(label, x, y; yerr=yerr, xerr=xerr)
    end
    return mpk_original_errorbar(x, y, args...; label=label, yerr=yerr, xerr=xerr, kwargs...)
end
# === end CosmoSliderWeb export instrumentation ===
'''


EXPORT_FOOTER = r'''

# === CosmoSliderWeb JSON export ===
mpk_pyjson = pyimport("json")
open("mpk_compilation_data.generated.json", "w") do f
    write(f, mpk_pyjson.dumps(mpk_export, indent=2))
end
# === end CosmoSliderWeb JSON export ===
'''


def validate_data(path: Path) -> None:
    payload = json.loads(path.read_text())
    datasets = payload.get("datasets", [])
    if len(datasets) != 6:
        raise ValueError(f"expected 6 datasets, found {len(datasets)}")

    labels = {dataset.get("label") for dataset in datasets}
    missing = TARGET_DATASETS - labels
    if missing:
        raise ValueError(f"missing datasets: {sorted(missing)}")

    for dataset in datasets:
        points = dataset.get("points", [])
        if not points:
            raise ValueError(f"{dataset.get('label')} has no points")

        for index, point in enumerate(points):
            for key in ["k", "pk", "kLow", "kHigh", "pkLow", "pkHigh"]:
                value = point.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{dataset.get('label')} point {index} has invalid {key}: {value}")
            if point["kLow"] > point["k"] or point["k"] > point["kHigh"]:
                raise ValueError(f"{dataset.get('label')} point {index} has invalid k bounds")
            if point["pkLow"] > point["pk"] or point["pk"] > point["pkHigh"]:
                raise ValueError(f"{dataset.get('label')} point {index} has invalid P(k) bounds")
            if point["kLow"] <= 0 or point["pkLow"] <= 0:
                raise ValueError(f"{dataset.get('label')} point {index} has non-positive log value")


def copy_or_clone_upstream(upstream_dir: Path | None, work_dir: Path) -> Path:
    checkout = work_dir / "mpk_compilation"
    if upstream_dir:
        shutil.copytree(upstream_dir, checkout, symlinks=True)
    else:
        subprocess.run(["git", "clone", UPSTREAM_REPO, str(checkout)], check=True)
        subprocess.run(["git", "checkout", UPSTREAM_COMMIT], cwd=checkout, check=True)
    return checkout


def instrument_notebook(checkout: Path) -> Path:
    notebook_path = checkout / "notebooks" / "mpk_compilation.ipynb"
    notebook = json.loads(notebook_path.read_text())
    final_cell = None
    for cell in reversed(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if "# Main panel" in source and "savefig(\"mpk_compilation.png\"" in source:
            final_cell = cell
            break
    if final_cell is None:
        raise RuntimeError("could not locate final plotting cell")

    final_cell["source"] = (
        INSTRUMENTATION.splitlines(keepends=True)
        + final_cell["source"]
        + EXPORT_FOOTER.splitlines(keepends=True)
    )
    export_path = checkout / "notebooks" / "mpk_compilation_export.ipynb"
    export_path.write_text(json.dumps(notebook))
    return export_path


def run_docker_export(checkout: Path, image: str) -> Path:
    export_notebook = instrument_notebook(checkout)
    command = (
        "cd /home/cosmo/work/notebooks && "
        "if [ ! -e dat/lrgdr7like ]; then mkdir -p dat && ln -s /home/cosmo/dat/lrgdr7like dat/lrgdr7like; fi && "
        "jupyter nbconvert --to notebook --execute "
        "--ExecutePreprocessor.timeout=2400 mpk_compilation_export.ipynb"
    )
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{checkout}:/home/cosmo/work", image, "bash", "-lc", command],
        check=True,
    )
    generated = export_notebook.parent / "mpk_compilation_data.generated.json"
    if not generated.exists():
        raise RuntimeError("notebook finished but did not produce mpk_compilation_data.generated.json")
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="validate an existing JSON file")
    parser.add_argument("--input", type=Path, default=Path("mpk_compilation_data.json"))
    parser.add_argument("--output", type=Path, default=Path("mpk_compilation_data.json"))
    parser.add_argument("--upstream-dir", type=Path, help="existing mpk_compilation checkout")
    parser.add_argument("--image", default="marius311/mpk_compilation")
    args = parser.parse_args()

    if args.validate:
        validate_data(args.input)
        print(f"validated {args.input}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        checkout = copy_or_clone_upstream(args.upstream_dir, Path(tmp))
        generated = run_docker_export(checkout, args.image)
        validate_data(generated)
        shutil.copyfile(generated, args.output)
        print(f"wrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
