from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build scheme0..schemeN hardware summaries by moving FasterRCNN layers "
            "from DCNM to ACIM following affinity rank."
        )
    )
    parser.add_argument("--affinity-ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-name", default="ranked_scheme_hardware_summary.json")
    parser.add_argument("--plot-name", default="ranked_scheme_energy_latency_scatter.png")
    parser.add_argument("--energy-plot-name", default="ranked_scheme_energy_by_scheme.png")
    parser.add_argument("--latency-plot-name", default="ranked_scheme_latency_by_scheme.png")
    parser.add_argument("--edp-plot-name", default="ranked_scheme_edp_by_scheme.png")
    parser.add_argument(
        "--title",
        default="FasterRCNN Affinity-ranked ACIM/DCNM Schemes",
    )
    return parser.parse_args()


def load_ranked_layers(path: Path) -> tuple[list[dict], dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    layers = payload["layers"]
    if not layers:
        raise ValueError(f"No layers found in {path}")
    return layers, payload.get("hardware_constants", {})


def layer_energy_latency(layer: dict, accelerator: str) -> dict:
    edp = layer["edp"]
    details = edp[f"{accelerator.lower()}_details"]
    energy = float(details["energy"])
    latency = float(details["latency"])
    return {
        "index": int(layer["index"]),
        "rank": int(layer["rank"]),
        "full_name": layer["full_name"],
        "short_name": layer["short_name"],
        "stage": layer["stage"],
        "module_type": layer["module_type"],
        "accelerator": accelerator,
        "energy_j": energy,
        "latency_s": latency,
        "power_w": energy / latency if latency > 0 else 0.0,
        "edp_j_s": energy * latency,
        "flops_per_frame": float(edp["flops"]),
        "macs_per_frame": float(edp["macs"]),
        "R": float(edp["R"]),
        "C": float(edp["C"]),
        "active_rows": float(edp["active_rows"]),
        "weight_shape": layer["weight_shape"],
        "affinity_score": float(layer["affinity_score"]),
        "total_kld_v1": float(layer["total_kld_v1"]),
        "edp_diff_dcnm_minus_acim": float(layer["edp_diff"]),
    }


def build_scheme(scheme_index: int, ranked_layers: list[dict]) -> dict:
    acim_indices = {int(layer["index"]) for layer in ranked_layers[:scheme_index]}
    per_layer = []
    for layer in sorted(ranked_layers, key=lambda item: int(item["index"])):
        accelerator = "ACIM" if int(layer["index"]) in acim_indices else "DCNM"
        per_layer.append(layer_energy_latency(layer, accelerator))

    total_energy = sum(item["energy_j"] for item in per_layer)
    total_latency = sum(item["latency_s"] for item in per_layer)
    acim_energy = sum(item["energy_j"] for item in per_layer if item["accelerator"] == "ACIM")
    dcnm_energy = total_energy - acim_energy
    acim_latency = sum(item["latency_s"] for item in per_layer if item["accelerator"] == "ACIM")
    dcnm_latency = total_latency - acim_latency

    added_layers = ranked_layers[:scheme_index]
    return {
        "scheme": f"scheme{scheme_index}",
        "scheme_index": scheme_index,
        "acim_count": len(added_layers),
        "dcnm_count": len(ranked_layers) - len(added_layers),
        "acim_layers_by_affinity_order": [
            {
                "rank": int(layer["rank"]),
                "index": int(layer["index"]),
                "full_name": layer["full_name"],
                "short_name": layer["short_name"],
                "affinity_score": float(layer["affinity_score"]),
            }
            for layer in added_layers
        ],
        "dcnm_layers_by_affinity_order": [
            {
                "rank": int(layer["rank"]),
                "index": int(layer["index"]),
                "full_name": layer["full_name"],
                "short_name": layer["short_name"],
                "affinity_score": float(layer["affinity_score"]),
            }
            for layer in ranked_layers[scheme_index:]
        ],
        "energy_j_per_frame": total_energy,
        "energy_mj_per_frame": total_energy * 1e3,
        "latency_s_per_frame": total_latency,
        "latency_ms_per_frame": total_latency * 1e3,
        "average_power_w": total_energy / total_latency if total_latency > 0 else 0.0,
        "edp_j_s_per_frame": total_energy * total_latency,
        "acim_energy_j_per_frame": acim_energy,
        "dcnm_energy_j_per_frame": dcnm_energy,
        "acim_latency_s_per_frame": acim_latency,
        "dcnm_latency_s_per_frame": dcnm_latency,
        "per_layer": per_layer,
    }


def add_rank_field(layers: list[dict]) -> list[dict]:
    ranked = []
    for rank, layer in enumerate(layers, start=1):
        row = dict(layer)
        row["rank"] = rank
        ranked.append(row)
    return ranked


def add_baseline_normalization(schemes: list[dict]) -> None:
    baseline = schemes[0]
    for scheme in schemes:
        scheme["relative_to_scheme0_dcnm"] = {
            "energy_ratio": scheme["energy_j_per_frame"] / baseline["energy_j_per_frame"],
            "latency_ratio": scheme["latency_s_per_frame"] / baseline["latency_s_per_frame"],
            "edp_ratio": scheme["edp_j_s_per_frame"] / baseline["edp_j_s_per_frame"],
            "energy_reduction_percent": (
                1.0 - scheme["energy_j_per_frame"] / baseline["energy_j_per_frame"]
            )
            * 100.0,
            "latency_reduction_percent": (
                1.0 - scheme["latency_s_per_frame"] / baseline["latency_s_per_frame"]
            )
            * 100.0,
            "edp_reduction_percent": (
                1.0 - scheme["edp_j_s_per_frame"] / baseline["edp_j_s_per_frame"]
            )
            * 100.0,
        }


def plot_schemes(schemes: list[dict], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    acim_counts = [scheme["acim_count"] for scheme in schemes]
    x_latency = [scheme["latency_ms_per_frame"] for scheme in schemes]
    y_energy = [scheme["energy_mj_per_frame"] for scheme in schemes]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    scatter = ax.scatter(
        x_latency,
        y_energy,
        c=acim_counts,
        cmap="viridis",
        s=82,
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
    )
    ax.plot(x_latency, y_energy, color="#7A7A7A", linewidth=1.6, alpha=0.65, zorder=2)

    offsets = {
        "scheme0": (-32, -5),
        "scheme1": (7, 6),
        "scheme2": (7, 6),
        "scheme3": (7, 6),
        "scheme4": (-36, -16),
        "scheme5": (-36, -4),
        "scheme6": (-36, 8),
        "scheme7": (6, 8),
        "scheme8": (6, 5),
        "scheme9": (7, 6),
        "scheme10": (7, 6),
        "scheme11": (7, -20),
        "scheme12": (7, -8),
        "scheme13": (7, 5),
        "scheme14": (7, 18),
    }
    for scheme in schemes:
        xytext = offsets.get(scheme["scheme"], (5, 4))
        ax.annotate(
            scheme["scheme"],
            (scheme["latency_ms_per_frame"], scheme["energy_mj_per_frame"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8.0,
        )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("ACIM layer count", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("Latency per frame (ms)", fontsize=12)
    ax.set_ylabel("Energy per frame (mJ)", fontsize=12)
    ax.set_ylim(0.245, 0.425)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric_by_scheme(
    schemes: list[dict],
    output_path: Path,
    metric_key: str,
    ylabel: str,
    title: str,
    color: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = [scheme["scheme_index"] for scheme in schemes]
    y = [scheme[metric_key] for scheme in schemes]
    labels = [scheme["scheme"] for scheme in schemes]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(
        x,
        y,
        marker="o",
        markersize=6.5,
        linewidth=2.2,
        color=color,
        markeredgecolor="black",
        markeredgewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("Scheme", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    ranked_layers, hardware_constants = load_ranked_layers(args.affinity_ranking)
    ranked_layers = add_rank_field(ranked_layers)
    schemes = [build_scheme(i, ranked_layers) for i in range(len(ranked_layers) + 1)]
    add_baseline_normalization(schemes)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_affinity_ranking": str(args.affinity_ranking.resolve()),
        "hardware_constants": hardware_constants,
        "aggregation_rule": (
            "For each scheme, per-layer energy and latency are selected from the saved "
            "ACIM/DCNM layer hardware model according to the mapping. Total network energy "
            "and latency are summed over target Conv2d/Linear layers."
        ),
        "scheme_rule": (
            "schemeN maps the first N layers in affinity descending order to ACIM; "
            "all remaining target layers stay on DCNM."
        ),
        "scheme_count_note": (
            "target_layers_v0 contains 14 Conv2d/Linear layers, so a strict one-layer-at-a-time "
            "sequence is scheme0..scheme14. Existing trained scheme7/scheme8 correspond to "
            "the first 7/8 affinity-ranked ACIM layers."
        ),
        "affinity_rank_order": [
            {
                "rank": int(layer["rank"]),
                "index": int(layer["index"]),
                "full_name": layer["full_name"],
                "short_name": layer["short_name"],
                "stage": layer["stage"],
                "module_type": layer["module_type"],
                "affinity_score": float(layer["affinity_score"]),
                "total_kld_v1": float(layer["total_kld_v1"]),
                "edp_diff_dcnm_minus_acim": float(layer["edp_diff"]),
            }
            for layer in ranked_layers
        ],
        "schemes": schemes,
    }

    output_json = args.output_dir / args.json_name
    output_plot = args.output_dir / args.plot_name
    output_energy_plot = args.output_dir / args.energy_plot_name
    output_latency_plot = args.output_dir / args.latency_plot_name
    output_edp_plot = args.output_dir / args.edp_plot_name
    write_json(output_json, payload)
    plot_schemes(schemes, output_plot, args.title)
    plot_metric_by_scheme(
        schemes,
        output_energy_plot,
        "energy_mj_per_frame",
        "Energy per frame (mJ)",
        "Energy Across Affinity-ranked Schemes",
        "#3B82A0",
    )
    plot_metric_by_scheme(
        schemes,
        output_latency_plot,
        "latency_ms_per_frame",
        "Latency per frame (ms)",
        "Latency Across Affinity-ranked Schemes",
        "#C47A39",
    )
    plot_metric_by_scheme(
        schemes,
        output_edp_plot,
        "edp_j_s_per_frame",
        "EDP per frame (J*s)",
        "EDP Across Affinity-ranked Schemes",
        "#5F8F3F",
    )

    print(f"Wrote JSON: {output_json}")
    print(f"Wrote plot: {output_plot}")
    print(f"Wrote energy plot: {output_energy_plot}")
    print(f"Wrote latency plot: {output_latency_plot}")
    print(f"Wrote EDP plot: {output_edp_plot}")
    print("Scheme summary:")
    for scheme in schemes:
        rel = scheme["relative_to_scheme0_dcnm"]
        print(
            f"  {scheme['scheme']:<8} ACIM={scheme['acim_count']:2d} "
            f"energy={scheme['energy_mj_per_frame']:.6f} mJ "
            f"latency={scheme['latency_ms_per_frame']:.6f} ms "
            f"power={scheme['average_power_w']:.6f} W "
            f"EDP={scheme['edp_j_s_per_frame']:.6e} J*s "
            f"EDP_red={rel['edp_reduction_percent']:.3f}%"
        )


if __name__ == "__main__":
    main()
