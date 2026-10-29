"""Causal Directed Acyclic Graph (DAG) Specification and Visualization.

Implements the structural causal model matching the research proposal.
Defines the observed environmental and financial exposures, confounders, and
institutional attrition outcome used by the legacy causal analysis.

Renders publication diagram to reports/figures/dag.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.utils.logging_config import setup_logger
from src.utils.paths import ensure_dir

logger = setup_logger("causal_dag")

EXPOSURES = [
    "financial_hardship",
    "long_commute",
    "inadequate_digital_access",
    "heavy_part_time_work",
]

CONFOUNDERS = [
    "district_type",
    "household_income_bracket",
    "year_of_study",
    "faculty",
    "baseline_gpa_band",
    "dependents_count",
]

OUTCOME = "attrition"


def build_causal_dag() -> nx.DiGraph:
    """Build the formal NetworkX Directed Acyclic Graph."""
    G = nx.DiGraph()

    # Add nodes with roles
    for c in CONFOUNDERS:
        G.add_node(c, role="confounder")
    for e in EXPOSURES:
        G.add_node(e, role="exposure")
    G.add_node(OUTCOME, role="outcome")

    # Directed edges: Confounders -> Exposures
    # District affects commute and digital access and income
    G.add_edge("district_type", "household_income_bracket")
    G.add_edge("district_type", "long_commute")
    G.add_edge("district_type", "inadequate_digital_access")

    # Income affects financial hardship, digital access, heavy part-time work
    G.add_edge("household_income_bracket", "financial_hardship")
    G.add_edge("household_income_bracket", "inadequate_digital_access")
    G.add_edge("household_income_bracket", "heavy_part_time_work")

    # Year of study affects employment.
    G.add_edge("year_of_study", "heavy_part_time_work")

    # Dependents count affects financial hardship, study space / employment
    G.add_edge("dependents_count", "financial_hardship")
    G.add_edge("dependents_count", "heavy_part_time_work")

    # Faculty affects baseline GPA distributions.
    G.add_edge("faculty", "baseline_gpa_band")

    # Baseline academic standing affects financial aid and work.
    G.add_edge("baseline_gpa_band", "heavy_part_time_work")

    # Directed edges: Confounders -> Outcome (Direct Confounding paths)
    for c in CONFOUNDERS:
        G.add_edge(c, OUTCOME)

    # Directed edges: Exposures -> Outcome (Direct causal effects)
    for e in EXPOSURES:
        G.add_edge(e, OUTCOME)

    assert nx.is_directed_acyclic_graph(G), "Causal graph contains cycles!"
    return G


def get_adjustment_set(exposure: str) -> list[str]:
    """Return the theoretically valid adjustment set for estimating total effect of exposure on outcome.

    Excludes the exposure itself and the outcome.
    """
    base_set = set(CONFOUNDERS)
    if exposure == "financial_hardship":
        # When financial_hardship is exposure (defined via income & stress),
        # income is kept or handled as component confounder
        pass
    base_set.discard(exposure)
    base_set.discard(OUTCOME)
    return sorted(base_set)


def render_dag_figure(output_path: Path = Path("reports/figures/dag.png")) -> Path:
    """Render a publication-quality diagram of the Causal DAG."""
    ensure_dir(output_path.parent)
    G = build_causal_dag()

    _, ax = plt.subplots(figsize=(14, 9), dpi=300)

    # Confounders feed exposures and the outcome.
    pos = {
        "district_type": (0.1, 0.85),
        "household_income_bracket": (0.1, 0.70),
        "dependents_count": (0.1, 0.55),
        "faculty": (0.1, 0.40),
        "year_of_study": (0.1, 0.25),
        "baseline_gpa_band": (0.28, 0.75),
        "financial_hardship": (0.48, 0.80),
        "long_commute": (0.48, 0.60),
        "inadequate_digital_access": (0.48, 0.40),
        "heavy_part_time_work": (0.48, 0.20),
        "attrition": (0.95, 0.50),
    }

    # Color mapping
    node_colors = []
    for node in G.nodes():
        role = G.nodes[node]["role"]
        if role == "confounder":
            node_colors.append("#3498db")  # Blue
        elif role == "exposure":
            node_colors.append("#e67e22")  # Orange
        else:
            node_colors.append("#e74c3c")  # Red

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=2400, alpha=0.9, ax=ax
    )

    # Draw exposure-to-outcome edges more prominently.
    for u, v in G.edges():
        if u in EXPOSURES and v == OUTCOME:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                edge_color="#2c3e50",
                style="solid",
                arrows=True,
                arrowstyle="-|>",
                arrowsize=18,
                alpha=0.85,
                ax=ax,
                width=2.2,
            )
        else:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                edge_color="#7f8c8d",
                style="solid",
                arrows=True,
                arrowstyle="-|>",
                arrowsize=12,
                alpha=0.45,
                ax=ax,
                width=1.0,
            )

    # Format labels
    labels = {node: node.replace("_", "\n") for node in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
        font_family="sans-serif",
        font_weight="bold",
        ax=ax,
    )

    # Add Legend & Annotations
    ax.text(
        0.1,
        0.96,
        "Baseline Confounders (W)\n[Adjusted Set]",
        ha="center",
        va="center",
        color="#2980b9",
        fontweight="bold",
        fontsize=11,
    )
    ax.text(
        0.48,
        0.96,
        "Target Exposures (A)\n[Policy Levers]",
        ha="center",
        va="center",
        color="#d35400",
        fontweight="bold",
        fontsize=11,
    )
    ax.text(
        0.95,
        0.96,
        "Primary Outcome (Y)\n[Institutional Attrition]",
        ha="center",
        va="center",
        color="#c0392b",
        fontweight="bold",
        fontsize=11,
    )

    ax.set_title(
        "Structural Causal Model Directed Acyclic Graph (DAG)\nEnvironmental & Financial Risk Intelligence (SLIIT CDAP KND-CDAP-IT-05)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Rendered Causal DAG to {output_path}")
    return output_path


if __name__ == "__main__":
    render_dag_figure()
