"""Causal Directed Acyclic Graph (DAG) Specification and Visualization.

Implements the structural causal model matching the research proposal.
Defines:
- Exposures (A): financial_hardship, long_commute, inadequate_digital_access, heavy_part_time_work
- Confounders / Adjustment Set (W): district_type, household_income_bracket, year_of_study,
  faculty, baseline_gpa_band, baseline_engagement_score, dependents_count
- Mediators (M): later_gpa_decline, later_engagement_decline (strictly EXCLUDED from total effect adjustment)
- Outcome (Y): institutional attrition

Renders publication diagram to reports/figures/dag.png.
"""

from pathlib import Path
from typing import Dict, List, Set
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
    "baseline_engagement_score",
    "dependents_count",
]

MEDIATORS = [
    "later_gpa_decline",
    "later_engagement_decline",
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
    for m in MEDIATORS:
        G.add_node(m, role="mediator")
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

    # Year of study affects employment and academic baselines
    G.add_edge("year_of_study", "heavy_part_time_work")
    G.add_edge("year_of_study", "baseline_engagement_score")

    # Dependents count affects financial hardship, study space / employment
    G.add_edge("dependents_count", "financial_hardship")
    G.add_edge("dependents_count", "heavy_part_time_work")

    # Faculty affects engagement and baseline GPA distributions
    G.add_edge("faculty", "baseline_gpa_band")
    G.add_edge("faculty", "baseline_engagement_score")

    # Baseline academic standing affects financial scholarship/hardship & work
    G.add_edge("baseline_gpa_band", "baseline_engagement_score")
    G.add_edge("baseline_gpa_band", "heavy_part_time_work")

    # Directed edges: Confounders -> Outcome (Direct Confounding paths)
    for c in CONFOUNDERS:
        G.add_edge(c, OUTCOME)

    # Directed edges: Exposures -> Mediators
    for e in EXPOSURES:
        G.add_edge(e, "later_gpa_decline")
        G.add_edge(e, "later_engagement_decline")

    # Directed edges: Exposures -> Outcome (Direct causal effects)
    for e in EXPOSURES:
        G.add_edge(e, OUTCOME)

    # Directed edges: Mediators -> Outcome
    for m in MEDIATORS:
        G.add_edge(m, OUTCOME)

    assert nx.is_directed_acyclic_graph(G), "Causal graph contains cycles!"
    return G


def get_adjustment_set(exposure: str) -> List[str]:
    """Return the theoretically valid adjustment set for estimating total effect of exposure on outcome.
    
    Excludes mediators and exposure itself; handles exposure-specific confounder overlaps.
    """
    base_set = set(CONFOUNDERS)
    if exposure == "financial_hardship":
        # When financial_hardship is exposure (defined via income & stress),
        # income is kept or handled as component confounder
        pass
    # Never include mediators
    for m in MEDIATORS:
        base_set.discard(m)
    base_set.discard(exposure)
    base_set.discard(OUTCOME)
    return sorted(list(base_set))


def render_dag_figure(output_path: Path = Path("reports/figures/dag.png")) -> Path:
    """Render a publication-quality diagram of the Causal DAG."""
    ensure_dir(output_path.parent)
    G = build_causal_dag()

    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)

    # Node positions partitioned by causal timeline: Confounders (Left) -> Exposures (Mid-Left) -> Mediators (Mid-Right) -> Outcome (Right)
    pos = {
        "district_type": (0.1, 0.85),
        "household_income_bracket": (0.1, 0.70),
        "dependents_count": (0.1, 0.55),
        "faculty": (0.1, 0.40),
        "year_of_study": (0.1, 0.25),
        "baseline_gpa_band": (0.28, 0.75),
        "baseline_engagement_score": (0.28, 0.35),

        "financial_hardship": (0.48, 0.80),
        "long_commute": (0.48, 0.60),
        "inadequate_digital_access": (0.48, 0.40),
        "heavy_part_time_work": (0.48, 0.20),

        "later_gpa_decline": (0.75, 0.65),
        "later_engagement_decline": (0.75, 0.35),

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
        elif role == "mediator":
            node_colors.append("#9b59b6")  # Purple
        else:
            node_colors.append("#e74c3c")  # Red

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2400, alpha=0.9, ax=ax)

    # Draw edges with different styling:
    # Mediator edges dashed purple, Confounder edges thin grey, Exposure edges solid dark
    for u, v in G.edges():
        if v in MEDIATORS or u in MEDIATORS:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], edge_color="#9b59b6", style="dashed",
                arrows=True, arrowstyle="-|>", arrowsize=14, alpha=0.6, ax=ax, width=1.4
            )
        elif u in EXPOSURES and v == OUTCOME:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], edge_color="#2c3e50", style="solid",
                arrows=True, arrowstyle="-|>", arrowsize=18, alpha=0.85, ax=ax, width=2.2
            )
        else:
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], edge_color="#7f8c8d", style="solid",
                arrows=True, arrowstyle="-|>", arrowsize=12, alpha=0.45, ax=ax, width=1.0
            )

    # Format labels
    labels = {node: node.replace("_", "\n") for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family="sans-serif", font_weight="bold", ax=ax)

    # Add Legend & Annotations
    ax.text(0.1, 0.96, "Baseline Confounders (W)\n[Adjusted Set]", ha="center", va="center", color="#2980b9", fontweight="bold", fontsize=11)
    ax.text(0.48, 0.96, "Target Exposures (A)\n[Policy Levers]", ha="center", va="center", color="#d35400", fontweight="bold", fontsize=11)
    ax.text(0.75, 0.96, "Post-Exposure Mediators (M)\n[EXCLUDED from Total Effect]", ha="center", va="center", color="#8e44ad", fontweight="bold", fontsize=11)
    ax.text(0.95, 0.96, "Primary Outcome (Y)\n[Institutional Attrition]", ha="center", va="center", color="#c0392b", fontweight="bold", fontsize=11)

    ax.set_title("Structural Causal Model Directed Acyclic Graph (DAG)\nEnvironmental & Financial Risk Intelligence (SLIIT CDAP KND-CDAP-IT-05)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Rendered Causal DAG to {output_path}")
    return output_path


if __name__ == "__main__":
    render_dag_figure()
