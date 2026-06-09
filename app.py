from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.special import gammaln, logsumexp

from vote_probability import alpha_from_counts, alpha_from_mean, collision_probability_dirichlet_mixture


SQRT3_OVER_2 = np.sqrt(3) / 2


st.set_page_config(
    page_title="Vote Count Collision Lab",
    page_icon=None,
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 120, 120, 0.25);
        border-radius: 14px;
        padding: 1rem;
        background: rgba(120, 120, 120, 0.06);
    }
    .small-note {
        color: rgba(120, 120, 120, 0.95);
        font-size: 0.92rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_collision_probability(total_a: int, total_b: int, weights_a, alphas_a, weights_b, alphas_b) -> float:
    return collision_probability_dirichlet_mixture(
        total_a,
        total_b,
        np.array(weights_a, dtype=float),
        [np.array(alpha, dtype=float) for alpha in alphas_a],
        np.array(weights_b, dtype=float),
        [np.array(alpha, dtype=float) for alpha in alphas_b],
    )


def format_probability(probability: float) -> tuple[str, str]:
    return f"{probability:.6e}", f"1 in {1 / probability:,.0f}"


def simplex_grid(resolution: int = 55) -> np.ndarray:
    points = []
    for i in range(1, resolution):
        for j in range(1, resolution - i):
            k = resolution - i - j
            if k > 0:
                points.append((i / resolution, j / resolution, k / resolution))
    return np.array(points)


def dirichlet_log_density(points: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    log_norm = gammaln(alpha.sum()) - np.sum(gammaln(alpha))
    return log_norm + np.sum((alpha - 1) * np.log(points), axis=1)


def mixture_heat_values(
    points: np.ndarray,
    weights: np.ndarray,
    alphas: list[np.ndarray],
    balance_components: bool,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    if balance_components:
        component_values = [
            weight * np.exp(dirichlet_log_density(points, alpha) - np.max(dirichlet_log_density(points, alpha)))
            for weight, alpha in zip(weights, alphas)
            if weight > 0
        ]
        heat_values = np.sum(np.vstack(component_values), axis=0)
        return heat_values / np.max(heat_values)

    component_logs = [
        np.log(weight) + dirichlet_log_density(points, alpha)
        for weight, alpha in zip(weights, alphas)
        if weight > 0
    ]
    log_density = logsumexp(np.vstack(component_logs), axis=0)
    return np.exp(log_density - np.max(log_density))


def barycentric_to_cartesian(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points[:, 1] + 0.5 * points[:, 2]
    y = SQRT3_OVER_2 * points[:, 2]
    return x, y


def plot_dirichlet_triangle(weights: np.ndarray, alphas: list[np.ndarray], title: str, show_components: bool):
    points = simplex_grid()
    heat_values = mixture_heat_values(points, weights, alphas, balance_components=show_components)
    x, y = barycentric_to_cartesian(points)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    contour = ax.tricontourf(x, y, heat_values, levels=18, cmap="YlOrRd")
    ax.triplot(x, y, color="white", alpha=0.05, linewidth=0.3)

    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, SQRT3_OVER_2, 0]
    ax.plot(triangle_x, triangle_y, color="black", linewidth=1.2)

    ax.text(-0.04, -0.04, "Candidate 1", ha="left", va="top", fontsize=10)
    ax.text(1.04, -0.04, "Candidate 2", ha="right", va="top", fontsize=10)
    ax.text(0.5, SQRT3_OVER_2 + 0.04, "Other", ha="center", va="bottom", fontsize=10)

    if show_components:
        means = np.array([alpha / alpha.sum() for alpha in alphas])
        mean_x, mean_y = barycentric_to_cartesian(means)
        marker_sizes = 80 + 520 * np.asarray(weights, dtype=float) / np.max(weights)
        ax.scatter(
            mean_x,
            mean_y,
            s=marker_sizes,
            c="white",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        for idx, (x_coord, y_coord) in enumerate(zip(mean_x, mean_y), start=1):
            ax.text(
                x_coord,
                y_coord,
                str(idx),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                zorder=6,
            )

    ax.set_title(title, pad=18)
    ax.set_aspect("equal")
    ax.axis("off")

    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    if show_components:
        cbar.set_label("Component-balanced relative density")
    else:
        cbar.set_label("Relative probability density")
    fig.tight_layout()
    return fig


def component_controls(
    label: str,
    defaults: dict,
    key_prefix: str,
    manual_alpha_default: bool = False,
) -> tuple[float, np.ndarray]:
    weight = st.number_input(
        "Component weight",
        min_value=0.0,
        max_value=1_000.0,
        value=defaults["weight"],
        step=0.05,
        format="%.3f",
        key=f"{key_prefix}_weight",
    )

    count_cols = st.columns(3)
    cand1 = count_cols[0].number_input(
        "Candidate 1 count",
        min_value=0,
        value=defaults["counts"][0],
        step=1,
        key=f"{key_prefix}_cand1",
    )
    cand2 = count_cols[1].number_input(
        "Candidate 2 count",
        min_value=0,
        value=defaults["counts"][1],
        step=1,
        key=f"{key_prefix}_cand2",
    )
    other = count_cols[2].number_input(
        "Other count",
        min_value=0,
        value=defaults["counts"][2],
        step=1,
        key=f"{key_prefix}_other",
    )

    kappa = st.number_input(
        "Dirichlet concentration kappa",
        min_value=0.001,
        max_value=100_000.0,
        value=defaults["kappa"],
        step=1.0,
        format="%.3f",
        key=f"{key_prefix}_kappa",
    )

    alpha = alpha_from_counts([cand1, cand2, other], kappa)
    manual_alpha = st.checkbox(
        "Edit alpha manually",
        value=manual_alpha_default,
        key=f"{key_prefix}_manual_alpha",
    )
    if manual_alpha:
        alpha_cols = st.columns(3)
        alpha = np.array(
            [
                alpha_cols[0].number_input(
                    "alpha 1",
                    min_value=0.001,
                    value=float(alpha[0]),
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_alpha1",
                ),
                alpha_cols[1].number_input(
                    "alpha 2",
                    min_value=0.001,
                    value=float(alpha[1]),
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_alpha2",
                ),
                alpha_cols[2].number_input(
                    "alpha other",
                    min_value=0.001,
                    value=float(alpha[2]),
                    step=0.1,
                    format="%.3f",
                    key=f"{key_prefix}_alpha3",
                ),
            ]
        )

    direction = alpha / alpha.sum()
    st.markdown(
        f"""
        <div class="small-note">
        {label}: alpha = ({alpha[0]:,.2f}, {alpha[1]:,.2f}, {alpha[2]:,.2f})<br>
        mean share = ({direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f})
        </div>
        """,
        unsafe_allow_html=True,
    )
    return float(weight), alpha


def mixture_component_controls(label: str, defaults: dict, key_prefix: str) -> tuple[float, np.ndarray | None]:
    weight = st.number_input(
        "Component weight",
        min_value=0.0,
        max_value=1_000.0,
        value=defaults["weight"],
        step=0.05,
        format="%.3f",
        key=f"{key_prefix}_weight",
    )

    st.caption("Set the cluster center directly. Other share is implied as 1 - u1 - u2.")
    shape_cols = st.columns(3)
    u1 = shape_cols[0].number_input(
        "u1: candidate 1 mean share",
        min_value=0.001,
        max_value=0.999,
        value=defaults["u1"],
        step=0.01,
        format="%.3f",
        key=f"{key_prefix}_u1",
    )
    u2 = shape_cols[1].number_input(
        "u2: candidate 2 mean share",
        min_value=0.001,
        max_value=0.999,
        value=defaults["u2"],
        step=0.01,
        format="%.3f",
        key=f"{key_prefix}_u2",
    )
    kappa = shape_cols[2].number_input(
        "kappa",
        min_value=0.001,
        max_value=100_000.0,
        value=defaults["kappa"],
        step=1.0,
        format="%.3f",
        key=f"{key_prefix}_kappa",
    )

    other = 1.0 - u1 - u2
    if other <= 1e-12:
        st.warning("This cluster is invalid: u1 + u2 must be less than 1.")
        return float(weight), None

    alpha = alpha_from_mean(u1, u2, kappa)
    st.markdown(
        f"""
        <div class="small-note">
        {label}: u = ({u1:.3f}, {u2:.3f}, {other:.3f}); alpha = ({alpha[0]:,.2f}, {alpha[1]:,.2f}, {alpha[2]:,.2f})
        </div>
        """,
        unsafe_allow_html=True,
    )
    return float(weight), alpha


def district_panel(label: str, defaults: dict, key_prefix: str) -> tuple[int, np.ndarray, list[np.ndarray]]:
    st.subheader(label)
    total_votes = st.number_input(
        "Target total votes",
        min_value=1,
        max_value=20_000,
        value=defaults["total_votes"],
        step=1,
        key=f"{key_prefix}_total_votes",
    )

    st.caption("Calibration counts set the direction `u`; `kappa` sets concentration.")
    _, alpha = component_controls("Single component", defaults["components"][0], f"{key_prefix}_single")
    weights = np.array([1.0])
    alphas = [alpha]
    st.pyplot(
        plot_dirichlet_triangle(weights, alphas, f"{label} Dirichlet distribution", show_components=False),
        clear_figure=True,
    )
    return int(total_votes), weights, alphas


def mixture_district_panel(label: str, defaults: dict, key_prefix: str) -> tuple[int, np.ndarray, list[np.ndarray]]:
    st.subheader(label)
    total_votes = st.number_input(
        "Target total votes",
        min_value=1,
        max_value=20_000,
        value=defaults["total_votes"],
        step=1,
        key=f"{key_prefix}_total_votes",
    )

    component_count = st.selectbox(
        "Number of alignment clusters",
        options=[2, 3],
        index=1,
        key=f"{key_prefix}_component_count",
    )
    st.caption("Each cluster is a Dirichlet component; weights are normalized automatically.")

    weights = []
    alphas = []
    for idx in range(component_count):
        component_default = defaults["mixture_components"][idx]
        with st.expander(f"Cluster {idx + 1}", expanded=True):
            weight, alpha = mixture_component_controls(
                f"Cluster {idx + 1}",
                component_default,
                f"{key_prefix}_component_{idx}",
            )
            weights.append(weight)
            if alpha is not None:
                alphas.append(alpha)

    weights = np.array(weights, dtype=float)
    if weights.sum() <= 0:
        st.warning("At least one cluster weight must be positive. Using equal weights for the plot.")
        weights = np.ones_like(weights)
    if len(alphas) != len(weights):
        st.warning("Fix invalid cluster centers before plotting or computing probability.")
        return int(total_votes), np.array([1.0]), [alpha_from_mean(0.5, 0.4, 1.0)]
    normalized_weights = weights / weights.sum()
    st.markdown(
        f"<div class='small-note'>Normalized weights: {', '.join(f'{w:.3f}' for w in normalized_weights)}</div>",
        unsafe_allow_html=True,
    )
    st.pyplot(
        plot_dirichlet_triangle(
            normalized_weights,
            alphas,
            f"{label} mixture distribution",
            show_components=True,
        ),
        clear_figure=True,
    )
    return int(total_votes), normalized_weights, alphas


st.title("Vote Count Collision Lab")
# st.markdown(
#     """
#     Explore the probability that two fixed districts independently produce the same vote counts
#     for candidate 1 and candidate 2 under two chosen 3-way Dirichlet-Multinomial models.
#     This is a modeling calculator, not evidence for or against any election claim.
#     """
# )

district_a_defaults = {
    "total_votes": 4548,
    "components": [
        {"weight": 1.0, "counts": [3030, 1440, 78], "kappa": 10_000.0},
    ],
    "mixture_components": [
        {"weight": 0.50, "u1": 0.666, "u2": 0.317, "kappa": 80.0},
        {"weight": 0.25, "u1": 0.550, "u2": 0.418, "kappa": 60.0},
        {"weight": 0.25, "u1": 0.770, "u2": 0.187, "kappa": 60.0},
    ],
}
district_b_defaults = {
    "total_votes": 4540,
    "components": [
        {"weight": 1.0, "counts": [3030, 1440, 70], "kappa": 10_000.0},
    ],
    "mixture_components": [
        {"weight": 0.50, "u1": 0.667, "u2": 0.317, "kappa": 80.0},
        {"weight": 0.25, "u1": 0.548, "u2": 0.419, "kappa": 60.0},
        {"weight": 0.25, "u1": 0.769, "u2": 0.187, "kappa": 60.0},
    ],
}

model_mode = st.radio(
    "Distribution model",
    options=["Single Dirichlet", "Mixture of Dirichlet clusters"],
    horizontal=True,
    help="Underlying prior distribution family. Treat it as the distribution of voter tendency.",
)

if model_mode == "Mixture of Dirichlet clusters":
    st.info(
        "Each cluster corresponds to a group with a tendency."
        "You can experiment with clusters that fit your own pressumption."
    )
else:
    st.info("Single Dirichlet mode: each district has one smooth distribution over the 3-way vote-share simplex.")

left, right = st.columns(2)
with left:
    if model_mode == "Mixture of Dirichlet clusters":
        total_a, weights_a, alphas_a = mixture_district_panel("District 1", district_a_defaults, "district_a")
    else:
        total_a, weights_a, alphas_a = district_panel("District 1", district_a_defaults, "district_a")
with right:
    if model_mode == "Mixture of Dirichlet clusters":
        total_b, weights_b, alphas_b = mixture_district_panel("District 2", district_b_defaults, "district_b")
    else:
        total_b, weights_b, alphas_b = district_panel("District 2", district_b_defaults, "district_b")

st.divider()
st.header("Probability that two candidates have the same number of votes")
# st.markdown(
#     """
#     The event is: district 1 and district 2 have the same count for candidate 1
#     **and** the same count for candidate 2. The count for `other` may differ because
#     the district totals can differ.
#     """
# )

if st.button("Compute probability", type="primary"):
    with st.spinner("Summing exact Dirichlet-Multinomial probabilities..."):
        probability = cached_collision_probability(
            total_a,
            total_b,
            tuple(weights_a),
            tuple(tuple(alpha) for alpha in alphas_a),
            tuple(weights_b),
            tuple(tuple(alpha) for alpha in alphas_b),
        )

    probability_text, scale_text = format_probability(probability)
    metric_left, metric_right = st.columns(2)
    metric_left.metric("P(same candidate 1 and 2 counts)", probability_text)
    metric_right.metric("Probability scale", scale_text)
else:
    st.info("Set the two distributions and totals, then click Compute exact probability.")

# st.caption(
#     "Tip: in mixture mode, u1 and u2 move each cluster center directly; kappa controls within-cluster concentration; mixture weight controls how much mass each latent alignment cluster gets."
# )
