"""
Feature engineering schematics and visualisations.

Illustrates the normalised [0, 1] x [0, 1] coordinate system, how
distance and angle to goal are calculated geometrically, and how
situation and shotType values are one-hot encoded and combined into
interaction features.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# --- Pitch constants (matching the training pipeline) ----------------------

_GOAL_CENTER = (1.0, 0.5)
_GOAL_POSTS = [(1.0, 0.45), (1.0, 0.55)]

# Real-world reference dimensions (metres) used for accurate pitch markings.
_PITCH_LENGTH = 105.0
_PITCH_WIDTH = 68.0

# Aspect ratio of a real pitch (width / height in data units, mapping normalised
# x-span to y-span).  With normalised coords both axes run 0→1, so we scale the
# figure so that 1 data-unit on x equals 105 m and 1 data-unit on y equals 68 m.
_ASPECT_RATIO = _PITCH_LENGTH / _PITCH_WIDTH  # ≈ 1.544


# --- Pitch drawing helper --------------------------------------------------

def _draw_pitch(
    ax: plt.Axes,
    pitch_colour: str = "#2d6a2d",
    line_colour: str = "white",
) -> None:
    """
    Draws a full-pitch football diagram with normalised [0, 1] x [0, 1]
    coordinates. The attacking direction is left-to-right; the goal
    relevant to the xG models sits at x = 1.0.

    Circles and arcs are drawn as Ellipses whose x/y radii are scaled by
    the inverse aspect ratio so they appear circular at the 105:68 figure
    proportions.

    Args:
        ax: Matplotlib Axes on which the pitch is drawn.
        pitch_colour: Background fill colour for the playing surface.
        line_colour: Colour used for all pitch markings and lines.
    """
    ax.set_facecolor(pitch_colour)
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)
    # No set_aspect — proportions are enforced via figsize instead.

    lw = 1.5

    # x-radius and y-radius of a circle with real-world radius r_m metres,
    # expressed in normalised coordinate units at the correct aspect ratio.
    def _rx(r_m: float) -> float:
        return r_m / _PITCH_LENGTH

    def _ry(r_m: float) -> float:
        return r_m / _PITCH_WIDTH

    # Outer boundary
    ax.add_patch(
        plt.Rectangle(
            (0, 0), 1.0, 1.0,
            linewidth=lw, edgecolor=line_colour, facecolor="none", zorder=2,
        )
    )

    # Halfway line
    ax.plot([0.5, 0.5], [0, 1], color=line_colour, lw=lw, zorder=2)

    # Centre circle (radius = 9.15 m)
    ax.add_patch(
        mpatches.Ellipse(
            (0.5, 0.5), 2 * _rx(9.15), 2 * _ry(9.15),
            color=line_colour, fill=False, lw=lw, zorder=2,
        )
    )
    ax.plot(0.5, 0.5, "o", color=line_colour, ms=3, zorder=3)

    # Penalty areas (18-yard box: 40.32 m wide, 16.5 m deep)
    box_depth = 16.5 / _PITCH_LENGTH
    box_half_w = 20.16 / _PITCH_WIDTH
    box_y0 = 0.5 - box_half_w
    box_y1 = 0.5 + box_half_w

    # Right (attacking) penalty area
    ax.add_patch(
        plt.Rectangle(
            (1.0 - box_depth, box_y0), box_depth, box_y1 - box_y0,
            linewidth=lw, edgecolor=line_colour, facecolor="none", zorder=2,
        )
    )
    # Left (defensive) penalty area
    ax.add_patch(
        plt.Rectangle(
            (0, box_y0), box_depth, box_y1 - box_y0,
            linewidth=lw, edgecolor=line_colour, facecolor="none", zorder=2,
        )
    )

    # 6-yard boxes (5.5 m deep, 18.32 m wide)
    six_depth = 5.5 / _PITCH_LENGTH
    six_half_w = 9.16 / _PITCH_WIDTH
    six_y0 = 0.5 - six_half_w
    six_y1 = 0.5 + six_half_w

    ax.add_patch(
        plt.Rectangle(
            (1.0 - six_depth, six_y0), six_depth, six_y1 - six_y0,
            linewidth=lw, edgecolor=line_colour, facecolor="none", zorder=2,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (0, six_y0), six_depth, six_y1 - six_y0,
            linewidth=lw, edgecolor=line_colour, facecolor="none", zorder=2,
        )
    )

    # Penalty spots (11 m from goal line)
    penalty_x = 11.0 / _PITCH_LENGTH
    ax.plot(1.0 - penalty_x, 0.5, "o", color=line_colour, ms=3, zorder=3)
    ax.plot(penalty_x, 0.5, "o", color=line_colour, ms=3, zorder=3)

    # Penalty arcs (radius = 9.15 m, centre at penalty spot).
    # Drawn as arc segments of an Ellipse to respect the aspect-ratio scaling.
    for spot_x, theta1, theta2 in [
        (1.0 - penalty_x, -53, 53),
        (penalty_x, 127, 233),
    ]:
        theta = np.linspace(np.radians(theta1), np.radians(theta2), 80)
        arc_x = spot_x + _rx(9.15) * np.cos(theta)
        arc_y = 0.5 + _ry(9.15) * np.sin(theta)
        # Clip to outside the penalty area
        if spot_x > 0.5:
            mask = arc_x <= (1.0 - box_depth)
        else:
            mask = arc_x >= box_depth
        ax.plot(arc_x[mask], arc_y[mask], color=line_colour, lw=lw, zorder=2)

    # Goals: narrow rectangles beyond the goal line
    goal_half_h = 0.05  # matches GOAL_POSTS constants (0.45 to 0.55)
    goal_depth = 0.012

    ax.add_patch(
        plt.Rectangle(
            (1.0, 0.5 - goal_half_h), goal_depth, 2 * goal_half_h,
            linewidth=lw, edgecolor=line_colour, facecolor=line_colour,
            alpha=0.5, zorder=2,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (-goal_depth, 0.5 - goal_half_h), goal_depth, 2 * goal_half_h,
            linewidth=lw, edgecolor=line_colour, facecolor=line_colour,
            alpha=0.5, zorder=2,
        )
    )


# --- Coordinate system diagram ---------------------------------------------

def plot_coordinate_system(output_dir: Path) -> None:
    """
    Saves an annotated pitch diagram illustrating the normalised [0, 1] x [0, 1]
    coordinate system used across the dataset.

    Args:
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure height chosen so the pitch body renders at 105:68 proportions.
    _fig_w = 14.0
    _fig_h = _fig_w / _ASPECT_RATIO
    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
    _draw_pitch(ax)

    # Axis tick configuration
    tick_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.set_xticklabels([f"{v:.2f}" for v in tick_vals], fontsize=9, color="white")
    ax.set_yticklabels([f"{v:.2f}" for v in tick_vals], fontsize=9, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    offset = 0.028
    label_kw = dict(fontsize=9, fontweight="bold", color="#FFD700")

    # Corner coordinate labels
    ax.text(0 + offset, 0 + offset, "(0.00, 0.00)", **label_kw)
    ax.text(1 - offset, 0 + offset, "(1.00, 0.00)", ha="right", **label_kw)
    ax.text(0 + offset, 1 - offset, "(0.00, 1.00)", va="top", **label_kw)
    ax.text(1 - offset, 1 - offset, "(1.00, 1.00)", ha="right", va="top", **label_kw)

    # Goal centre annotation
    ax.annotate(
        "Goal centre\n(1.00, 0.50)",
        xy=(1.0, 0.5),
        xytext=(0.80, 0.5),
        fontsize=9,
        color="#FFD700",
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.5),
    )

    # Attacking direction arrow
    ax.annotate(
        "",
        xy=(0.93, 0.09),
        xytext=(0.72, 0.09),
        arrowprops=dict(arrowstyle="-|>", color="#FFD700", lw=2),
    )
    ax.text(0.825, 0.05, "Attacking direction", color="#FFD700", fontsize=9, ha="center")

    # Pitch centre label
    ax.text(0.5, 0.5 + 0.04, "(0.50, 0.50)", ha="center", fontsize=8,
            color="#FFD700", fontweight="bold")

    ax.set_title(
        "Normalised Pitch Coordinate System  ·  X ∈ [0, 1],  Y ∈ [0, 1]\n"
        "All shots are recorded with attacking team shooting towards x = 1.0",
        fontsize=12, fontweight="bold", color="white", pad=12,
    )

    fig.patch.set_facecolor("#1a1a2e")
    output_path = output_dir / "coordinate_system.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# --- Distance and angle schematic ------------------------------------------

def plot_distance_angle_schematic(output_dir: Path) -> None:
    """
    Saves a two-panel pitch diagram with example shots annotated to illustrate
    how distance and angle to goal are computed.  Panels are stacked
    vertically so each pitch renders at the correct 105:68 aspect ratio
    with no label overlap.

    Distance: Euclidean from shot coords to goal centre.
    Angle: arccos of normalised dot product of vectors to each goalpost.

    Args:
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shots = [
        {"X": 0.78, "Y": 0.50, "label": "Shot A  (central)"},
        {"X": 0.87, "Y": 0.22, "label": "Shot B  (wide angle)"},
    ]
    shot_colours = ["#00BFFF", "#FF6347"]

    # Each panel width is the full figure width; height preserves 105:68 ratio.
    _panel_w = 12.0
    _panel_h = _panel_w / _ASPECT_RATIO
    fig, axes = plt.subplots(2, 1, figsize=(_panel_w, _panel_h * 2))
    fig.patch.set_facecolor("#1a1a2e")

    gx, gy = _GOAL_CENTER
    gp1x, gp1y = _GOAL_POSTS[0]
    gp2x, gp2y = _GOAL_POSTS[1]

    for ax, shot, colour in zip(axes, shots, shot_colours):
        _draw_pitch(ax)

        sx, sy = shot["X"], shot["Y"]

        # Compute distance
        dist = np.hypot(sx - gx, sy - gy)

        # Compute angle via normalised dot product of goalpost vectors,
        # matching the training and inference pipelines.
        v1x, v1y = gp1x - sx, gp1y - sy
        v2x, v2y = gp2x - sx, gp2y - sy
        dot = v1x * v2x + v1y * v2y
        mag_v1 = np.hypot(v1x, v1y)
        mag_v2 = np.hypot(v2x, v2y)
        cos_angle = np.clip(dot / (mag_v1 * mag_v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        # Distance dashed line from shot to goal centre
        ax.plot([sx, gx], [sy, gy], "--", color=colour, lw=2, zorder=5)

        # Distance label: placed along the midpoint, offset upward for Shot A
        # and downward for Shot B to avoid overlapping the pitch elements.
        mid_x = (sx + gx) / 2
        mid_y = (sy + gy) / 2
        d_label_offset = 0.055 if sy >= 0.5 else -0.055
        ax.text(
            mid_x,
            mid_y + d_label_offset,
            f"d = {dist:.3f}",
            color=colour,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

        # Lines from shot to both goalposts (angle triangle)
        ax.plot([sx, gp1x], [sy, gp1y], "-", color="#FFD700", lw=1.8, zorder=5, alpha=0.85)
        ax.plot([sx, gp2x], [sy, gp2y], "-", color="#FFD700", lw=1.8, zorder=5, alpha=0.85)

        # Shot location marker
        ax.plot(
            sx, sy, "o",
            color=colour, ms=11, zorder=7,
            markeredgecolor="white", markeredgewidth=1.5,
        )

        # Shot label: above for central shot (clear of goal), below for wide shot
        label_offset = 0.07 if sy >= 0.5 else -0.07
        label_va = "bottom" if sy >= 0.5 else "top"
        ax.text(
            sx, sy + label_offset,
            shot["label"],
            color=colour, fontsize=8.5,
            fontweight="bold", ha="center", va=label_va, zorder=8,
        )

        # Angle annotation box: placed well left of the shot point to avoid
        # overlapping the goal triangle.
        angle_text = (
            f"θ = {angle_deg:.1f}°\n"
            f"({angle_rad:.3f} rad)"
        )
        angle_box_x = sx - 0.20
        angle_box_y = sy + 0.09 if sy < 0.5 else sy - 0.05
        ax.text(
            angle_box_x, angle_box_y,
            angle_text,
            color="#FFD700",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom" if sy < 0.5 else "top",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1a1a2e", ec="#FFD700", alpha=0.88),
            zorder=9,
        )

        # Formula inset: top-left corner of each panel
        formula = (
            r"$d = \sqrt{(X-1)^2 + (Y-0.5)^2}$"
            "\n"
            r"$\theta = \arccos\!\left(\frac{v_1 \cdot v_2}{\|v_1\|\|v_2\|}\right)$"
        )
        ax.text(
            0.02, 0.98,
            formula,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="white", alpha=0.75),
            zorder=10,
        )

        ax.set_title(
            shot["label"],
            fontsize=11,
            fontweight="bold",
            color="white",
            pad=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Distance & Angle Features  ·  Normalised Coordinate System",
        fontsize=14, fontweight="bold", color="white", y=1.005,
    )
    plt.tight_layout(h_pad=2.0)

    output_path = output_dir / "distance_angle_schematic.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# --- Feature construction overview -----------------------------------------

def plot_feature_construction_overview(output_dir: Path) -> None:
    """
    Saves a three-panel illustration showing how situation and shotType are
    one-hot encoded and then combined into interaction features.

    Args:
        output_dir: Directory into which the figure is saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    situations = ["OpenPlay", "DirectFreekick", "SetPiece", "FromCorner", "Penalty"]
    shot_types = ["RightFoot", "LeftFoot", "Head", "OtherBodyPart"]
    example_sit = "OpenPlay"
    example_type = "RightFoot"
    all_interactions = [
        f"{s}_{t}" for s in situations for t in shot_types
        if (s, t) in [  # Only realistic combinations present in data
            ("OpenPlay", "RightFoot"), ("OpenPlay", "LeftFoot"),
            ("OpenPlay", "Head"), ("OpenPlay", "OtherBodyPart"),
            ("DirectFreekick", "RightFoot"), ("DirectFreekick", "LeftFoot"),
            ("DirectFreekick", "OtherBodyPart"),
            ("SetPiece", "RightFoot"), ("SetPiece", "LeftFoot"),
            ("SetPiece", "Head"), ("SetPiece", "OtherBodyPart"),
            ("FromCorner", "RightFoot"), ("FromCorner", "LeftFoot"),
            ("FromCorner", "Head"), ("FromCorner", "OtherBodyPart"),
            ("Penalty", "RightFoot"), ("Penalty", "LeftFoot"),
        ]
    ]
    active_interaction = f"{example_sit}_{example_type}"

    panels = [
        ("situation_*", situations, example_sit),
        ("shotType_*", shot_types, example_type),
        ("interaction_*", all_interactions, active_interaction),
    ]

    _active_col = "#2196F3"
    _inactive_col = "#E0E0E0"

    # Use a taller figure to accommodate the interaction panel
    col_widths = [1, 1, 2.5]
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 7),
        gridspec_kw={"width_ratios": col_widths},
    )
    fig.patch.set_facecolor("#f8f9fa")

    for ax, (panel_title, values, active) in zip(axes, panels):
        n = len(values)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, n + 0.6)
        ax.axis("off")

        ax.text(
            0.5, n + 0.35,
            panel_title,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="#222",
        )

        for i, val in enumerate(values):
            row_y = n - 1 - i
            is_active = val == active
            cell_col = _active_col if is_active else _inactive_col
            txt_col = "white" if is_active else "#555"

            # Feature name cell
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.02, row_y + 0.08), 0.70, 0.72,
                boxstyle="round,pad=0.02",
                facecolor=cell_col, edgecolor="white", linewidth=1.2,
            ))
            ax.text(
                0.37, row_y + 0.44, val,
                ha="center", va="center",
                fontsize=7.5,
                color=txt_col,
                fontweight="bold" if is_active else "normal",
            )

            # Binary value cell
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.76, row_y + 0.08), 0.20, 0.72,
                boxstyle="round,pad=0.02",
                facecolor=cell_col, edgecolor="white", linewidth=1.2,
            ))
            ax.text(
                0.86, row_y + 0.44,
                "1" if is_active else "0",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=txt_col,
            )

    fig.suptitle(
        f'Feature Construction  ·  Example: situation="{example_sit}", '
        f'shotType="{example_type}"\n'
        "Highlighted cells (=1) show the active category for this shot.",
        fontsize=11, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    output_path = output_dir / "feature_construction_overview.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# --- Entry point -----------------------------------------------------------

def run_feature_engineering_analysis(project_root: Path) -> None:
    """
    Orchestrates all feature engineering visualisations.

    Args:
        project_root: Absolute path to the backend project root.
    """
    output_dir = project_root / "exploration" / "figures" / "Feature_Engineering"

    print("  Plotting coordinate system diagram...")
    plot_coordinate_system(output_dir)

    print("  Plotting distance/angle schematic...")
    plot_distance_angle_schematic(output_dir)

    print("  Plotting feature construction overview...")
    plot_feature_construction_overview(output_dir)
