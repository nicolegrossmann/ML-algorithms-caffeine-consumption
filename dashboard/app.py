from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


@dataclass
class EffectProfile:
    name: str
    effect_scale: float
    jitter_scale: float
    mood_scale: float


EFFECT_PROFILES = {
    "Focused / productive": EffectProfile("Focused / productive", 1.0, 0.35, 0.2),
    "Balanced": EffectProfile("Balanced", 0.8, 0.2, 0.15),
    "Sensitive / anxious": EffectProfile("Sensitive / anxious", 0.7, 0.55, 0.35),
}


def caffeine_remaining(initial_mg: float, hours_since: float, half_life_hours: float) -> float:
    if half_life_hours <= 0:
        return 0.0
    return initial_mg * (0.5 ** (hours_since / half_life_hours))


def build_hourly_curve(
    cups: int,
    mg_per_cup: int,
    first_cup_hour: float,
    spacing_hours: float,
    half_life_hours: float,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    dose_hours = [first_cup_hour + i * spacing_hours for i in range(cups)]
    timeline = np.arange(0, horizon_hours + 1, 1)
    total = []

    for hour in timeline:
        mg_hour = 0.0
        for dose_h in dose_hours:
            if hour >= dose_h:
                mg_hour += caffeine_remaining(mg_per_cup, hour - dose_h, half_life_hours)
        total.append(mg_hour)

    return pd.DataFrame({"hour": timeline, "active_caffeine_mg": total})


def effect_indices(curve: pd.DataFrame, profile: EffectProfile, sugar_with_coffee: bool) -> pd.DataFrame:
    signal = curve["active_caffeine_mg"].to_numpy(dtype=float)
    norm = signal / max(signal.max(), 1.0)
    sugar_bonus = 1.15 if sugar_with_coffee else 1.0

    focus = np.clip(100 * np.tanh(profile.effect_scale * sugar_bonus * norm * 1.4), 0, 100)
    jitter = np.clip(100 * (norm ** 1.5) * profile.jitter_scale * sugar_bonus, 0, 100)
    mood = np.clip(100 * np.tanh(profile.mood_scale * sugar_bonus * norm * 1.1), 0, 100)

    out = curve.copy()
    out["focus_index"] = focus
    out["jitter_index"] = jitter
    out["mood_index"] = mood
    return out


def main() -> None:
    st.set_page_config(page_title="Caffeine Effect Simulator", page_icon="☕", layout="wide")

    st.title("Caffeine Effect & Half-life Simulator")
    st.caption("Interactive model for caffeine decay and perceived effects across a day.")

    with st.sidebar:
        st.header("Inputs")
        cups = st.slider("Cups of coffee", min_value=0, max_value=10, value=3, step=1)
        mg_per_cup = st.slider("Caffeine per cup (mg)", min_value=40, max_value=200, value=95, step=5)
        first_cup_hour = st.slider("First cup hour", min_value=0.0, max_value=12.0, value=8.0, step=0.5)
        spacing_hours = st.slider("Hours between cups", min_value=0.5, max_value=6.0, value=2.5, step=0.5)

        st.subheader("Metabolism")
        half_life_mode = st.selectbox(
            "Half-life preset",
            options=[
                "Fast metabolism (~3.5h)",
                "Average (~5h)",
                "Slow metabolism (~7h)",
                "Custom",
            ],
            index=1,
        )
        if half_life_mode == "Fast metabolism (~3.5h)":
            half_life_hours = 3.5
        elif half_life_mode == "Average (~5h)":
            half_life_hours = 5.0
        elif half_life_mode == "Slow metabolism (~7h)":
            half_life_hours = 7.0
        else:
            half_life_hours = st.slider("Custom half-life (hours)", min_value=1.5, max_value=12.0, value=5.0, step=0.5)

        st.subheader("Effect assumptions")
        profile_name = st.selectbox("Effect profile", list(EFFECT_PROFILES.keys()), index=1)
        sugar_with_coffee = st.toggle("Sugar consumed with coffee", value=False)
        horizon = st.slider("Simulation horizon (hours)", min_value=12, max_value=48, value=24, step=1)

    profile = EFFECT_PROFILES[profile_name]
    curve = build_hourly_curve(
        cups=cups,
        mg_per_cup=mg_per_cup,
        first_cup_hour=first_cup_hour,
        spacing_hours=spacing_hours,
        half_life_hours=half_life_hours,
        horizon_hours=horizon,
    )
    effects = effect_indices(curve, profile=profile, sugar_with_coffee=sugar_with_coffee)

    total_intake = cups * mg_per_cup
    peak_caffeine = float(effects["active_caffeine_mg"].max())
    hour_peak = int(effects.loc[effects["active_caffeine_mg"].idxmax(), "hour"]) if cups > 0 else 0
    evening_hour = 20 if horizon >= 20 else horizon
    evening_value = float(effects.loc[effects["hour"] == evening_hour, "active_caffeine_mg"].iloc[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total intake", f"{total_intake} mg")
    c2.metric("Peak active caffeine", f"{peak_caffeine:.1f} mg", delta=f"hour {hour_peak}")
    c3.metric("Half-life used", f"{half_life_hours:.1f} h")
    c4.metric(f"Active at hour {evening_hour}", f"{evening_value:.1f} mg")

    line1 = px.line(
        effects,
        x="hour",
        y="active_caffeine_mg",
        title="Estimated active caffeine over time",
        labels={"hour": "Hour", "active_caffeine_mg": "Active caffeine (mg)"},
    )
    line1.update_traces(line_color="#2D6CDF", line_width=4)
    st.plotly_chart(line1, use_container_width=True)

    effects_long = effects.melt(
        id_vars=["hour"],
        value_vars=["focus_index", "jitter_index", "mood_index"],
        var_name="effect",
        value_name="score",
    )
    line2 = px.line(
        effects_long,
        x="hour",
        y="score",
        color="effect",
        title="Estimated effect indices (0-100)",
        color_discrete_map={
            "focus_index": "#F7C948",
            "jitter_index": "#FF5CA8",
            "mood_index": "#2D6CDF",
        },
    )
    line2.update_layout(legend_title_text="Effect")
    st.plotly_chart(line2, use_container_width=True)

    with st.expander("Model notes and caveats"):
        st.markdown(
            """
            - This is an educational simulator, not a clinical model.
            - Caffeine decay uses a standard exponential half-life assumption.
            - Effect indices are stylized proxies built from active caffeine concentration.
            - "Sugar consumed with coffee" applies a simple amplification factor for exploration.
            - Use this tool to reason about patterns, not to make health decisions.
            """
        )


if __name__ == "__main__":
    main()
