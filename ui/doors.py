"""
Шаг 3: учёт противопожарных дверей — формула (8), п. 48.
"""

import math

import streamlit as st

from core.constants import P_DOOR_CLOSED, P_DOOR_OPEN
from core.formulas import r_i_with_doors


def render_doors(r_total: float) -> float:
    
    st.subheader("Учёт противопожарных дверей — формула (8), п. 48")

    use_fire_doors = st.checkbox(
        "Учитывать противопожарные двери на путях эвакуации",
        value=st.session_state.use_fire_doors,
        key="cb_fire_doors",
        help=(
            "Если на путях эвакуации есть противопожарные двери, калитки в противопожарных воротах, "
            "расчёт выполняется для двух случаев: дверь открыта и дверь закрыта (п. 48)."
        ),
    )
    st.session_state.use_fire_doors = use_fire_doors

    r_final = r_total

    if use_fire_doors:
        st.caption(
            "По п. 48 Методики: Rᵢ = Pоткр · Rᵢ(откр) + Pзакр · Rᵢ(закр), "
            "где Pоткр = 0.3, Pзакр = 0.7"
        )

        dc1, dc2 = st.columns(2)
        with dc1:
            st.metric(
                "Rᵢ (дверь открыта), год⁻¹",
                f"{r_total:.2e}",
                help="Значение R при открытой противопожарной двери — берётся из расчёта выше.",
            )
            r_open_input = r_total
        with dc2:
            step = 10 ** (math.floor(math.log10(r_total)) - 1) if r_total > 0 else 1e-7
            r_closed_input = st.number_input(
                "Rᵢ (дверь закрыта), год⁻¹",
                min_value=0.0,
                max_value=float(r_total),
                value=0.0,
                step=step,
                format="%.2e",
                key="r_door_closed_input",
                help=(
                    "Значение R при закрытой противопожарной двери. "
                    "Опасные факторы пожара через закрытую дверь не распространяются."
                ),
            )

        r_final = r_i_with_doors(r_open_input, r_closed_input)
        st.metric(
            "R (с учётом дверей) по формуле (8)",
            f"{r_final:.2e} год⁻¹",
            help=f"= {P_DOOR_OPEN} × {r_open_input:.2e} + {P_DOOR_CLOSED} × {r_closed_input:.2e}",
        )

    return r_final
