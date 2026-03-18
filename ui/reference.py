"""
Справочные таблицы и формулы (expander-блоки).
Таблица П9.1 вынесена в core/t_block.py и используется там для расчёта tбл.
"""

import pandas as pd
import streamlit as st

from core.tables import FIRE_FREQ_TABLE, T_NE_TABLE
from utils.helpers import format_df_scientific


def render_reference() -> None:
    """Отрисовать справочный раздел с формулами и таблицами Методики."""

    # ── Формулы ─────────────────────────────────────────────
    st.subheader("📐 Формулы Методики №1140")

    with st.expander("Показать формулы (1)–(8)", expanded=False):
        st.markdown("**Условие соответствия — формула (1), п. 8:**")
        st.latex(r"R \le R_{\text{норм}}, \quad R_{\text{норм}} = 10^{-6}\ \text{год}^{-1}")

        st.markdown("**Формула (2) — расчётная величина ИПР (п. 9):**")
        st.latex(r"R = \max\{R_1, \ldots, R_i, \ldots, R_K\}")

        st.markdown("**Формула (3) — ИПР по сценарию (п. 14):**")
        st.latex(r"R_i = \max\{R_{i,1}, \ldots, R_{i,j}, \ldots, R_{i,m}\}")

        st.markdown("**Формула (4) — ИПР для группы (п. 15):**")
        st.latex(
            r"R_{i,j} = Q_{\text{п},i} \cdot (1 - K_{\text{ап},i}) \cdot "
            r"P_{\text{пр},i} \cdot (1 - P_{\text{э},i,j}) \cdot (1 - K_{\text{п.з},i})"
        )

        st.markdown("**Формула (5) — вероятность присутствия (п. 16):**")
        st.latex(r"P_{\text{пр},i} = \frac{t_{\text{пр},i}}{24}")

        st.markdown("**Формула (6) — вероятность эвакуации (п. 17):**")
        st.latex(r"""
P_{\text{э},i,j} =
\begin{cases}
0{,}999 \cdot \dfrac{0{,}8 \, t_{\text{бл},i} - t_{\text{р},i,j}}{t_{\text{н.э},i,j}},
  & t_{\text{р},i,j} < 0{,}8 \, t_{\text{бл},i} < t_{\text{р},i,j} + t_{\text{н.э},i,j},\ t_{\text{ск}} \le 6 \\[6pt]
0{,}999,
  & t_{\text{р},i,j} + t_{\text{н.э},i,j} \le 0{,}8 \, t_{\text{бл},i},\ t_{\text{ск}} \le 6 \\[6pt]
0,
  & t_{\text{р},i,j} \ge 0{,}8 \, t_{\text{бл},i}\ \text{или}\ t_{\text{ск}} > 6
\end{cases}
""")

        st.markdown("**Формула (7) — коэффициент противопожарной защиты (п. 21):**")
        st.latex(
            r"K_{\text{п.з},i} = 1 - (1 - K_{\text{обн},i} \cdot K_{\text{СОУЭ},i}) "
            r"\cdot (1 - K_{\text{обн},i} \cdot K_{\text{ПДЗ},i})"
        )

        st.markdown("**Формула (8) — учёт противопожарных дверей (п. 48):**")
        st.latex(
            r"R_i = P_{\text{откр}} \cdot R_i^{\text{откр}} + P_{\text{закр}} \cdot R_i^{\text{закр}}, "
            r"\quad P_{\text{откр}} = 0{,}3,\ P_{\text{закр}} = 0{,}7"
        )

        st.markdown("**Формула (П4.1) — время начала эвакуации для помещения очага (Приложение 4):**")
        st.latex(r"t_{\text{н.э}} = \frac{5 + 0{,}01 \cdot F_{\text{пом}}}{60}\ \text{мин}")

    # ── Справочные таблицы ───────────────────────────────────
    st.subheader("📚 Справочные таблицы из Приложений к Методике")

    with st.expander("Приложение 3 — Частота возникновения пожара (Таблица П3.1)", expanded=False):
        df_freq = pd.DataFrame([
            {"№": i + 1, "Наименование здания": k, "Qп (год⁻¹)": v}
            for i, (k, v) in enumerate(FIRE_FREQ_TABLE.items())
            if k != "Иное (Q_п = 4·10⁻²)"
        ])
        df_freq = format_df_scientific(df_freq, sci_cols=["Qп (год⁻¹)"], digits=2)
        st.dataframe(df_freq, use_container_width=True, hide_index=True)

    with st.expander("Приложение 4 — Время начала эвакуации (Таблица П4.1)", expanded=False):
        df_tne = pd.DataFrame([
            {"№": i + 1, "Класс и характеристика": k, "tн.э (мин)": v}
            for i, (k, v) in enumerate(T_NE_TABLE.items())
        ])
        st.dataframe(df_tne, use_container_width=True, hide_index=True)
