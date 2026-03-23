"""
Шаг 4: результаты расчёта, детализация, диагностика Pэ.
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.constants import R_NORM
from utils.helpers import format_df_scientific, prettify_columns, safe_float
from visualization.gauge import risk_gauge_html


def render_results(r_final: float, df_scen_calc: pd.DataFrame,
                   df_rows_calc: pd.DataFrame, df_agg: pd.DataFrame) -> None:
    st.subheader(" Результаты расчёта - формулы (1)-(2)")

    passed = r_final <= R_NORM

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("R (расчётная величина ИПР), год⁻¹", f"{r_final:.6g}")
    with m2:
        st.metric("Rнорм (нормативное значение), год⁻¹", f"{R_NORM:.1e}")
    with m3:
        if passed:
            st.success("R ≤ Rнорм: Пожарный риск ДОПУСТИМЫЙ (формула 1)")
        else:
            st.error("R > Rнорм: Пожарный риск ПРЕВЫШЕН (формула 1)")

    components.html(risk_gauge_html(r_final, R_NORM), height=420, scrolling=False)

    # Сводка по сценариям 
    st.subheader("Сводная таблица ИПР по сценариям - формулы (2) и (3)")
    st.caption("")

    df_agg_view = format_df_scientific(df_agg, sci_cols=["R_i = max_j(R_i,j)"], digits=2)
    st.dataframe(prettify_columns(df_agg_view), use_container_width=True)

    # ИПР по группам 
    st.subheader("Сводная таблица ИПР по группам - формула (4)")

    cols_show = [
        "ID", "Сценарий i", "Группа j",
        "Q_п,i (год⁻¹)", "K_ап,i",
        "t_бл,i (мин)", "t_р,i,j (мин)", "t_н.э,i,j (мин)", "t_ск,i,j (мин)",
        "P_пр,i", "P_э,i,j", "K_п.з,i", "R_i,j",
    ]
    cols_show = [c for c in cols_show if c in df_rows_calc.columns]
    df_rows_view = df_rows_calc[cols_show].copy()
    df_rows_view = format_df_scientific(df_rows_view, sci_cols=["R_i,j", "Q_п,i (год⁻¹)"], digits=2)
    st.dataframe(prettify_columns(df_rows_view), use_container_width=True)

    # Коэффициенты по сценариям 
    st.subheader("Принятые коэффциенты по сценариям - формула (7)")

    cols_scen = [
        "Сценарий i", "Тип здания", "Q_п,i (год⁻¹)", "K_ап,i",
        "K_обн,i", "K_СОУЭ,i", "K_ПДЗ,i", "K_п.з,i", "P_пр,i",
    ]
    cols_scen = [c for c in cols_scen if c in df_scen_calc.columns]
    df_scen_view = df_scen_calc[cols_scen].copy()
    df_scen_view = format_df_scientific(df_scen_view, sci_cols=["Q_п,i (год⁻¹)"], digits=2)
    st.dataframe(prettify_columns(df_scen_view), use_container_width=True)

    # Анализ P_э 
    st.subheader("Расчет Pэ,ᵢ,ⱼ - формула (6)")

    sel_id = int(st.session_state.selected_group_id)
    if len(df_rows_calc) > 0 and "ID" in df_rows_calc.columns:
        ids = df_rows_calc["ID"].astype(int).to_list()
        default_index = ids.index(sel_id) if sel_id in ids else 0
        sel_diag_id = st.selectbox("ID строки для диагностики Pэ", ids, index=default_index)

        row = df_rows_calc.loc[df_rows_calc["ID"].astype(int) == int(sel_diag_id)].iloc[0]
        t_bl = safe_float(row.get("t_бл,i (мин)", 0.0))
        t_p = safe_float(row.get("t_р,i,j (мин)", 0.0))
        t_ne = safe_float(row.get("t_н.э,i,j (мин)", 0.0))
        t_ck = safe_float(row.get("t_ск,i,j (мин)", 0.0))
        border = 0.8 * t_bl

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("0.8·tбл (мин)", f"{border:.3f}")
        c2.metric("tр (мин)", f"{t_p:.3f}")
        c3.metric("tр + tн.э (мин)", f"{(t_p + t_ne):.3f}")
        c4.metric("tск (мин)", f"{t_ck:.3f}")

        st.metric("Pэ,ᵢ,ⱼ", f"{safe_float(row.get('P_э,i,j', 0.0)):.4f}")

        # ИСПРАВЛЕНО: диагностика Pэ по п. 17 Методики №1140
        t_sum = t_p + t_ne
        if t_ck > 6:
            st.warning("tск > 6 мин → по формуле (6): Pэ = 0")
        elif t_sum < border:
            st.success("tр + tн.э < 0.8·tбл и tск ≤ 6 → Pэ = 0.999")
        elif t_sum >= t_bl:
            st.warning("tр + tн.э ≥ tбл → по формуле (6): Pэ = 0")
        else:
            st.info(
                f"0.8·tбл ≤ tр + tн.э < tбл → Pэ = 1 − (tр + tн.э)/tбл "
                f"= 1 − {t_sum:.3f}/{t_bl:.3f} = {1 - t_sum/t_bl:.4f}"
            )
    else:
        st.info("Нет строк групп для диагностики.")
