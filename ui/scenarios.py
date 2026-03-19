"""
Шаг 1: таблица сценариев пожара.
"""

import numpy as np
import pandas as pd
import streamlit as st

from core.constants import K_STD
from core.tables import FIRE_FREQ_TABLE
from utils.helpers import force_rerun


def render_scenarios() -> None:
    st.subheader("Сценарии пожара (п. 9–13)")
    st.caption(
        "Определите сценарии пожара. Для каждого сценария задайте: частоту возникновения пожара Qп,ᵢ "
        "(Приложение 3), время присутствия людей tпр, время блокирования tбл, "
        "коэффициент АУП Kап (п. 15), и параметры систем ПС/СОУЭ/ПДЗ (п. 41, 44, 45)."
    )
    st.info(
        "Время блокирования tбл определяется расчётом по Приложению 6 Методики "
        "или моделированием (FDS/PyroSim). Используйте справочник в боковой панели "
        "для аналитического расчёта.",
    )

    df_scen_raw = st.session_state.df_scen.copy()
    if "Сценарий i" not in df_scen_raw.columns:
        df_scen_raw["Сценарий i"] = np.arange(1, len(df_scen_raw) + 1, dtype=int)
    df_scen_raw["Сценарий i"] = (
        pd.to_numeric(df_scen_raw["Сценарий i"], errors="coerce").fillna(0).astype(int)
    )
    df_scen_raw = df_scen_raw.loc[df_scen_raw["Сценарий i"] > 0].copy()

    scen_list = sorted(df_scen_raw["Сценарий i"].unique().tolist())
    if len(scen_list) == 0:
        scen_list = [1]

    c1, c2, c3 = st.columns([1.2, 1.8, 2.0])
    with c1:
        if st.button("Добавить сценарий", use_container_width=True):
            df = st.session_state.df_scen.copy()
            if len(df) == 0:
                next_i = 1
            else:
                df["Сценарий i"] = (
                    pd.to_numeric(df["Сценарий i"], errors="coerce").fillna(0).astype(int)
                )
                next_i = int(df["Сценарий i"].max()) + 1
            new_row = {
                "Сценарий i": next_i,
                "Тип здания": "Иное (Q_п = 4·10⁻²)",
                "Q_п,i (год⁻¹)": 4.0e-2,
                "t_пр,i (ч/сут)": 12.0,
                "t_бл,i (мин)": 12.0,
                "K_ап,i": 0.9,
                "ПС соответствует? (K_обн=0.8)": True,
                "СОУЭ соответствует? (K_СОУЭ=0.8)": True,
                "ПДЗ соответствует? (K_ПДЗ=0.8)": True,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.df_scen = df
            force_rerun()

    with c2:
        scen_del = st.selectbox("Удалить сценарий", scen_list, key="scen_del_select")
        if st.button("Удалить сценарий", use_container_width=True):
            df_s = st.session_state.df_scen.copy()
            df_s["Сценарий i"] = (
                pd.to_numeric(df_s["Сценарий i"], errors="coerce").fillna(0).astype(int)
            )
            df_s = df_s.loc[df_s["Сценарий i"] != int(scen_del)].copy()
            st.session_state.df_scen = df_s
            df_g = st.session_state.df_grp.copy()
            df_g["Сценарий i"] = (
                pd.to_numeric(df_g["Сценарий i"], errors="coerce").fillna(0).astype(int)
            )
            df_g = df_g.loc[df_g["Сценарий i"] != int(scen_del)].copy()
            st.session_state.df_grp = df_g
            force_rerun()

    with c3:
        st.info("Удаление сценария удаляет связанные группы.")

    # Preview с расчётными K
    df_scen_preview = st.session_state.df_scen.copy()
    df_scen_preview["Сценарий i"] = (
        pd.to_numeric(df_scen_preview["Сценарий i"], errors="coerce").fillna(0).astype(int)
    )
    df_scen_preview = df_scen_preview.loc[df_scen_preview["Сценарий i"] > 0].copy()

    for col_bool, col_k in [
        ("ПС соответствует? (K_обн=0.8)", "K_обн (расч.)"),
        ("СОУЭ соответствует? (K_СОУЭ=0.8)", "K_СОУЭ (расч.)"),
        ("ПДЗ соответствует? (K_ПДЗ=0.8)", "K_ПДЗ (расч.)"),
    ]:
        if col_bool in df_scen_preview.columns:
            df_scen_preview[col_k] = df_scen_preview[col_bool].astype(bool).map(
                lambda x: K_STD if x else 0.0
            )

    df_scen_edit = st.data_editor(
        df_scen_preview,
        num_rows="dynamic",
        use_container_width=True,
        disabled=["K_обн (расч.)", "K_СОУЭ (расч.)", "K_ПДЗ (расч.)"],
        column_config={
            "Сценарий i": st.column_config.NumberColumn("Сценарий i", min_value=1, step=1),
            "Тип здания": st.column_config.SelectboxColumn(
                "Тип здания", options=list(FIRE_FREQ_TABLE.keys())
            ),
            "Q_п,i (год⁻¹)": st.column_config.NumberColumn("Qп,ᵢ (год⁻¹)", format="%.4e"),
            "t_пр,i (ч/сут)": st.column_config.NumberColumn(
                "tпр,ᵢ (ч/сут)", format="%.1f", min_value=0.0, max_value=24.0
            ),
            "t_бл,i (мин)": st.column_config.NumberColumn(
                "tбл,ᵢ (мин)", format="%.2f", min_value=0.1
            ),
            "K_ап,i": st.column_config.NumberColumn(
                "Kап,ᵢ", format="%.2f", min_value=0.0, max_value=0.9
            ),
            "ПС соответствует? (K_обн=0.8)": st.column_config.CheckboxColumn("ПС (Kобн)"),
            "СОУЭ соответствует? (K_СОУЭ=0.8)": st.column_config.CheckboxColumn("СОУЭ (Kсоуэ)"),
            "ПДЗ соответствует? (K_ПДЗ=0.8)": st.column_config.CheckboxColumn("ПДЗ (Kпдз)"),
            "K_обн (расч.)": st.column_config.NumberColumn("Kобн (расч.)"),
            "K_СОУЭ (расч.)": st.column_config.NumberColumn("Kсоуэ (расч.)"),
            "K_ПДЗ (расч.)": st.column_config.NumberColumn("Kпдз (расч.)"),
        },
        key="editor_scenarios",
    )

    drop_cols = ["K_обн (расч.)", "K_СОУЭ (расч.)", "K_ПДЗ (расч.)"]
    df_scen_store = df_scen_edit.drop(
        columns=[c for c in drop_cols if c in df_scen_edit.columns], errors="ignore"
    ).copy()
    df_scen_store["Сценарий i"] = (
        pd.to_numeric(df_scen_store["Сценарий i"], errors="coerce").fillna(0).astype(int)
    )
    df_scen_store = (
        df_scen_store.loc[df_scen_store["Сценарий i"] > 0]
        .drop_duplicates(subset=["Сценарий i"], keep="first")
        .copy()
    )
    st.session_state.df_scen = df_scen_store
