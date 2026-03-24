"""
Шаг 2: таблица групп эвакуируемого контингента.
"""

import pandas as pd
import streamlit as st

from utils.helpers import ensure_unique_positive_int_ids, force_rerun


def render_groups() -> None:
    st.subheader("Группы эвакуируемого контингента (п. 14, Приложение 2)")
    st.caption(
        "Для каждого сценария задайте группы людей: "
        "tр - расчётное время эвакуации (Приложения 5-8); "
        "tн.э - время начала эвакуации (Приложение 4); "
        "tск - время существования скоплений (плотность > 0.5 м2/м2). "
        "Влияют на Pэ по формуле (6). "
        "tр и tск можно рассчитать автоматически в боковой панели "
        "(Приложение 6) или ввести вручную."
    )

    df_scen_for_groups = st.session_state.df_scen.copy()
    df_scen_for_groups["Сценарий i"] = (
        pd.to_numeric(df_scen_for_groups["Сценарий i"], errors="coerce").fillna(0).astype(int)
    )
    scen_list2 = sorted(
        df_scen_for_groups.loc[df_scen_for_groups["Сценарий i"] > 0, "Сценарий i"]
        .unique()
        .tolist()
    )
    if len(scen_list2) == 0:
        scen_list2 = [1]

    df_grp_raw = st.session_state.df_grp.copy()
    df_grp_raw = ensure_unique_positive_int_ids(df_grp_raw, "ID", start_from=1)
    df_grp_raw["Сценарий i"] = (
        pd.to_numeric(df_grp_raw["Сценарий i"], errors="coerce")
        .fillna(scen_list2[0])
        .astype(int)
    )
    st.session_state.df_grp = df_grp_raw

    g1, g2, g3 = st.columns([1.3, 1.7, 2.0])
    with g1:
        scen_for_new_group = st.selectbox(
            "Сценарий для новой группы", scen_list2, key="add_group_scen"
        )
        if st.button("Добавить группу", use_container_width=True):
            df = st.session_state.df_grp.copy()
            df = ensure_unique_positive_int_ids(df, "ID", start_from=1)
            next_id = (
                int(pd.to_numeric(df["ID"], errors="coerce").fillna(0).astype(int).max()) + 1
                if len(df)
                else 1
            )
            new_row = {
                "ID": next_id,
                "Сценарий i": int(scen_for_new_group),
                "Группа j": "Новая группа",
                "t_р,i,j (мин)": 6.0,
                "t_н.э,i,j (мин)": 1.5,
                "t_ск,i,j (мин)": 1.0,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.df_grp = df
            st.session_state.selected_group_id = int(next_id)
            force_rerun()

    with g2:
        df_tmp = st.session_state.df_grp.copy()
        df_tmp = ensure_unique_positive_int_ids(df_tmp, "ID", start_from=1)
        id_list2 = sorted(
            pd.to_numeric(df_tmp["ID"], errors="coerce").dropna().astype(int).unique().tolist()
        )
        if not id_list2:
            id_list2 = [1]
        gid_del = st.selectbox("Удалить группу по ID", id_list2, key="del_group_id")
        if st.button("Удалить группу", use_container_width=True):
            df = st.session_state.df_grp.copy()
            df["ID"] = pd.to_numeric(df["ID"], errors="coerce").fillna(0).astype(int)
            df = df.loc[df["ID"] != int(gid_del)].copy()
            df = ensure_unique_positive_int_ids(df, "ID", start_from=1)
            st.session_state.df_grp = df
            if len(df) > 0:
                st.session_state.selected_group_id = int(df["ID"].iloc[0])
            force_rerun()

    with g3:
        st.info("Группы привязаны к сценариям через поле «Сценарий i».")

    df_grp_raw2 = st.session_state.df_grp.copy()
    df_grp_raw2 = ensure_unique_positive_int_ids(df_grp_raw2, "ID", start_from=1)

    df_grp_edit = st.data_editor(
        df_grp_raw2,
        num_rows="dynamic",
        use_container_width=True,
        disabled=["ID"],
        column_config={
            "ID": st.column_config.NumberColumn("ID", min_value=1, step=1),
            "Сценарий i": st.column_config.NumberColumn("Сценарий i", min_value=1, step=1),
            "Группа j": st.column_config.TextColumn("Группа j"),
            "t_р,i,j (мин)": st.column_config.NumberColumn("tр,ᵢ,ⱼ (мин)", format="%.3f"),
            "t_н.э,i,j (мин)": st.column_config.NumberColumn("tн.э,ᵢ,ⱼ (мин)", format="%.3f"),
            "t_ск,i,j (мин)": st.column_config.NumberColumn("tск,ᵢ,ⱼ (мин)", format="%.3f"),
        },
        key="editor_groups",
    )

    df_grp_edit = ensure_unique_positive_int_ids(df_grp_edit, "ID", start_from=1)
    df_grp_edit["Сценарий i"] = (
        pd.to_numeric(df_grp_edit["Сценарий i"], errors="coerce")
        .fillna(scen_list2[0])
        .astype(int)
    )
    st.session_state.df_grp = df_grp_edit
