"""
UI для расчета tр и tск по Приложению 6 Методики N 1140.
Упрощенная аналитическая модель движения людского потока.
"""

import math

import pandas as pd
import streamlit as st

from core.evacuation import (
    FLOW_PARAMS_M0,
    FLOW_PARAMS_MGN,
    F_PROJECTION,
    calc_evacuation,
    get_all_flow_groups,
)
from utils.helpers import ensure_unique_positive_int_ids, force_rerun


# Соответствие групп и площадей проекции по умолчанию
_GROUP_DEFAULT_F: dict[str, float] = {
    "М0-1": 0.06,
    "М0-2": 0.09,
    "М0-3": 0.10,
    "М0-4": 0.09,
    "М0-5": 0.121,
    "М0-6": 0.127,
    "М0-7": 0.121,
    "Пожилые (старше 60)": 0.10,
    "Пожилые немощные (М2)": 0.20,
    "Дошкольники": 0.03,
    "Дети с ОВЗ": 0.15,
    "С поражением ОДА (М3)": 0.20,
    "Слепые/слабовидящие (М2)": 0.40,
    "Глухие/слабослышащие (М1)": 0.125,
    "Беременные": 0.15,
    "Инвалиды на колясках (М4)": 0.96,
}


def render_evacuation() -> None:
    """Блок расчета tр в боковой панели."""
    st.subheader("Расчет времени эвакуации tр (Приложение 6)")
    st.caption(
        "Приложение 6 Методики N 1140: упрощенная аналитическая модель "
        "движения людского потока. Параметры движения по Приложению 2."
    )

    # 1. Группа контингента
    all_groups = list(get_all_flow_groups().keys())
    default_idx = all_groups.index("М0-3") if "М0-3" in all_groups else 0

    selected_group = st.selectbox(
        "Группа контингента",
        all_groups,
        index=default_idx,
        key="evac_flow_group",
    )

    # 2. Площадь проекции f
    default_f = _GROUP_DEFAULT_F.get(selected_group, 0.10)
    f_proj = st.number_input(
        "Площадь проекции f (м2/чел)",
        min_value=0.01,
        max_value=2.0,
        value=default_f,
        step=0.01,
        format="%.3f",
        key="evac_f_projection",
    )

    # 3. Таблица участков маршрута
    st.markdown("**Участки маршрута эвакуации**")

    if "evac_segments" not in st.session_state:
        st.session_state.evac_segments = pd.DataFrame([
            {"N": 1, "Тип": "горизонтальный", "Длина (м)": 15.0, "Ширина (м)": 6.0, "Людей": 50},
            {"N": 2, "Тип": "проем", "Длина (м)": 0.0, "Ширина (м)": 1.2, "Людей": 0},
            {"N": 3, "Тип": "горизонтальный", "Длина (м)": 20.0, "Ширина (м)": 2.0, "Людей": 0},
        ])

    path_types = ["горизонтальный", "проем", "лестница_вниз", "лестница_вверх"]
    # Для инвалидов на колясках добавляем пандусы
    if selected_group == "Инвалиды на колясках (М4)":
        path_types = ["горизонтальный", "проем", "пандус_вниз", "пандус_вверх"]

    df_seg = st.data_editor(
        st.session_state.evac_segments,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "N": st.column_config.NumberColumn("N", min_value=1, step=1, disabled=True),
            "Тип": st.column_config.SelectboxColumn("Тип участка", options=path_types),
            "Длина (м)": st.column_config.NumberColumn("Длина (м)", min_value=0.0, format="%.1f"),
            "Ширина (м)": st.column_config.NumberColumn("Ширина (м)", min_value=0.1, format="%.2f"),
            "Людей": st.column_config.NumberColumn("Людей", min_value=0, step=1),
        },
        key="evac_segments_editor",
    )

    # Обновляем номера
    df_seg = df_seg.copy()
    df_seg["N"] = range(1, len(df_seg) + 1)
    st.session_state.evac_segments = df_seg

    # 4. Кнопки
    col1, col2 = st.columns(2)
    with col1:
        add_clicked = st.button("Добавить участок", key="evac_add_segment", use_container_width=True)
    with col2:
        calc_clicked = st.button("Рассчитать tр", key="evac_calc", use_container_width=True)

    if add_clicked:
        df = st.session_state.evac_segments.copy()
        next_n = len(df) + 1
        new_row = pd.DataFrame([{
            "N": next_n,
            "Тип": "горизонтальный",
            "Длина (м)": 10.0,
            "Ширина (м)": 2.0,
            "Людей": 0,
        }])
        st.session_state.evac_segments = pd.concat([df, new_row], ignore_index=True)
        force_rerun()

    if calc_clicked:
        # Преобразуем DataFrame в список dict для calc_evacuation
        segments = []
        for _, row in df_seg.iterrows():
            typ = row.get("Тип")
            if typ is None or (isinstance(typ, float) and math.isnan(typ)):
                continue
            segments.append({
                "тип": typ,
                "длина": float(row.get("Длина (м)", 0) or 0),
                "ширина": float(row.get("Ширина (м)", 1) or 1),
                "людей": int(row.get("Людей", 0) or 0),
            })

        try:
            result = calc_evacuation(
                segments=segments,
                flow_group=selected_group,
                f_projection=f_proj,
            )
            st.session_state["evac_result"] = result
            st.session_state["evac_result_group"] = selected_group
        except Exception as e:
            st.error(f"Ошибка расчета: {e}")

    # 5. Вывод результатов
    if "evac_result" in st.session_state:
        res = st.session_state["evac_result"]

        m1, m2 = st.columns(2)
        with m1:
            st.metric("tр (мин)", f"{res['t_p']:.3f}")
        with m2:
            st.metric("tск (мин)", f"{res['t_ck']:.3f}")

        # Таблица детализации
        if res["segments_detail"]:
            df_detail = pd.DataFrame(res["segments_detail"])
            st.dataframe(
                df_detail,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "N": "N",
                    "тип": "Тип",
                    "длина": st.column_config.NumberColumn("Длина (м)", format="%.3f"),
                    "ширина": st.column_config.NumberColumn("Ширина (м)", format="%.3f"),
                    "D": st.column_config.NumberColumn("D (м2/м2)", format="%.3f"),
                    "V": st.column_config.NumberColumn("V (м/мин)", format="%.3f"),
                    "q": st.column_config.NumberColumn("q (м/мин)", format="%.3f"),
                    "t": st.column_config.NumberColumn("t (мин)", format="%.3f"),
                    "скопление": "Скопление",
                    "tз": st.column_config.NumberColumn("tз (мин)", format="%.3f"),
                },
            )

        # Предупреждения о скоплениях
        for w in res.get("warnings", []):
            st.warning(w)

        # 6. Применить к группе
        st.markdown("**Применить tр и tск к группе**")

        df_grp_apply = st.session_state.df_grp.copy()
        df_grp_apply = ensure_unique_positive_int_ids(df_grp_apply, "ID", start_from=1)

        if len(df_grp_apply) > 0:
            grp_labels = df_grp_apply.apply(
                lambda r: f"ID {int(r['ID'])} | сцен. {int(r['Сценарий i'])} | {r.get('Группа j', '')}",
                axis=1,
            ).to_list()

            target_grp = st.selectbox(
                "Применить к группе",
                grp_labels,
                key="evac_apply_group",
            )

            if st.button("Применить tр и tск к группе", key="evac_apply_btn", use_container_width=True):
                sel_idx = grp_labels.index(target_grp)
                grp_id = int(df_grp_apply.iloc[sel_idx]["ID"])
                grp_df = st.session_state.df_grp.copy()
                mask = grp_df["ID"].astype(int) == grp_id
                if mask.any():
                    idx = grp_df.index[mask][0]
                    grp_df.at[idx, "t_р,i,j (мин)"] = res["t_p"]
                    grp_df.at[idx, "t_ск,i,j (мин)"] = res["t_ck"]
                    st.session_state.df_grp = grp_df
                    st.success(
                        f"tр = {res['t_p']:.3f} мин, tск = {res['t_ck']:.3f} мин "
                        f"применены к группе ID {grp_id}."
                    )
        else:
            st.info("Нет групп. Добавьте группу в основной таблице.")
