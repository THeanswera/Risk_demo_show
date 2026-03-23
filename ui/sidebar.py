"""
Боковая панель: справочники, слайдеры, автоматический расчёт tбл.
"""

import pandas as pd
import streamlit as st

from core.constants import K_STD
from core.formulas import t_ne_formula
from core.tables import FIRE_FREQ_TABLE, T_NE_TABLE
from core.t_block import FIRE_LOAD_TABLE, calc_t_block
from utils.helpers import ensure_unique_positive_int_ids, force_rerun, safe_float


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Настройки параметров и справочники")

        # Справочник Q_п
        st.subheader("Определение частоты пожара (Qп,ᵢ)")
        st.caption("Приложение 3, Таблица П3.1")
        building_type = st.selectbox(
            "Тип здания",
            list(FIRE_FREQ_TABLE.keys()),
            index=list(FIRE_FREQ_TABLE.keys()).index("Иное (Qп = 4·10⁻²)"),
            key="building_type_select",
        )
        q_p_ref = FIRE_FREQ_TABLE[building_type]
        st.metric("Qп (год⁻¹)", f"{q_p_ref:.2e}")

        df_scen_sidebar = st.session_state.df_scen.copy()
        df_scen_sidebar["Сценарий i"] = (
            pd.to_numeric(df_scen_sidebar["Сценарий i"], errors="coerce").fillna(0).astype(int)
        )
        scen_ids_for_qp = sorted(
            df_scen_sidebar.loc[df_scen_sidebar["Сценарий i"] > 0, "Сценарий i"]
            .unique()
            .tolist()
        )
        if not scen_ids_for_qp:
            scen_ids_for_qp = [1]

        target_scen_qp = st.selectbox("Применить к сценарию №", scen_ids_for_qp, key="target_scen_qp")

        if st.button(f" Применить Qп к сценарию {target_scen_qp}", use_container_width=True):
            df = st.session_state.df_scen.copy()
            df["Сценарий i"] = pd.to_numeric(df["Сценарий i"], errors="coerce").fillna(0).astype(int)
            mask = df["Сценарий i"] == int(target_scen_qp)
            if mask.any():
                idx = df.index[mask][0]
                df.at[idx, "Q_п,i (год⁻¹)"] = q_p_ref
                df.at[idx, "Тип здания"] = building_type
                st.session_state.df_scen = df
                force_rerun()

        st.divider()

        # Справочник t_нэ 
        st.subheader("Время начала эвакуации tн.э")
        st.caption("Приложение 4 (п. 35)")

        t_ne_mode = st.radio(
            "Способ определения",
            ["По Таблице П4.1 (класс Ф)", "По формуле П4.1 (площадь помещения)"],
            key="t_ne_mode",
        )

        if t_ne_mode.startswith("По Таблице"):
            func_class = st.selectbox(
                "Класс функциональной пожарной опасности", list(T_NE_TABLE.keys())
            )
            t_ne_ref = T_NE_TABLE[func_class]
            st.metric("tн.э (мин)", f"{t_ne_ref:.1f}")
        else:
            f_pom = st.number_input(
                "Площадь помещения Fпом (м²)", min_value=1.0, value=100.0, step=10.0
            )
            t_ne_ref = t_ne_formula(f_pom)
            st.metric("tн.э (мин) = (5 + 0.01·Fпом)", f"{t_ne_ref:.3f}")
            st.caption(f"= {t_ne_ref * 60:.1f} секунд")

        df_grp_sidebar = st.session_state.df_grp.copy()
        df_grp_sidebar = ensure_unique_positive_int_ids(df_grp_sidebar, "ID", start_from=1)
        if len(df_grp_sidebar) > 0:
            grp_labels_tne = df_grp_sidebar.apply(
                lambda r: f"ID {int(r['ID'])} | сцен. {int(r['Сценарий i'])} | {r.get('Группа j', '')}",
                axis=1,
            ).to_list()

            target_grp_tne = st.selectbox(
                "Применить tн.э к группе", grp_labels_tne, key="target_grp_tne"
            )

            if st.button("Применить время начала эвакуации (tн.э) к группе", use_container_width=True):
                sel_idx_tne = grp_labels_tne.index(target_grp_tne)
                grp_id_tne = int(df_grp_sidebar.iloc[sel_idx_tne]["ID"])
                grp_df = st.session_state.df_grp.copy()
                mask = grp_df["ID"].astype(int) == grp_id_tne
                if mask.any():
                    idx = grp_df.index[mask][0]
                    grp_df.at[idx, "t_н.э,i,j (мин)"] = t_ne_ref
                    st.session_state.df_grp = grp_df
                    force_rerun()

        st.divider()

        # Слайдеры временных параметров
        st.subheader("Временные параметры")
        st.caption("Настройка для выбранной группы")

        df_grp_sel = st.session_state.df_grp.copy()
        if len(df_grp_sel) > 0:
            df_grp_sel["label"] = df_grp_sel.apply(
                lambda r: f"ID {int(r['ID'])} | сцен. {int(r['Сценарий i'])} | {r.get('Группа j', '')}",
                axis=1,
            )
            labels = df_grp_sel["label"].to_list()
            sel_label = st.selectbox(
                "Группа для настройки", labels, index=0, key="slider_group_select"
            )

            sel_id = int(df_grp_sel.loc[df_grp_sel["label"] == sel_label, "ID"].iloc[0])
            st.session_state.selected_group_id = sel_id

            grp_df = st.session_state.df_grp.copy()
            row_idx = grp_df.index[grp_df["ID"] == sel_id][0]
            scen_id = int(grp_df.at[row_idx, "Сценарий i"])

            scen_df = st.session_state.df_scen.copy()
            scen_df["Сценарий i"] = (
                pd.to_numeric(scen_df["Сценарий i"], errors="coerce").fillna(0).astype(int)
            )
            scen_match = scen_df.index[scen_df["Сценарий i"] == scen_id]

            if len(scen_match) > 0:
                scen_idx = scen_match[0]
                t_bl_cur = safe_float(scen_df.at[scen_idx, "t_бл,i (мин)"], 12.0)

                t_bl_new = st.slider("tбл,ᵢ (мин) - время блокирования", 0.5, 180.0, float(t_bl_cur), 0.5)
                t_p_new = st.slider(
                    "tр,ᵢ,ⱼ (мин) - расчётное время эвакуации",
                    0.0, 180.0,
                    float(grp_df.at[row_idx, "t_р,i,j (мин)"]),
                    0.5,
                )
                t_ne_new = st.slider(
                    "tн.э,ᵢ,ⱼ (мин) - время начала эвакуации",
                    0.0, 60.0,
                    float(grp_df.at[row_idx, "t_н.э,i,j (мин)"]),
                    0.5,
                )
                t_ck_new = st.slider(
                    "tск,ᵢ,ⱼ (мин) - время скоплений",
                    0.0, 30.0,
                    float(grp_df.at[row_idx, "t_ск,i,j (мин)"]),
                    0.5,
                )

                scen_df.at[scen_idx, "t_бл,i (мин)"] = t_bl_new
                grp_df.at[row_idx, "t_р,i,j (мин)"] = t_p_new
                grp_df.at[row_idx, "t_н.э,i,j (мин)"] = t_ne_new
                grp_df.at[row_idx, "t_ск,i,j (мин)"] = t_ck_new

                st.session_state.df_scen = scen_df
                st.session_state.df_grp = grp_df
            else:
                st.warning("Для выбранной группы не найден сценарий.")
        else:
            st.info("Нет групп. Добавьте группу в таблице.")

        st.divider()

        # Аналитический расчёт tбл (Приложение N 1, раздел IV)
        st.subheader("Аналитический расчёт времени блокирования tбл")
        st.caption("Приложение N 1, раздел IV Методики №1140 (интегральная модель, одиночное помещение H ≤ 6 м)")
        st.info(
            "Аналитические соотношения применимы для одиночного помещения высотой не более 6 м "
            "при отсутствии систем противопожарной защиты, влияющих на развитие пожара. "
            "Для более сложных случаев используйте зонные или полевые модели (FDS, PyroSim).",
        )

        room_types = FIRE_LOAD_TABLE["Вид помещения"].tolist()
        selected_room = st.selectbox("Вид помещения (Таблица П9.1)", room_types, key="tbl_room_type")
        area_m2 = st.number_input("Площадь помещения (м²)", min_value=1.0, value=50.0, step=5.0, key="tbl_area")
        height_m = st.number_input("Высота помещения (м)", min_value=1.0, value=3.0, step=0.5, key="tbl_height")
        eta_val = st.number_input(
            "Коэффициент свободного объёма η",
            min_value=0.1, max_value=1.0, value=0.8, step=0.05,
            key="tbl_eta",
            help="Обычно 0.8 - доля объёма помещения, не занятая конструкциями и оборудованием",
        )
        spread_type_label = st.selectbox(
            "Тип распространения пожара",
            ["Круговое", "Линейное"],
            key="tbl_spread_type",
            help="Круговое - для большинства помещений; линейное - для стоянок и узких длинных помещений",
        )
        spread_type = "circular" if spread_type_label == "Круговое" else "linear"
        b_width = 1.0
        if spread_type == "linear":
            b_width = st.number_input(
                "Ширина полосы горения b (м)",
                min_value=0.1, value=1.0, step=0.1,
                key="tbl_b_width",
                help="Перпендикулярный размер зоны горения при линейном распространении",
            )

        if st.button("Рассчитать tбл", use_container_width=True, key="btn_calc_tbl"):
            try:
                result = calc_t_block(selected_room, area_m2, height_m, eta_val, spread_type, b_width)
                st.session_state["tbl_result"] = result
            except Exception as e:
                st.error(f"Ошибка расчёта: {e}")

        if "tbl_result" in st.session_state:
            res = st.session_state["tbl_result"]
            st.success(f"**tбл = {res['t_bl']} мин** - {res['limiting_factor']}")
            st.caption(f"Свободный объём Vсв = {res['V_sw']} м³")

            factor_labels = {
                "t_temp": "Температура 70°С",
                "t_vis": "Видимость 20 м",
                "t_o2": "O₂ 0.226 кг/м³",
                "t_co2": "CO₂ 0.11 кг/м³",
                "t_co": "CO 1.16·10⁻³ кг/м³",
                "t_hcl": "HCl 23·10⁻⁶ кг/м³",
            }
            rows = []
            for k, lbl in factor_labels.items():
                val = res.get(k)
                rows.append({"ОФП": lbl, "tкр (мин)": val if val is not None else "-"})
            import pandas as _pd
            st.dataframe(_pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Применить к сценарию
            df_s = st.session_state.df_scen.copy()
            df_s["Сценарий i"] = (
                pd.to_numeric(df_s["Сценарий i"], errors="coerce").fillna(0).astype(int)
            )
            scen_ids_tbl = sorted(df_s.loc[df_s["Сценарий i"] > 0, "Сценарий i"].unique().tolist())
            if not scen_ids_tbl:
                scen_ids_tbl = [1]
            target_scen_tbl = st.selectbox(
                "Применить tбл к сценарию №", scen_ids_tbl, key="tbl_apply_scen"
            )
            if st.button("Применить tбл к сценарию", use_container_width=True, key="btn_apply_tbl"):
                mask = df_s["Сценарий i"] == int(target_scen_tbl)
                if mask.any() and res["t_bl"] is not None:
                    idx = df_s.index[mask][0]
                    df_s.at[idx, "t_бл,i (мин)"] = res["t_bl"]
                    st.session_state.df_scen = df_s
                    force_rerun()
