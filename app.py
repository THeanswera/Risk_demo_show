"""
Калькулятор индивидуального пожарного риска (ИПР)
по Приказу МЧС России от 14.11.2022 № 1140

Методика определения расчётных величин пожарного риска
в зданиях, сооружениях и пожарных отсеках различных классов
функциональной пожарной опасности.

Реализованы формулы (1)–(8) методики, справочные таблицы
из Приложений 3, 4, 9.

Структура проекта:
  core/       - константы, формулы, таблицы, расчёт tбл
  ui/         - Streamlit-интерфейс (шаги 1–4, справочники)
  export/     - генерация Word-отчёта
  visualization/ - HTML-компоненты (шкала риска)
  utils/      - вспомогательные функции
"""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Калькулятор ИПР по Приказу МЧС №1140",
    layout="wide",
)

# Импорты после set_page_config
from core.compute import compute_all  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
from ui.scenarios import render_scenarios  # noqa: E402
from ui.groups import render_groups  # noqa: E402
from ui.doors import render_doors  # noqa: E402
from ui.results import render_results  # noqa: E402
from ui.reference import render_reference  # noqa: E402
from export.report_docx import generate_report_docx  # noqa: E402
from utils.helpers import ensure_unique_positive_int_ids  # noqa: E402

# Данные по умолчанию

default_scenarios = pd.DataFrame([{
    "Сценарий i": 1,
    "Тип здания": "Иное (Q_п = 4·10⁻²)",
    "Q_п,i (год⁻¹)": 4.0e-2,
    "t_пр,i (ч/сут)": 12.0,
    "t_бл,i (мин)": 12.0,
    "K_ап,i": 0.9,
    "ПС соответствует? (K_обн=0.8)": True,
    "СОУЭ соответствует? (K_СОУЭ=0.8)": True,
    "ПДЗ соответствует? (K_ПДЗ=0.8)": True,
}])

default_groups = pd.DataFrame([
    {
        "ID": 1, "Сценарий i": 1, "Группа j": "Основной контингент",
        "t_р,i,j (мин)": 6.0, "t_н.э,i,j (мин)": 1.5, "t_ск,i,j (мин)": 1.0,
    },
    {
        "ID": 2, "Сценарий i": 1, "Группа j": "Маломобильные",
        "t_р,i,j (мин)": 7.0, "t_н.э,i,j (мин)": 2.0, "t_ск,i,j (мин)": 2.5,
    },
])

# ─────────────────────────────────────────────────────
# Инициализация session_state
# ─────────────────────────────────────────────────────

if "df_scen" not in st.session_state:
    st.session_state.df_scen = default_scenarios.copy()
if "df_grp" not in st.session_state:
    st.session_state.df_grp = default_groups.copy()
if "use_fire_doors" not in st.session_state:
    st.session_state.use_fire_doors = False
if "r_door_open" not in st.session_state:
    st.session_state.r_door_open = 0.0
if "r_door_closed" not in st.session_state:
    st.session_state.r_door_closed = 0.0

st.session_state.df_grp = ensure_unique_positive_int_ids(
    st.session_state.df_grp, "ID", start_from=1
)

if "selected_group_id" not in st.session_state:
    st.session_state.selected_group_id = (
        int(st.session_state.df_grp["ID"].iloc[0]) if len(st.session_state.df_grp) else 1
    )


# Заголовок
st.title("Калькулятор индивидуального пожарного риска")
st.caption(
    "По Методике определения расчётных величин пожарного риска - "
    "Приказ МЧС России от 14.11.2022 № 1140"
)

# Боковая панель

render_sidebar()

# Шаг 1: Сценарии пожара
render_scenarios()

# Шаг 2: Группы эвакуируемого контингента

render_groups()

# Расчёт

df_scen_calc, df_rows_calc, df_agg, r_total = compute_all(
    st.session_state.df_scen,
    st.session_state.df_grp,
)

# Шаг 3: Противопожарные двери

r_final = render_doors(r_total)

# Шаг 4: Результаты

render_results(r_final, df_scen_calc, df_rows_calc, df_agg)

# Справочные таблицы и формулы

render_reference()

# Выгрузка результатов

st.subheader("💾 Выгрузка результатов")

csv_rows = df_rows_calc.to_csv(index=False).encode("utf-8-sig")
csv_scen = df_scen_calc.to_csv(index=False).encode("utf-8-sig")
csv_agg = df_agg.to_csv(index=False).encode("utf-8-sig")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.download_button(
        "CSV: группы (построчно)", data=csv_rows,
        file_name="ipr_groups.csv", mime="text/csv",
    )
with c2:
    st.download_button(
        "CSV: сценарии (коэффициенты)", data=csv_scen,
        file_name="ipr_scenarios.csv", mime="text/csv",
    )
with c3:
    st.download_button(
        "CSV: агрегирование (итог)", data=csv_agg,
        file_name="ipr_aggregate.csv", mime="text/csv",
    )
with c4:
    try:
        docx_bytes = generate_report_docx(
            df_scen=st.session_state.df_scen,
            df_grp=st.session_state.df_grp,
            df_scen_calc=df_scen_calc,
            df_rows_calc=df_rows_calc,
            df_agg=df_agg,
            r_total=r_total,
            r_final=r_final,
            use_fire_doors=st.session_state.use_fire_doors,
        )
        st.download_button(
            "Скачать отчёт (Word)",
            data=docx_bytes,
            file_name="ipr_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except ImportError:
        st.warning("Для генерации Word-отчёта установите: pip install python-docx", icon="⚠️")
