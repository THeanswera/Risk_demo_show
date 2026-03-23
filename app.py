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

import io

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
    "Тип здания": "Иное (Qп = 4·10⁻²)",
    "Qп,i (год⁻¹)": 4.0e-2,
    "tпр,i (ч/сут)": 12.0,
    "tбл,i (мин)": 12.0,
    "Kап,i": 0.9,
    "ПС соответствует? (Kобн=0.8)": True,
    "СОУЭ соответствует? (KСОУЭ=0.8)": True,
    "ПДЗ соответствует? (KПДЗ=0.8)": True,
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
if "r_door_closed_input" not in st.session_state:
    st.session_state.r_door_closed_input = 0.0

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
    "По приказу МЧС России от 14 ноября 2022 г. № 1140 “Об утверждении методики определения расчетных величин пожарного риска в зданиях, сооружениях и пожарных отсеках различных классов функциональной пожарной опасности”"
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

xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
    df_agg.to_excel(writer, sheet_name="Итог", index=False)
    df_rows_calc.to_excel(writer, sheet_name="Группы", index=False)
    df_scen_calc.to_excel(writer, sheet_name="Сценарии", index=False)
xlsx_bytes = xlsx_buf.getvalue()

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Скачать Excel (3 листа)", data=xlsx_bytes,
        file_name="ipr_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with c2:
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
