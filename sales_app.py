import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dynamic Data Visualization Dashboard", page_icon="📊", layout="wide")

st.title("📊 Dynamic Data Visualization Dashboard")
st.caption("Upload CSV/Excel files, explore columns, filter data, and build only the charts that fit your data.")

# -------------------- File upload --------------------
uploaded = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Upload a CSV or Excel file to begin. No fixed dataset is required.")
    st.stop()

def read_uploaded_file(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    errors = []
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    raise ValueError("Could not read CSV. " + " | ".join(errors))

try:
    raw = read_uploaded_file(uploaded)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

if raw.empty or len(raw.columns) == 0:
    st.error("The uploaded file has no rows or columns.")
    st.stop()

# Clean column names and infer types
df = raw.copy()
original_cols = list(df.columns)
def clean_col(c):
    return str(c).replace("\ufeff", "").strip()
df.columns = [clean_col(c) for c in df.columns]
# Keep duplicate column names distinct
seen = {}
new_cols = []
for c in df.columns:
    seen[c] = seen.get(c, 0) + 1
    new_cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
df.columns = new_cols

# Infer date-like columns without forcing all strings into dates
for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
        non_null = df[col].dropna()
        if len(non_null):
            sample = non_null.astype(str).head(100)
            looks_date = (
                any(token in col.lower() for token in ("date", "time", "timestamp"))
                or (sample.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", regex=True).mean() > 0.75)
            )
            if looks_date:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() >= 0.7:
                    df[col] = parsed

# Convert numeric-looking text columns conservatively
for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
        series = df[col].astype("string").str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("%", "", regex=False).str.strip()
        converted = pd.to_numeric(series, errors="coerce")
        non_null_count = df[col].notna().sum()
        if non_null_count and converted.notna().sum() / non_null_count >= 0.9:
            df[col] = converted

date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
categorical_cols = [c for c in df.columns if c not in date_cols and c not in numeric_cols]

# Sidebar data filters
st.sidebar.header("🔎 Data Filters")
filtered = df.copy()
with st.sidebar:
    st.caption("Filters apply to all charts and summary metrics.")
    for col in categorical_cols:
        vals = df[col].dropna().unique().tolist()
        if 1 < len(vals) <= 100:
            vals_sorted = sorted(vals, key=lambda x: str(x))
            chosen = st.multiselect(f"{col}", vals_sorted, default=vals_sorted, key=f"filter_{col}")
            filtered = filtered[filtered[col].isin(chosen)]
    for col in date_cols:
        valid = pd.to_datetime(df[col], errors="coerce").dropna()
        if not valid.empty:
            lo, hi = valid.min().date(), valid.max().date()
            date_range = st.date_input(f"{col} range", value=(lo, hi), min_value=lo, max_value=hi, key=f"date_{col}")
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                filtered = filtered[(filtered[col] >= start) & (filtered[col] <= end)]
    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) and vals.nunique() > 10:
            lo, hi = float(vals.min()), float(vals.max())
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                selected = st.slider(f"{col} range", min_value=lo, max_value=hi, value=(lo, hi), key=f"num_{col}")
                filtered = filtered[pd.to_numeric(filtered[col], errors="coerce").between(selected[0], selected[1])]

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows after filters", f"{len(filtered):,}")
m2.metric("Columns", f"{len(filtered.columns):,}")
m3.metric("Numeric columns", f"{len(numeric_cols):,}")
m4.metric("Missing cells", f"{int(filtered.isna().sum().sum()):,}")

# -------------------- Missing values analysis --------------------
st.divider()
st.header("🧹 Missing Values Analysis")

original_missing_total = int(df.isna().sum().sum())
filtered_missing_total = int(filtered.isna().sum().sum())
original_cells = int(df.shape[0] * df.shape[1])
filtered_cells = int(filtered.shape[0] * filtered.shape[1])
original_missing_pct = (original_missing_total / original_cells * 100) if original_cells else 0.0
filtered_missing_pct = (filtered_missing_total / filtered_cells * 100) if filtered_cells else 0.0

mv1, mv2, mv3, mv4 = st.columns(4)
mv1.metric("Missing cells (original data)", f"{original_missing_total:,}")
mv2.metric("Original missing rate", f"{original_missing_pct:.2f}%")
mv3.metric("Missing cells (after filters)", f"{filtered_missing_total:,}")
mv4.metric("Filtered missing rate", f"{filtered_missing_pct:.2f}%")

missing_by_column = pd.DataFrame({
    "Column": df.columns,
    "Missing values": [int(df[c].isna().sum()) for c in df.columns],
    "Missing %": [float(df[c].isna().mean() * 100) if len(df) else 0.0 for c in df.columns],
    "Non-missing values": [int(df[c].notna().sum()) for c in df.columns],
})
missing_by_column = missing_by_column.sort_values(
    ["Missing values", "Missing %"], ascending=False
).reset_index(drop=True)

if original_missing_total == 0:
    st.success("No missing values were detected in the uploaded dataset.")
else:
    st.caption("This section reports missing values; it does not automatically fill or delete them.")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Missing values by column")
        st.dataframe(missing_by_column, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Missing values chart")
        chart_data = missing_by_column[missing_by_column["Missing values"] > 0].sort_values("Missing values")
        if not chart_data.empty:
            fig_missing = px.bar(
                chart_data,
                x="Missing values",
                y="Column",
                orientation="h",
                hover_data={"Missing %": ":.2f", "Non-missing values": True},
                title="Missing cells per column",
                text="Missing values",
            )
            fig_missing.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_missing, use_container_width=True, key="missing_values_by_column")

    with st.expander("Show columns with missing values only"):
        st.dataframe(
            missing_by_column[missing_by_column["Missing values"] > 0],
            use_container_width=True,
            hide_index=True,
        )

with st.expander("Dataset profile", expanded=False):
    st.write("**Detected column types**")
    profile = pd.DataFrame({
        "Column": df.columns,
        "Detected type": [
            "Date/time" if c in date_cols else "Numeric" if c in numeric_cols else "Categorical/text"
            for c in df.columns
        ],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(profile, use_container_width=True)
    st.write("**Preview**")
    st.dataframe(filtered.head(100), use_container_width=True)

if filtered.empty:
    st.warning("No rows match the current filters. Widen or clear some filters.")
    st.stop()

# -------------------- Dynamic KPI cards --------------------
# KPIs are inferred from the uploaded dataset and always use the filtered rows.
def _kpi_is_identifier(column):
    name = str(column).strip().lower().replace(" ", "_")
    return (
        name in {"id", "product_id", "customer_id", "order_id", "transaction_id", "invoice_id", "employee_id", "sales_rep_id"}
        or name.endswith(("_id", "_code", "_key", "_no", "_number"))
        or name.startswith(("id_", "code_"))
    )

def _kpi_name_match(columns, keywords):
    for column in columns:
        label = str(column).lower().replace("_", " ").replace("-", " ")
        if any(word in label for word in keywords):
            return column
    return None

_kpi_numeric = [c for c in numeric_cols if not _kpi_is_identifier(c)]
_kpi_measure = _kpi_name_match(
    _kpi_numeric,
    ["sales", "revenue", "turnover", "profit", "amount", "total", "price", "value", "quantity", "units", "cost", "salary", "score", "fare", "charges"]
)
if _kpi_measure is None and _kpi_numeric:
    _kpi_measure = _kpi_numeric[0]

_kpi_entity = _kpi_name_match(
    list(filtered.columns),
    ["customer id", "customer_id", "order id", "order_id", "transaction id", "transaction_id", "product id", "product_id", "employee id", "employee_id"]
)
if _kpi_entity is None:
    _kpi_entity = _kpi_name_match(
        [c for c in categorical_cols if not _kpi_is_identifier(c)],
        ["customer", "order", "transaction", "product", "employee", "category", "region", "department", "city", "country"]
    )
if _kpi_entity is None:
    _kpi_entity = next((c for c in categorical_cols if not _kpi_is_identifier(c)), None)

st.divider()
st.header("📌 Key Performance Indicators")
st.caption("These summary cards update automatically when you change the sidebar filters.")
_kpi_cards = st.columns(4)
_kpi_cards[0].metric("Rows after filters", f"{len(filtered):,}")
if _kpi_measure is not None:
    _kpi_values = pd.to_numeric(filtered[_kpi_measure], errors="coerce").dropna()
    _kpi_label = str(_kpi_measure)
    _kpi_total_label = "Total " + _kpi_label if any(w in _kpi_label.lower() for w in ("sales", "revenue", "amount", "profit", "cost", "price", "value")) else "Sum of " + _kpi_label
    _kpi_cards[1].metric(_kpi_total_label, f"{_kpi_values.sum():,.2f}" if len(_kpi_values) else "N/A")
    _kpi_cards[2].metric("Average " + _kpi_label, f"{_kpi_values.mean():,.2f}" if len(_kpi_values) else "N/A")
else:
    _kpi_cards[1].metric("Columns", f"{len(filtered.columns):,}")
    _kpi_cards[2].metric("Missing cells", f"{int(filtered.isna().sum().sum()):,}")
if _kpi_entity is not None:
    _kpi_cards[3].metric("Unique " + str(_kpi_entity), f"{filtered[_kpi_entity].nunique(dropna=True):,}")
else:
    _kpi_cards[3].metric("Missing cells", f"{int(filtered.isna().sum().sum()):,}")

st.divider()
st.header("📈 Automatic chart suggestions")

# Suggest charts using column meaning, not just column order.
def is_identifier(col):
    name = str(col).strip().lower().replace(" ", "_")
    tokens = ("id", "code", "key", "number", "no")
    return (
        name in {"id", "product_id", "customer_id", "order_id", "transaction_id", "invoice_id", "employee_id", "sales_rep_id"}
        or name.endswith(("_id", "_code", "_key", "_no", "_number"))
        or name.startswith(("id_", "code_"))
    )

def name_match(columns, keywords):
    for col in columns:
        low = str(col).lower().replace("_", " ")
        if any(word in low for word in keywords):
            return col
    return None

measure_cols = [c for c in numeric_cols if not is_identifier(c)]
category_cols = [c for c in categorical_cols if not is_identifier(c)]
id_cols = [c for c in df.columns if is_identifier(c)]

sales_measure = name_match(measure_cols, ["sales amount", "sales", "revenue", "turnover", "amount", "total"])
profit_measure = name_match(measure_cols, ["profit", "margin", "earnings"])
date_measure = name_match(date_cols, ["date", "month", "time", "year"])
product_id = name_match(id_cols, ["product id", "product_id", "product code", "product no"])
product_category = name_match(category_cols, ["product category", "category", "subcategory", "product type"])
region_col = name_match(category_cols, ["region", "territory", "area", "location"])
rep_col = name_match(category_cols, ["sales rep", "sales representative", "salesperson", "employee"])
suggestions = []

# Specific, meaningful product analysis when the relevant fields exist.
if product_id and sales_measure:
    suggestions.append(("Bar", product_id, sales_measure, None))
# General charts use meaningful categories/measures, avoiding ID-like columns.
primary_cat = product_category or region_col or rep_col or (category_cols[0] if category_cols else None)
primary_num = sales_measure or profit_measure or (measure_cols[0] if measure_cols else None)

if primary_cat and primary_num:
    if primary_cat != product_id:
        suggestions.append(("Bar", primary_cat, primary_num, None))
    if product_category and sales_measure:
        suggestions.append(("Treemap", product_category, sales_measure,
                           region_col if region_col and region_col != product_category else None))
    elif len(category_cols) >= 2:
        tree_parent = product_category or category_cols[0]
        tree_child = region_col or next((c for c in category_cols if c != tree_parent), None)
        if tree_child and tree_parent != tree_child:
            suggestions.append(("Treemap", tree_parent, primary_num, tree_child))
    if df[primary_cat].nunique(dropna=True) <= 8:
        suggestions.append(("Donut", primary_cat, primary_num, None))

if date_cols and primary_num:
    suggestions.append(("Line", date_measure or date_cols[0], primary_num, None))
if len(measure_cols) >= 2:
    suggestions.append(("Scatter", measure_cols[0], measure_cols[1], None))
if measure_cols:
    suggestions.append(("Histogram", measure_cols[0], None, None))
    suggestions.append(("Box", None, measure_cols[0], primary_cat))
if len(measure_cols) >= 3:
    suggestions.append(("Bubble", measure_cols[0], measure_cols[1], measure_cols[2]))
if primary_cat and primary_num:
    suggestions.append(("Horizontal bar", primary_cat, primary_num, None))

# De-duplicate and render at most 4 useful automatic charts.
# Each chart has dashboard controls so users can change columns without editing app.py.
seen_suggestions = set()
unique_suggestions = []
for item in suggestions:
    if item not in seen_suggestions:
        unique_suggestions.append(item)
        seen_suggestions.add(item)

auto_chart_types = [
    "Bar", "Horizontal bar", "Line", "Area", "Pie", "Donut",
    "Treemap", "Scatter", "Bubble", "Histogram", "Box plot", "Violin plot"
]
numeric_options = list(numeric_cols)
all_column_options = list(filtered.columns)

for idx, (suggested_kind, suggested_x, suggested_y, suggested_color) in enumerate(unique_suggestions[:4]):
    st.markdown("---")
    st.subheader(
        f"Suggested chart {idx + 1}: "
        f"{'Top 10 Products by ' + str(suggested_y) if suggested_x == product_id and suggested_y == sales_measure else str(suggested_y or 'Rows') + ' by ' + str(suggested_x or 'Distribution')}"
    )

    default_x = suggested_x if suggested_x in all_column_options else (
        category_cols[0] if category_cols else all_column_options[0]
    )
    default_y = suggested_y if suggested_y in numeric_options else (
        sales_measure if sales_measure in numeric_options else (numeric_options[0] if numeric_options else None)
    )
    default_kind = suggested_kind if suggested_kind in auto_chart_types else "Bar"
    default_group = suggested_color if suggested_color in all_column_options else "None"

    c1, c2, c3 = st.columns(3)
    with c1:
        chart_kind = st.selectbox(
            "Chart type", auto_chart_types, index=auto_chart_types.index(default_kind),
            key=f"auto_kind_{idx}"
        )
        x_choice = st.selectbox(
            "X-axis / category", all_column_options, index=all_column_options.index(default_x),
            key=f"auto_x_{idx}"
        )
    with c2:
        y_options = ["(row count)"] + numeric_options
        default_y_choice = default_y if default_y in numeric_options else "(row count)"
        y_choice = st.selectbox(
            "Y-axis / value", y_options, index=y_options.index(default_y_choice),
            key=f"auto_y_{idx}"
        )
        group_options = ["None"] + all_column_options
        group_choice = st.selectbox(
            "Group / color (optional)", group_options, index=group_options.index(default_group),
            key=f"auto_group_{idx}"
        )
    with c3:
        agg_choice = st.selectbox(
            "Aggregation", ["Sum", "Mean", "Median", "Count"],
            key=f"auto_agg_{idx}"
        )
        top_n = st.selectbox("Categories to show", [5, 10, 20, 30, 50], index=1, key=f"auto_topn_{idx}")

    try:
        plot_df = filtered.copy()
        y_choice = None if y_choice == "(row count)" else y_choice
        group_choice = None if group_choice == "None" else group_choice

        if chart_kind in ("Bar", "Horizontal bar", "Pie", "Donut", "Treemap"):
            if y_choice is None:
                plot_df["_row_count_auto_"] = 1
                value_col = "_row_count_auto_"
                method = "sum"
            else:
                value_col = y_choice
                method = {"Sum": "sum", "Mean": "mean", "Median": "median", "Count": "count"}[agg_choice]

            if agg_choice == "Count":
                grouped = plot_df.groupby(x_choice, dropna=False).size().reset_index(name="Row count")
                value_col = "Row count"
            elif group_choice and chart_kind in ("Bar", "Horizontal bar") and group_choice != x_choice:
                grouped = plot_df.groupby([x_choice, group_choice], dropna=False)[value_col].agg(method).reset_index()
            else:
                grouped = plot_df.groupby(x_choice, dropna=False)[value_col].agg(method).reset_index()

            grouped = grouped.sort_values(value_col, ascending=False).head(top_n)
            # ID values are labels, not a continuous numeric scale.
            if is_identifier(x_choice):
                grouped[x_choice] = grouped[x_choice].astype(str)
                plot_df[x_choice] = plot_df[x_choice].astype(str)

            if chart_kind == "Bar":
                fig = px.bar(grouped, x=x_choice, y=value_col,
                             color=group_choice if group_choice in grouped.columns else None,
                             title=f"{agg_choice} {value_col} by {x_choice}",
                             category_orders={x_choice: grouped[x_choice].astype(str).tolist()})
            elif chart_kind == "Horizontal bar":
                fig = px.bar(grouped, x=value_col, y=x_choice, orientation="h",
                             color=group_choice if group_choice in grouped.columns else None,
                             title=f"{agg_choice} {value_col} by {x_choice}")
            elif chart_kind in ("Pie", "Donut"):
                if group_choice and group_choice != x_choice:
                    st.info("Pie/Donut uses the X-axis category only; the group selection is ignored.")
                fig = px.pie(grouped, names=x_choice, values=value_col,
                             hole=0.45 if chart_kind == "Donut" else 0,
                             title=f"{chart_kind}: {value_col} by {x_choice}")
            else:
                if is_identifier(x_choice) or (group_choice and is_identifier(group_choice)):
                    st.warning("Treemap works best with category columns, not ID/code columns. Choose Product_Category, Region, or another descriptive category.")
                    continue
                if y_choice is None:
                    st.warning("Treemap needs a numeric value column. Select one for Y-axis.")
                    continue
                if group_choice and group_choice != x_choice:
                    tree = plot_df.groupby([x_choice, group_choice], dropna=False)[value_col].sum().reset_index()
                    fig = px.treemap(tree, path=[x_choice, group_choice], values=value_col,
                                     title=f"Treemap: {value_col} by {x_choice} and {group_choice}")
                else:
                    fig = px.treemap(grouped, path=[x_choice], values=value_col,
                                     title=f"Treemap: {value_col} by {x_choice}")

        elif chart_kind in ("Line", "Area"):
            if y_choice is None:
                st.warning(f"{chart_kind} chart requires a numeric Y-axis column.")
                continue
            grouped = plot_df.groupby(x_choice, dropna=False)[y_choice].agg(
                {"Sum": "sum", "Mean": "mean", "Median": "median", "Count": "count"}[agg_choice]
            ).reset_index().sort_values(x_choice)
            if chart_kind == "Line":
                fig = px.line(grouped, x=x_choice, y=y_choice, markers=True,
                              title=f"{agg_choice} {y_choice} over {x_choice}")
            else:
                fig = px.area(grouped, x=x_choice, y=y_choice,
                              title=f"{agg_choice} {y_choice} over {x_choice}")

        elif chart_kind == "Scatter":
            if x_choice not in numeric_options or y_choice not in numeric_options:
                st.warning("Scatter requires numeric columns on both X and Y axes.")
                continue
            fig = px.scatter(plot_df, x=x_choice, y=y_choice, color=group_choice,
                             title=f"{y_choice} vs {x_choice}", opacity=0.75)

        elif chart_kind == "Bubble":
            if x_choice not in numeric_options or y_choice not in numeric_options:
                st.warning("Bubble requires numeric columns on both X and Y axes.")
                continue
            size_choice = st.selectbox(
                "Bubble size column", numeric_options, key=f"auto_size_{idx}"
            )
            fig = px.scatter(plot_df, x=x_choice, y=y_choice, size=size_choice,
                             color=group_choice, title=f"{y_choice} vs {x_choice} (size: {size_choice})")

        elif chart_kind == "Histogram":
            target = x_choice if x_choice in numeric_options else y_choice
            if target is None or target not in numeric_options:
                st.warning("Histogram requires a numeric column. Select a numeric X-axis or Y-axis.")
                continue
            fig = px.histogram(plot_df, x=target, color=group_choice, title=f"Distribution of {target}")

        else:  # Box plot or Violin plot
            target = y_choice if y_choice in numeric_options else (x_choice if x_choice in numeric_options else None)
            if target is None:
                st.warning(f"{chart_kind} requires a numeric column.")
                continue
            if chart_kind == "Box plot":
                fig = px.box(plot_df, x=group_choice, y=target, points="outliers",
                             title=f"Box plot: {target}" + (f" by {group_choice}" if group_choice else ""))
            else:
                fig = px.violin(plot_df, x=group_choice, y=target, box=True, points="outliers",
                                title=f"Violin plot: {target}" + (f" by {group_choice}" if group_choice else ""))

        st.plotly_chart(fig, use_container_width=True, key=f"auto_{idx}")
    except Exception as exc:
        st.warning(f"Could not render this chart with the selected columns: {exc}")

st.divider()
st.header("🛠️ Custom Plot Builder")
st.caption("18 plot types are supported. Only chart types compatible with your selected columns are offered.")

chart_types = [
    "Bar", "Horizontal bar", "Grouped bar", "Stacked bar",
    "Line", "Area", "Pie", "Donut", "Scatter", "Bubble",
    "Histogram", "Box plot", "Violin plot", "Heatmap",
    "Treemap", "Density contour", "Funnel", "Waterfall"
]

with st.form("plot_builder"):
    chart_type = st.selectbox("Plot type", chart_types)
    all_cols = list(filtered.columns)
    default_x = categorical_cols[0] if categorical_cols else (date_cols[0] if date_cols else all_cols[0])
    x_options = all_cols
    x_col = st.selectbox("X / Category column", x_options, index=x_options.index(default_x))
    y_options = numeric_cols
    y_col = st.selectbox("Y / Value column (if needed)", ["(row count)"] + y_options)
    color_options = ["None"] + all_cols
    color_col = st.selectbox("Color / Group column (optional)", color_options)
    size_col = st.selectbox("Bubble size column (numeric)", ["None"] + numeric_cols)
    agg = st.selectbox("Aggregation for category charts", ["Sum", "Mean", "Median", "Count"])
    bins = st.slider("Histogram bins", min_value=5, max_value=80, value=25)
    show_labels = st.checkbox("Show labels where supported", value=False)
    submitted = st.form_submit_button("Build plot")

if submitted:
    try:
        d = filtered.copy()
        y = None if y_col == "(row count)" else y_col
        color = None if color_col == "None" else color_col
        if chart_type in ("Bar", "Horizontal bar", "Grouped bar", "Stacked bar", "Pie", "Donut", "Treemap", "Funnel", "Waterfall"):
            if y is None and chart_type not in ("Funnel", "Waterfall"):
                d["_row_count_"] = 1
                y = "_row_count_"
            if chart_type in ("Pie", "Donut") and d[x_col].nunique(dropna=True) > 25:
                st.warning("Pie/Donut works best with a small number of categories (25 or fewer).")
            if agg == "Count":
                grouped = d.groupby(x_col, dropna=False).size().reset_index(name="Count")
                val = "Count"
            else:
                if y is None:
                    d["_row_count_"] = 1
                    y = "_row_count_"
                method = {"Sum": "sum", "Mean": "mean", "Median": "median"}[agg]
                grouped = d.groupby(x_col, dropna=False)[y].agg(method).reset_index()
                val = y
            grouped = grouped.sort_values(val, ascending=False).head(100)
            if chart_type == "Bar":
                fig = px.bar(grouped, x=x_col, y=val, color=color if color in grouped.columns else None, title=f"{agg} by {x_col}", text_auto=show_labels)
            elif chart_type == "Horizontal bar":
                fig = px.bar(grouped, x=val, y=x_col, orientation="h", title=f"{agg} by {x_col}", text_auto=show_labels)
            elif chart_type in ("Grouped bar", "Stacked bar"):
                if color is None or color == x_col:
                    st.warning("Choose a different Color / Group column for grouped or stacked bars.")
                    st.stop()
                g = d.groupby([x_col, color], dropna=False).size().reset_index(name="Count") if agg == "Count" else d.groupby([x_col, color], dropna=False)[y].agg({"Sum":"sum","Mean":"mean","Median":"median"}[agg]).reset_index()
                val2 = "Count" if agg == "Count" else y
                fig = px.bar(g, x=x_col, y=val2, color=color, barmode="group" if chart_type == "Grouped bar" else "stack", title=f"{chart_type}: {val2} by {x_col}")
            elif chart_type in ("Pie", "Donut"):
                fig = px.pie(grouped, names=x_col, values=val, hole=0.45 if chart_type == "Donut" else 0, title=f"{chart_type}: {val} by {x_col}")
            elif chart_type == "Treemap":
                if is_identifier(x_col) or (color and color != x_col and is_identifier(color)):
                    st.error("Treemap hierarchy should use meaningful categories, not identifier columns such as Product_ID. Try Product_Category with Sales_Amount instead.")
                    st.stop()
                if y is None:
                    st.error("Treemap requires a numeric value column, such as Sales_Amount.")
                    st.stop()
                if color and color != x_col:
                    grouped_tree = d.groupby([x_col, color], dropna=False)[y].sum().reset_index()
                    fig = px.treemap(grouped_tree, path=[x_col, color], values=y, title=f"Treemap: {y} by categories")
                else:
                    fig = px.treemap(grouped, path=[x_col], values=val, title=f"Treemap: {val} by {x_col}")
            elif chart_type == "Funnel":
                fig = px.funnel(grouped, x=val, y=x_col, title=f"Funnel: {val} by {x_col}")
            else:  # Waterfall
                vals = grouped[val].fillna(0).tolist()
                labels = grouped[x_col].astype(str).tolist()
                fig = go.Figure(go.Waterfall(x=labels, y=vals, measure=["relative"] * len(vals)))
                fig.update_layout(title=f"Waterfall: {val} by {x_col}")
        elif chart_type == "Line":
            if y is None:
                st.error("Line chart requires a numeric Y column.")
                st.stop()
            g = d.groupby(x_col, dropna=False)[y].sum().reset_index().sort_values(x_col)
            fig = px.line(g, x=x_col, y=y, color=color, markers=True, title=f"{y} over {x_col}")
        elif chart_type == "Area":
            if y is None:
                st.error("Area chart requires a numeric Y column.")
                st.stop()
            g = d.groupby(x_col, dropna=False)[y].sum().reset_index().sort_values(x_col)
            fig = px.area(g, x=x_col, y=y, color=color, title=f"{y} over {x_col}")
        elif chart_type == "Scatter":
            if y is None or x_col not in numeric_cols or y not in numeric_cols:
                st.error("Scatter plot requires numeric columns on both X and Y axes.")
                st.stop()
            fig = px.scatter(d, x=x_col, y=y, color=color, title=f"{y} vs {x_col}")
        elif chart_type == "Bubble":
            if y is None or x_col not in numeric_cols or y not in numeric_cols or size_col == "None":
                st.error("Bubble plot requires numeric X, numeric Y, and a numeric size column.")
                st.stop()
            fig = px.scatter(d, x=x_col, y=y, size=size_col, color=color, title=f"Bubble: {y} vs {x_col}")
        elif chart_type == "Histogram":
            target = x_col if x_col in numeric_cols else y
            if target is None or target not in numeric_cols:
                st.error("Histogram requires a numeric column. Select one for X or Y.")
                st.stop()
            fig = px.histogram(d, x=target, nbins=bins, color=color, title=f"Distribution of {target}")
        elif chart_type in ("Box plot", "Violin plot"):
            target = y if y in numeric_cols else (x_col if x_col in numeric_cols else None)
            if target is None:
                st.error("Box/Violin plot requires a numeric column.")
                st.stop()
            if chart_type == "Box plot":
                fig = px.box(d, x=color, y=target, points="outliers", title=f"Box plot: {target}")
            else:
                fig = px.violin(d, x=color, y=target, box=True, points="outliers", title=f"Violin plot: {target}")
        elif chart_type == "Heatmap":
            nums = numeric_cols
            if len(nums) < 2:
                st.error("Heatmap requires at least two numeric columns.")
                st.stop()
            corr = d[nums].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=show_labels, aspect="auto", title="Numeric correlation heatmap")
        elif chart_type == "Density contour":
            if y is None or x_col not in numeric_cols or y not in numeric_cols:
                st.error("Density contour requires numeric X and Y columns.")
                st.stop()
            fig = px.density_contour(d, x=x_col, y=y, title=f"Density contour: {x_col} vs {y}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not build this plot with the selected columns: {exc}")

st.divider()
st.subheader("📥 Export filtered data")
st.download_button(
    "Download filtered dataset as CSV",
    data=filtered.to_csv(index=False).encode("utf-8-sig"),
    file_name="filtered_data.csv",
    mime="text/csv"
)
