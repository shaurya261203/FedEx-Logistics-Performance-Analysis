"""
FedEx Logistics Performance Analysis - Streamlit Dashboard
=============================================================
Production-ready conversion of the original Google Colab EDA notebook
("Shaurya_FedEX_capstone.ipynb") into an interactive Streamlit application.

Author: Shaurya Nitin (original analysis) / Converted to Streamlit
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# ============================================================
# SECTION 2: PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="FedEx Logistics Performance Analysis",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent seaborn / matplotlib styling for all charts
sns.set_theme(style="whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


# ============================================================
# SECTION 3: DATA LOADING (cached)
# ============================================================
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    """
    Load the SCMS Delivery History Dataset from the project directory.
    (Replaces the original Google Colab drive.mount() + /content/drive path.)
    """
    df = pd.read_csv("SCMS_Delivery_History_Dataset.csv")
    return df


# ============================================================
# SECTION 4: DATA CLEANING / WRANGLING (cached)
# ============================================================
@st.cache_data(show_spinner="Cleaning and preparing data...")
def clean_data(df_raw):
    """
    Recreates every data-wrangling step performed in the original notebook:
      - Handle missing values (ffill / bfill / mean-fill)
      - Convert date columns to datetime
      - Derive helper numeric / year columns
      - Compute delivery delay
    Returns the fully cleaned dataframe.
    """
    df = df_raw.copy()

    # --- Null value counts (used for the "missing values" chart) ---
    null_counts_before = df.isnull().sum()
    null_pct_before = df.isnull().mean() * 100

    # --- Handle missing values exactly as in the notebook ---
    # Shipment Mode: forward fill
    df["Shipment Mode"] = df["Shipment Mode"].ffill()
    # Dosage: backward fill (first row was null)
    df["Dosage"] = df["Dosage"].bfill()
    # Line Item Insurance (USD): fill numeric column with mean
    df["Line Item Insurance (USD)"] = df["Line Item Insurance (USD)"].fillna(
        df["Line Item Insurance (USD)"].mean()
    )

    # --- Convert date columns to datetime (format dd-Mon-yy) ---
    date_columns = [
        "Scheduled Delivery Date",
        "Delivered to Client Date",
        "Delivery Recorded Date",
    ]
    df[date_columns] = df[date_columns].apply(
        lambda x: pd.to_datetime(x, format="%d-%b-%y", errors="coerce")
    )

    # --- Derive year columns ---
    df["year_recorded"] = df["Delivery Recorded Date"].dt.year
    df["year"] = df["Delivered to Client Date"].dt.year

    # --- Numeric coercion for columns that contain non-numeric text values ---
    df["Line Item Value"] = pd.to_numeric(df["Line Item Value"], errors="coerce")
    df["Freight Cost (USD)"] = pd.to_numeric(df["Freight Cost (USD)"], errors="coerce")
    df["Weight (Kilograms)"] = pd.to_numeric(df["Weight (Kilograms)"], errors="coerce")

    # --- Derived value columns (millions) ---
    df["Value_Millions"] = df["Line Item Value"] / 1_000_000
    df["Item Value in Million"] = df["Line Item Value"] / 1_000_000

    # --- Delivery delay (days) ---
    df["delivery delay"] = (
        df["Delivered to Client Date"] - df["Scheduled Delivery Date"]
    ).dt.days

    return df, null_counts_before, null_pct_before


@st.cache_data(show_spinner=False)
def remove_delay_outliers(df):
    """
    IQR-based outlier removal on 'delivery delay', exactly as done in the
    notebook before plotting the delivery delay box plot.
    """
    Q1 = df["delivery delay"].quantile(0.25)
    Q3 = df["delivery delay"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df[(df["delivery delay"] >= lower) & (df["delivery delay"] <= upper)]
    return df_clean


# ============================================================
# SECTION 5: KPI CALCULATION
# ============================================================
def compute_kpis(df):
    """Compute the four headline KPI metrics for the dashboard."""
    total_shipments = len(df)
    total_revenue = df["Line Item Value"].sum()
    avg_freight_cost = df["Freight Cost (USD)"].mean()
    avg_delay = df["delivery delay"].mean()
    return total_shipments, total_revenue, avg_freight_cost, avg_delay


def render_kpi_cards(df):
    """Render the KPI cards using st.columns()."""
    total_shipments, total_revenue, avg_freight_cost, avg_delay = compute_kpis(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Shipments", f"{total_shipments:,}")
    with col2:
        st.metric("💰 Total Revenue", f"${total_revenue/1_000_000:,.2f} M")
    with col3:
        st.metric("🚚 Avg Freight Cost", f"${avg_freight_cost:,.2f}")
    with col4:
        delay_label = "On schedule" if avg_delay <= 0 else "Avg delay"
        st.metric(f"⏱ {delay_label}", f"{avg_delay:,.1f} days")


# ============================================================
# SECTION 6: SIDEBAR - FILTERS & NAVIGATION
# ============================================================
def render_sidebar(df):
    """Render sidebar navigation and interactive filters, return selections."""
    st.sidebar.title("📦 FedEx Analytics")
    st.sidebar.markdown("---")

    section = st.sidebar.radio(
        "Navigate to",
        [
            "🏠 Overview & Objective",
            "🧹 Data Cleaning & Quality",
            "📊 KPI Dashboard",
            "🚚 Shipment Mode Analysis",
            "📈 Revenue Trends",
            "⏱ Delivery Performance",
            "💰 Cost Analysis",
            "🌍 Geographic Insights",
            "🏭 Vendor & Product Insights",
            "🔗 Correlation Analysis",
            "💡 Business Insights",
            "✅ Recommendations",
            "🏁 Conclusion",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")

    modes = sorted(df["Shipment Mode"].dropna().unique().tolist())
    countries = sorted(df["Country"].dropna().unique().tolist())
    vendors = sorted(df["Vendor"].dropna().unique().tolist())
    years = sorted(df["year"].dropna().unique().tolist())

    selected_modes = st.sidebar.multiselect("Shipment Mode", modes, default=[])
    selected_countries = st.sidebar.multiselect("Country", countries, default=[])
    selected_vendors = st.sidebar.multiselect("Vendor", vendors, default=[])
    selected_years = st.sidebar.multiselect("Year", years, default=[])

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Leave a filter empty to include all values for that field."
    )

    filters = {
        "Shipment Mode": selected_modes,
        "Country": selected_countries,
        "Vendor": selected_vendors,
        "year": selected_years,
    }
    return section, filters


def apply_filters(df, filters):
    """Apply the sidebar filter selections to the dataframe."""
    filtered = df.copy()
    for col, selected_values in filters.items():
        if selected_values:
            filtered = filtered[filtered[col].isin(selected_values)]
    return filtered


# ============================================================
# SECTION 7: CHART HELPER FUNCTIONS (one per notebook chart)
# ============================================================

def chart_missing_values(null_counts):
    """Bonus chart: bar plot of missing values per column (pre-cleaning)."""
    chart_data = null_counts[null_counts > 0]
    if chart_data.empty:
        st.info("No missing values found in the current dataset.")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(chart_data.index, chart_data.values, color="blue")
    ax.set_title("Count of NULL Values per Column")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)


def chart1_yearly_revenue(df):
    """Chart 1: Line chart - Year-wise revenue growth."""
    yearly_sales = df.groupby("year_recorded")["Line Item Value"].sum().reset_index()
    yearly_sales["Line Item Value"] = yearly_sales["Line Item Value"] / 1_000_000

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        yearly_sales["year_recorded"],
        yearly_sales["Line Item Value"],
        marker="o",
        linestyle=":",
    )
    ax.set_title("Year Wise Revenue Growth")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales Per Year (Millions)")
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)


def chart2_shipment_mode(df):
    """Chart 2: Bar charts - Most used shipment mode & most profitable mode."""
    shipment_count = df["Shipment Mode"].value_counts()
    shipment_revenue = df.groupby("Shipment Mode")["Line Item Value"].sum() / 1_000_000

    fig, ax = plt.subplots(2, 1, figsize=(7, 8))
    ax[0].bar(shipment_count.index, shipment_count.values, color="grey")
    ax[0].set_title("Most Used Shipment Type")
    ax[0].set_xlabel("Shipment Mode")
    ax[0].set_ylabel("Total Count")

    ax[1].bar(shipment_revenue.index, shipment_revenue.values, color="blue")
    ax[1].set_title("Most Profitable Shipment Mode")
    ax[1].set_xlabel("Shipment Mode")
    ax[1].set_ylabel("Total Revenue (Millions)")

    fig.suptitle("Shipment Mode and Profitability")
    fig.tight_layout()
    st.pyplot(fig)


def chart3_delivery_delay_boxplot(df):
    """Chart 3: Box plot - Delivery delay by shipment mode (IQR-cleaned)."""
    df_clean = remove_delay_outliers(df)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=df_clean, x="Shipment Mode", y="delivery delay", color="black", ax=ax)
    ax.set_title("Delivery Delay with Shipment Mode")
    ax.set_xlabel("Shipment Mode")
    ax.set_ylabel("Delivery Delay (days)")
    fig.tight_layout()
    st.pyplot(fig)

    delayed_count = df_clean.loc[df_clean["delivery delay"] < 0, "delivery delay"].count()
    st.caption(f"Total delayed deliveries (after outlier removal): **{delayed_count:,}**")


def chart4_scatter_revenue_quantity(df):
    """Chart 4: Scatter plot - Revenue vs Quantity by Shipment Mode."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="Line Item Quantity", y="Value_Millions",
        hue="Shipment Mode", alpha=0.6, ax=ax,
    )
    ax.set_title("Revenue VS Quantity in Shipment Mode")
    ax.set_xlabel("Quantities of Deliverables")
    ax.set_ylabel("Value Generated (Millions)")
    fig.tight_layout()
    st.pyplot(fig)


def chart5_stacked_bar_year_mode(df):
    """Chart 5: Stacked bar - Year-wise revenue contribution by shipment mode."""
    year_mode = (
        df.groupby(["year", "Shipment Mode"])["Item Value in Million"]
        .sum()
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    year_mode.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Year-wise Revenue Contribution by Shipment Mode")
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue (Millions)")
    ax.legend(title="Shipment Mode")
    fig.tight_layout()
    st.pyplot(fig)


def chart6_top_countries(df):
    """Chart 6: Horizontal bar - Top 5 countries by revenue."""
    top_countries = (
        df.groupby("Country")["Item Value in Million"].sum().sort_values(ascending=False).head(5)
    )
    colors = ["orange", "blue", "grey", "pink", "yellow"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_countries.values, y=top_countries.index, palette=colors, ax=ax)
    ax.set_title("Top 5 Countries by Revenue")
    ax.set_xlabel("Revenue (Millions)")
    ax.set_ylabel("Countries")
    fig.tight_layout()
    st.pyplot(fig)


def chart7_vendor_lollipop(df):
    """Chart 7: Lollipop chart - Top 5 vendors by revenue."""
    data = (
        df.groupby("Vendor")["Item Value in Million"]
        .sum()
        .reset_index()
        .sort_values(by="Item Value in Million", ascending=False)
        .head()
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hlines(y=data["Vendor"], xmin=0, xmax=data["Item Value in Million"], color="red")
    ax.plot(data["Item Value in Million"], data["Vendor"], "o", color="black")
    ax.set_title("Vendor Value Distribution")
    ax.set_xlabel("Value (Millions)")
    fig.tight_layout()
    st.pyplot(fig)


def chart8_regression_weight_freight(df):
    """Chart 8: Regression plot - Freight cost vs Weight (log-log scale)."""
    plot_df = df.dropna(subset=["Weight (Kilograms)", "Freight Cost (USD)"])
    plot_df = plot_df[(plot_df["Weight (Kilograms)"] > 0) & (plot_df["Freight Cost (USD)"] > 0)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        data=plot_df, x="Weight (Kilograms)", y="Freight Cost (USD)",
        scatter_kws={"alpha": 0.3}, line_kws={"color": "red"}, ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Regression Analysis: Freight Cost vs. Weight")
    fig.tight_layout()
    st.pyplot(fig)


def chart9_pie_product_group(df):
    """Chart 9: Pie chart - Distribution of total spend by top 3 product groups."""
    pie_data = (
        df.groupby("Product Group")["Line Item Value"]
        .sum()
        .reset_index()
        .sort_values(by="Line Item Value", ascending=False)
        .head(3)
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        pie_data["Line Item Value"], labels=pie_data["Product Group"],
        autopct="%1.2f%%", startangle=90, colors=plt.cm.Pastel1.colors,
        wedgeprops={"edgecolor": "black"},
    )
    ax.set_title("Distribution of Total Spend by Product Group")
    fig.tight_layout()
    st.pyplot(fig)


def chart10_choropleth(df):
    """Chart 10: Plotly choropleth - Total orders delivered by country."""
    country_orders = df.groupby("Country").size().reset_index(name="Total Orders")
    fig = px.choropleth(
        country_orders, locations="Country", locationmode="country names",
        color="Total Orders", color_continuous_scale="Viridis",
        title="Total Orders Delivered by Country",
    )
    fig.update_layout(geo=dict(showframe=True, showcoastlines=True))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("View underlying data table"):
        st.dataframe(country_orders.sort_values("Total Orders", ascending=False))


def chart11_facetgrid(df):
    """Chart 11: FacetGrid - Avg freight cost vs avg item value by shipment mode."""
    df_avg = df.groupby("Shipment Mode").agg(
        {"Freight Cost (USD)": "mean", "Line Item Value": "mean"}
    ).reset_index()

    g = sns.FacetGrid(df_avg, col="Shipment Mode", height=4)
    g.map_dataframe(sns.scatterplot, x="Freight Cost (USD)", y="Line Item Value")
    g.fig.suptitle("Average Freight Cost vs Average Item Value by Shipment Mode", y=1.05)
    st.pyplot(g.fig)


def chart12_treemap(df):
    """Chart 12: Plotly treemap - Business spend portfolio."""
    treemap_df = (
        df.groupby(["Product Group", "Sub Classification", "Vendor"])["Line Item Value"]
        .sum()
        .reset_index()
    )
    fig = px.treemap(
        treemap_df, path=["Product Group", "Sub Classification", "Vendor"],
        values="Line Item Value", color="Line Item Value",
        color_continuous_scale="Spectral_r", title="Business Spend Portfolio",
    )
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def chart13_countplot_first_line(df):
    """Chart 13: Count plot - First line treatment vs others."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(
        data=df, x="First Line Designation", palette="Set1",
        order=df["First Line Designation"].value_counts().index, ax=ax,
    )
    ax.set_title("Shipment Volume: First Line Treatment vs. Others")
    ax.set_xlabel("Is it a First Line Treatment?")
    ax.set_ylabel("Number of Shipments (Count)")
    fig.tight_layout()
    st.pyplot(fig)


def chart14_heatmap(df):
    """Chart 14: Heatmap - Country vs Product Group spend ($ Millions)."""
    top_countries = df.groupby("Country")["Line Item Value"].sum().nlargest(15).index
    df_top = df[df["Country"].isin(top_countries)]

    heatmap_data = df_top.pivot_table(
        index="Country", columns="Product Group", values="Line Item Value", aggfunc="sum"
    )
    heatmap_data = heatmap_data / 1_000_000

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={"label": "Total Value ($M)"}, ax=ax)
    ax.set_title("Regional Demand Heatmap: Country vs. Product Group ($ Millions)")
    ax.set_xlabel("Product Group")
    ax.set_ylabel("Country")
    fig.tight_layout()
    st.pyplot(fig)


def chart15_pairplot(df):
    """Chart 15: Pair plot - relationships between key numeric variables."""
    numeric_cols = ["Weight (Kilograms)", "Freight Cost (USD)", "Item Value in Million", "Unit Price"]
    plot_df = df[numeric_cols].dropna()

    # Sample for performance on large/unfiltered datasets while staying representative
    if len(plot_df) > 1500:
        plot_df = plot_df.sample(1500, random_state=42)
        st.caption("Note: a random sample of 1,500 rows is used for pair-plot performance.")

    g = sns.pairplot(plot_df, diag_kind="kde")
    st.pyplot(g.fig)


# ============================================================
# SECTION 8: MAIN APPLICATION
# ============================================================
def main():
    # --- Load & clean data ---
    df_raw = load_data()
    df_clean_full, null_counts_before, null_pct_before = clean_data(df_raw)

    # --- Sidebar: navigation + filters ---
    section, filters = render_sidebar(df_clean_full)

    # --- Apply filters to the cleaned dataset ---
    df = apply_filters(df_clean_full, filters)

    if df.empty:
        st.warning("No data matches the selected filters. Please broaden your filter selection.")
        return

    # --- Header ---
    st.title("📦 FedEx Logistics Performance Analysis")
    st.markdown(
        "An interactive dashboard analyzing FedEx/SCMS shipment data to uncover "
        "delivery performance, cost behavior, and operational insights."
    )
    st.markdown("---")

    # ============================================================
    # SECTION: OVERVIEW
    # ============================================================
    if section == "🏠 Overview & Objective":
        st.header("Project Overview")
        st.write(
            "This project performs Exploratory Data Analysis (EDA) on a FedEx/SCMS "
            "logistics dataset to understand how shipments are handled and how delivery "
            "performance can be improved. The workflow covers dataset inspection, data "
            "cleaning (duplicates, missing values, date formatting), shipment pattern "
            "analysis, delivery timeline analysis, cost analysis, and storytelling with "
            "charts to surface trends, patterns, and outliers."
        )

        st.header("Problem Statement")
        st.write(
            "The objective is to analyze the FedEx logistics dataset to understand "
            "shipment patterns, delivery performance, and cost behavior. The company "
            "handles a large number of shipments across different locations, service "
            "types, and transportation modes, and it is important to evaluate how "
            "efficiently these operations are being managed, identify factors that "
            "influence delivery delays and shipping costs, and highlight areas for "
            "improvement."
        )

        st.header("Business Objective")
        st.write(
            "Analyze FedEx shipment data to improve operational efficiency and delivery "
            "performance. By studying shipment trends, service types, delivery timelines, "
            "and cost patterns, the aim is to identify delays, high-cost areas, and "
            "performance gaps in the logistics process — supporting better decisions on "
            "route optimization, cost control, and customer satisfaction."
        )

        st.header("Dataset Snapshot")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Rows", f"{df_raw.shape[0]:,}")
        with c2:
            st.metric("Total Columns", f"{df_raw.shape[1]:,}")
        st.dataframe(df_raw.head())

    # ============================================================
    # SECTION: DATA CLEANING & QUALITY
    # ============================================================
    elif section == "🧹 Data Cleaning & Quality":
        st.header("Dataset Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", f"{df_raw.shape[0]:,}")
        with c2:
            st.metric("Columns", f"{df_raw.shape[1]:,}")
        with c3:
            st.metric("Duplicate Rows", f"{int(df_raw.duplicated().sum()):,}")

        st.subheader("Missing Values (Before Cleaning)")
        missing_table = pd.DataFrame({
            "Missing Count": null_counts_before,
            "Missing %": null_pct_before.round(2),
        })
        st.dataframe(missing_table[missing_table["Missing Count"] > 0])

        st.subheader("Missing Values Visualization")
        chart_missing_values(null_counts_before)

        st.subheader("Data Wrangling Steps Applied")
        st.markdown(
            """
            - **Shipment Mode**: missing values forward-filled (`ffill`).
            - **Dosage**: missing values backward-filled (`bfill`), since the first
              value in the column was null.
            - **Line Item Insurance (USD)**: missing numeric values filled with the
              column mean.
            - **Date columns** (`Scheduled Delivery Date`, `Delivered to Client Date`,
              `Delivery Recorded Date`): parsed into proper `datetime` format
              (`%d-%b-%y`).
            - **Freight Cost (USD)** and **Weight (Kilograms)**: coerced to numeric,
              converting non-numeric text entries to `NaN`.
            - **Delivery Delay**: derived as the day difference between
              `Delivered to Client Date` and `Scheduled Delivery Date`.
            """
        )
        st.success(f"After cleaning, total remaining null values: {int(df_clean_full.isnull().sum().sum()):,}")

        st.subheader("Statistical Summary (Numerical Columns)")
        st.dataframe(df.describe())

        st.subheader("Unique Values per Column")
        st.dataframe(df_clean_full.nunique().rename("Unique Values"))

    # ============================================================
    # SECTION: KPI DASHBOARD
    # ============================================================
    elif section == "📊 KPI Dashboard":
        st.header("Key Performance Indicators")
        render_kpi_cards(df)
        st.markdown("---")
        st.caption(
            "KPIs reflect the dataset after applying the sidebar filters "
            "(Shipment Mode, Country, Vendor, Year)."
        )
        st.subheader("Filtered Data Preview")
        st.dataframe(df.head(50))

    # ============================================================
    # SECTION: SHIPMENT MODE ANALYSIS
    # ============================================================
    elif section == "🚚 Shipment Mode Analysis":
        st.header("Shipment Mode Analysis")

        st.subheader("Chart 2 — Most Used vs. Most Profitable Shipment Mode")
        chart2_shipment_mode(df)
        st.info(
            "Air dominates shipment volume (over 50% of shipments) due to its speed, "
            "while Ocean is the least used. Air also generates the most revenue "
            "(600M+), followed by Truck (550M+)."
        )

        st.subheader("Chart 13 — First Line Treatment vs. Others")
        chart13_countplot_first_line(df)
        st.info(
            "Shows whether most shipments relate to first-line treatment. A higher "
            "count indicates strong demand for essential medicines; this supports "
            "inventory and supply-chain prioritization decisions."
        )

    # ============================================================
    # SECTION: REVENUE TRENDS
    # ============================================================
    elif section == "📈 Revenue Trends":
        st.header("Revenue Trends")

        st.subheader("Chart 1 — Year-wise Revenue Growth")
        chart1_yearly_revenue(df)
        st.info(
            "Revenue increased almost linearly year over year, with declines noted "
            "in 2013 and 2015. 2014 was the best-performing year, generating over "
            "$250M in revenue."
        )

        st.subheader("Chart 5 — Year-wise Revenue Contribution by Shipment Mode")
        chart5_stacked_bar_year_mode(df)
        st.info(
            "Truck and Air contribute the largest share of yearly revenue, "
            "highlighting the importance of domestic distribution. Ocean's "
            "contribution declined after 2011 and warrants review."
        )

        st.subheader("Chart 9 — Spend Distribution by Product Group")
        chart9_pie_product_group(df)
        st.info(
            "Spending is highly concentrated in one major product group, "
            "indicating both a strength to leverage and a concentration risk."
        )

    # ============================================================
    # SECTION: DELIVERY PERFORMANCE
    # ============================================================
    elif section == "⏱ Delivery Performance":
        st.header("Delivery Performance")

        st.subheader("Chart 3 — Delivery Delay by Shipment Mode")
        chart3_delivery_delay_boxplot(df)
        st.info(
            "After removing outliers (IQR method), most deliveries are completed "
            "on time, with Air Charter performing best (often slightly early). "
            "No shipment mode shows a major systemic delay problem."
        )

    # ============================================================
    # SECTION: COST ANALYSIS
    # ============================================================
    elif section == "💰 Cost Analysis":
        st.header("Cost Analysis")

        st.subheader("Chart 4 — Revenue vs. Quantity by Shipment Mode")
        chart4_scatter_revenue_quantity(df)
        st.info(
            "Revenue is strongly quantity-driven. Truck shipments deliver the "
            "largest quantities and generate the highest revenue (~$6M), while "
            "Air handles smaller, urgent shipments."
        )

        st.subheader("Chart 8 — Freight Cost vs. Weight (Regression)")
        chart8_regression_weight_freight(df)
        st.info(
            "A clear positive relationship exists between shipment weight and "
            "freight cost — heavier shipments cost more to ship, confirmed by the "
            "upward-sloping trend line on a log-log scale."
        )

        st.subheader("Chart 11 — Avg Freight Cost vs. Avg Item Value by Shipment Mode")
        chart11_facetgrid(df)
        st.info(
            "Comparing average freight cost against average item value across "
            "shipment modes helps identify which modes are cost-efficient versus "
            "which carry disproportionately high freight costs relative to value."
        )

    # ============================================================
    # SECTION: GEOGRAPHIC INSIGHTS
    # ============================================================
    elif section == "🌍 Geographic Insights":
        st.header("Geographic Insights")

        st.subheader("Chart 6 — Top 5 Countries by Revenue")
        chart6_top_countries(df)
        st.info(
            "Nigeria leads with ~$350M in revenue, followed by Zambia — these "
            "top markets are critical but also represent concentration risk."
        )

        st.subheader("Chart 10 — Total Orders Delivered by Country")
        chart10_choropleth(df)
        st.info(
            "The choropleth highlights geographic concentration of demand, "
            "useful for identifying key markets and expansion opportunities."
        )

        st.subheader("Chart 14 — Regional Demand Heatmap (Country vs. Product Group)")
        chart14_heatmap(df)
        st.info(
            "Some countries show strong concentration in specific product groups, "
            "while others have more balanced demand across categories."
        )

    # ============================================================
    # SECTION: VENDOR & PRODUCT INSIGHTS
    # ============================================================
    elif section == "🏭 Vendor & Product Insights":
        st.header("Vendor & Product Insights")

        st.subheader("Chart 7 — Top 5 Vendors by Revenue")
        chart7_vendor_lollipop(df)
        st.info(
            "A visible gap exists between the top vendor and the rest, showing "
            "uneven revenue distribution and reliance on a few key vendors."
        )

        st.subheader("Chart 12 — Business Spend Portfolio (Treemap)")
        chart12_treemap(df)
        st.info(
            "The treemap reveals which product groups, sub-classifications, and "
            "vendors dominate overall spend — useful for vendor management and "
            "identifying over-dependence risk."
        )

    # ============================================================
    # SECTION: CORRELATION ANALYSIS
    # ============================================================
    elif section == "🔗 Correlation Analysis":
        st.header("Correlation Analysis")

        st.subheader("Chart 15 — Pair Plot of Key Numeric Variables")
        chart15_pairplot(df)
        st.info(
            "Weight and freight cost tend to move together (heavier shipments cost "
            "more), and item value generally scales with unit price. The diagonal "
            "KDE plots reveal the distribution and skewness of each variable."
        )

    # ============================================================
    # SECTION: BUSINESS INSIGHTS
    # ============================================================
    elif section == "💡 Business Insights":
        st.header("Business Insights")
        st.markdown(
            """
            1. **Shipment Mode**: Air dominates shipment volume and revenue, but this
               concentration creates high cost exposure — other modes should be
               balanced to manage risk.
            2. **Revenue Trend**: Revenue grew almost linearly year over year, with
               2014 being the strongest year and 2013/2015 showing declines worth
               investigating.
            3. **Delivery Performance**: Once outliers are removed, delivery timing is
               largely stable and under control, with Air Charter performing best.
            4. **Cost Behavior**: Freight cost rises with shipment weight; if cost
               growth outpaces item value for heavier shipments, profit margins can
               erode.
            5. **Revenue Concentration**: A small number of countries (Nigeria,
               Zambia, etc.) and vendors generate a disproportionate share of
               revenue, indicating both strength and concentration risk.
            6. **Product Mix**: Spend is concentrated in a single dominant product
               group, which is efficient but risky if demand for that category
               shifts.
            7. **Cost Efficiency by Mode**: Some shipment modes carry high freight
               cost without proportional value generation, reducing profit margins.
            """
        )

    # ============================================================
    # SECTION: RECOMMENDATIONS
    # ============================================================
    elif section == "✅ Recommendations":
        st.header("Recommendations")
        st.write(
            "Based on the overall analysis of shipment patterns, cost behavior, and "
            "regional demand, the company should focus on optimizing freight cost by "
            "selecting more cost-efficient shipment modes wherever possible. High "
            "freight-cost modes that do not generate proportional revenue should be "
            "reviewed to improve profit margins. The company should also strengthen "
            "operations in high-demand countries and product groups, as these regions "
            "contribute major revenue, while reducing over-dependence on a few "
            "countries or vendors to avoid business risk. Improving delivery "
            "performance by monitoring delays and comparing scheduled vs. actual "
            "delivery dates can enhance customer satisfaction. Overall, using "
            "data-driven insights for shipment planning, vendor management, and "
            "regional expansion strategy can help the company improve operational "
            "efficiency, reduce costs, and achieve sustainable business growth."
        )

    # ============================================================
    # SECTION: CONCLUSION
    # ============================================================
    elif section == "🏁 Conclusion":
        st.header("Conclusion")
        st.markdown(
            """
            1. The dataset shows shipment, cost, and delivery patterns clearly.
            2. Revenue is concentrated in a few countries and product groups.
            3. Some shipment modes increase freight cost disproportionately.
            4. Cost optimization can improve profit margins.
            5. Key vendors drive the majority of revenue.
            6. Over-dependence on a few vendors/countries creates business risk.
            7. Data-driven insights support better, more confident decisions.
            """
        )
        st.success("🎉 FedEx Logistics Performance Analysis — Dashboard Complete!")

    # --- Persistent footer KPI strip for quick reference (all sections) ---
    st.markdown("---")
    with st.expander("📌 Quick KPI Reference (current filters)"):
        render_kpi_cards(df)


if __name__ == "__main__":
    main()
