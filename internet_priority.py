import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import pydeck as pdk

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("data/Infrastruktur/Data UD - Kepmendagri.xlsx", sheet_name="Master Data")
    df_bts = pd.read_excel('data/Infrastruktur/Data BTS - Kepmendagri.xlsx')
    df_cs = pd.read_excel('data/Infrastruktur/Data CS - Kepmendagri.xlsx')
    df_pop = pd.read_excel('data/Infrastruktur/Data PoP - Kepmendagri.xlsx')
    df_odp = pd.read_excel("data/Infrastruktur/Data ODP - Kepmendagri.xlsx", sheet_name="PP_DES23_3")
    df['composite_score'] = (df['download'] + df['upload']) / 2
    return df, df_bts, df_cs, df_pop, df_odp

df, df_bts, df_cs, df_pop, df_odp = load_data()

# Sidebar navigation
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio(
    "Select Analysis", 
    [
        "🏠 Overview", 
        "📊 Digital Readiness", 
        "🌍 Province Aggregates", 
        "📡 Speed Clustering", 
        "🛰️ Usage-Based Segmentation",
        "🌐 ISP Performance",
        "🏗️ BTS Coverage",
        "📶 CS Coverage Analysis",           
        "🔗 Combined Coverage & Speed",     
        "🧵 ODP Utilization",
        "📍 Combined ODP & BTS",
        "⚙️ Deployment Infrastructure",
        "📎 Other Analysis"
    ]
)

# --- Page: Overview ---
if page == "🏠 Overview":
    st.title("📶 Internet Performance Dashboard")
    st.markdown("""
Welcome! Use the sidebar to navigate between different analyses.  
This dashboard is based on internet performance data (download & upload speeds) across Indonesia.
""")
    st.dataframe(df.head())

# --- Page: Digital Readiness ---
elif page == "📊 Digital Readiness":
    st.title("📊 Digital Readiness Analysis")

    # Sidebar filters
    min_score = st.sidebar.slider("Minimum Composite Score", 0.0, 120.0, 50.0)
    top_n = st.sidebar.slider("Top N Regions", 5, 50, 10)

    # Filter data
    filtered_df = df[df['composite_score'] >= min_score]
    top_regions = filtered_df.sort_values(by='composite_score', ascending=False).head(top_n)

    # Table
    st.subheader(f"🏆 Top {top_n} Regions")
    st.dataframe(top_regions[['provinsi', 'kab/kota', 'download', 'upload', 'composite_score']])

    # Chart
    chart = alt.Chart(top_regions).mark_bar().encode(
        x=alt.X("composite_score:Q", title="Composite Score (Mbps)"),
        y=alt.Y("kab/kota:N", sort='-x', title="Kabupaten/Kota"),
        color="provinsi:N",
        tooltip=["provinsi", "kab/kota", "download", "upload", "composite_score"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

# --- Page: Province Aggregates & Digital Divide ---
elif page == "🌍 Province Aggregates":
    st.title("🌍 Province-Level Aggregate & Digital Divide Mapping")

    # Aggregate by province
    province_avg = df.groupby("provinsi").agg({
        "download": "mean",
        "upload": "mean",
        "composite_score": "mean"
    }).reset_index().sort_values(by="composite_score", ascending=False)

    # Show table
    st.subheader("📊 Average Scores by Province")
    st.dataframe(province_avg)

    # Bar chart
    st.subheader("📈 Composite Score by Province")
    bar_chart = alt.Chart(province_avg).mark_bar().encode(
        x=alt.X("composite_score:Q", title="Composite Score (Mbps)"),
        y=alt.Y("provinsi:N", sort='-x'),
        tooltip=["download", "upload", "composite_score"]
    ).properties(height=500)
    st.altair_chart(bar_chart, use_container_width=True)

# --- Page: Speed Clustering ---
elif page == "📡 Speed Clustering":
    st.title("📡 Internet Speed Clustering")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Sidebar: number of clusters
    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)

    # Feature selection
    features = df[["download", "upload", "composite_score"]]

    # Normalize
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(scaled_features)

    # Show cluster summary
    st.subheader("📋 Clustered Data")
    st.dataframe(df[["provinsi", "kab/kota", "download", "upload", "composite_score", "cluster"]])

    # Cluster centers
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                                   columns=["download", "upload", "composite_score"])
    cluster_centers["cluster"] = cluster_centers.index

    st.subheader("📌 Cluster Centers")
    st.dataframe(cluster_centers)

    # Scatter plot
    st.subheader("📈 Cluster Visualization")
    chart = alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X("download", scale=alt.Scale(zero=False)),
        y=alt.Y("upload", scale=alt.Scale(zero=False)),
        color=alt.Color("cluster:N", legend=alt.Legend(title="Cluster")),
        tooltip=["provinsi", "kab/kota", "download", "upload", "composite_score", "cluster"]
    ).interactive().properties(height=500)
    st.altair_chart(chart, use_container_width=True)

# --- Page: Usage-Based Segmentation ---
elif page == "🛰️ Usage-Based Segmentation":
    st.title("🛰️ Internet Usage Pattern Segmentation")

    st.markdown("""
    This segmentation groups regions by usage patterns based on download and upload speeds:
    - **Digitally Ready**: High download + high upload  
    - **Consumption-Heavy**: High download + low upload  
    - **Underserved**: Low download + low upload  
    """)

    # Percentile thresholds
    dl_threshold = df['download'].quantile(0.6)
    ul_threshold = df['upload'].quantile(0.6)

    def classify_region(row):
        if row['download'] >= dl_threshold and row['upload'] >= ul_threshold:
            return "Digitally Ready"
        elif row['download'] >= dl_threshold and row['upload'] < ul_threshold:
            return "Consumption-Heavy"
        else:
            return "Underserved"

    df['usage_segment'] = df.apply(classify_region, axis=1)

    # Segment counts
    segment_counts = df['usage_segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Number of Regions']

    st.subheader("📊 Segment Distribution")
    st.dataframe(segment_counts)

    # Bar chart
    chart = alt.Chart(df).mark_circle(size=90).encode(
        x=alt.X("download", scale=alt.Scale(zero=False), title="Download Speed (Mbps)"),
        y=alt.Y("upload", scale=alt.Scale(zero=False), title="Upload Speed (Mbps)"),
        color=alt.Color("usage_segment:N", title="Segment"),
        tooltip=["provinsi", "kab/kota", "download", "upload", "usage_segment"]
    ).interactive().properties(height=500)
    st.altair_chart(chart, use_container_width=True)

    # Show segmented table
    st.subheader("📋 Region Details by Segment")
    st.dataframe(df[["provinsi", "kab/kota", "download", "upload", "usage_segment"]].sort_values(by="usage_segment"))

        # === Underserved Regions ===
    st.subheader("🚨 Underserved Regions: Priority for Improvement")

    underserved_df = df[df['usage_segment'] == "Underserved"].copy()
    underserved_df = underserved_df.sort_values(by="composite_score")

    st.markdown(f"**Total Underserved Regions:** {len(underserved_df)}")

    st.dataframe(underserved_df[['provinsi', 'kab/kota', 'download', 'upload', 'composite_score']])

    # Bottom N visualization
    st.subheader("📉 Bottom 10 Regions by Composite Score")
    bottom_10 = underserved_df.nsmallest(10, "composite_score")

    chart = alt.Chart(bottom_10).mark_bar().encode(
        x=alt.X("composite_score:Q", title="Composite Score (Mbps)"),
        y=alt.Y("kab/kota:N", sort='-x', title="Kabupaten/Kota"),
        color=alt.Color("provinsi:N", title="Province"),
        tooltip=["provinsi", "kab/kota", "download", "upload", "composite_score"]
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

    # Optional: Export
    st.download_button(
        label="📥 Download Underserved Regions (CSV)",
        data=underserved_df.to_csv(index=False),
        file_name="underserved_regions.csv",
        mime="text/csv"
    )
    
        # === Upgrade Gap Analysis ===
    st.subheader("🔧 Upgrade Requirement for Underserved Regions")

    # Thresholds for target speeds
    target_download = 50
    target_upload = 20

    # Calculate gaps
    underserved_df["download_gap"] = (target_download - underserved_df["download"]).clip(lower=0)
    underserved_df["upload_gap"] = (target_upload - underserved_df["upload"]).clip(lower=0)
    underserved_df["total_gap"] = underserved_df["download_gap"] + underserved_df["upload_gap"]

    st.markdown(f"""
    Each region is evaluated against a digital inclusion baseline:
    - **Download target**: {target_download} Mbps  
    - **Upload target**: {target_upload} Mbps
    """)

    st.dataframe(underserved_df[[
        "provinsi", "kab/kota", "download", "upload",
        "download_gap", "upload_gap", "total_gap"
    ]].sort_values(by="total_gap", ascending=False))

    # Bar chart of biggest shortfalls
    st.subheader("📉 Regions with Largest Total Internet Speed Gap")
    largest_gaps = underserved_df.sort_values(by="total_gap", ascending=False).head(10)

    chart = alt.Chart(largest_gaps).mark_bar().encode(
        x=alt.X("total_gap:Q", title="Total Speed Gap (Mbps)"),
        y=alt.Y("kab/kota:N", sort='-x'),
        color=alt.Color("provinsi:N", title="Province"),
        tooltip=["provinsi", "kab/kota", "download", "upload", "download_gap", "upload_gap", "total_gap"]
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

    # Total bandwidth shortfall
    total_dl_gap = underserved_df["download_gap"].sum()
    total_ul_gap = underserved_df["upload_gap"].sum()

    st.markdown(f"""
    ### 📊 Total Estimated Upgrade Need:
    - **Download shortfall:** {total_dl_gap:.1f} Mbps  
    - **Upload shortfall:** {total_ul_gap:.1f} Mbps  
    """)

    # Download button
    st.download_button(
        label="📥 Download Upgrade Gap Data (CSV)",
        data=underserved_df.to_csv(index=False),
        file_name="internet_upgrade_needs.csv",
        mime="text/csv"
    )

        # === Province-level Aggregation of Upgrade Needs ===
    st.subheader("🗺️ Province-Level Upgrade Needs Summary")

    province_summary = underserved_df.groupby("provinsi").agg({
        "download_gap": "sum",
        "upload_gap": "sum",
        "total_gap": "sum",
        "kab/kota": "count"
    }).rename(columns={"kab/kota": "region_count"}).reset_index()

    province_summary = province_summary.sort_values(by="total_gap", ascending=False)

    st.dataframe(province_summary)

    # Province chart
    chart = alt.Chart(province_summary).mark_bar().encode(
        x=alt.X("total_gap:Q", title="Total Speed Gap (Mbps)"),
        y=alt.Y("provinsi:N", sort='-x'),
        color=alt.Color("region_count:Q", title="Number of Underserved Regions"),
        tooltip=["provinsi", "region_count", "download_gap", "upload_gap", "total_gap"]
    ).properties(height=500)

    st.altair_chart(chart, use_container_width=True)

    # Download grouped data
    st.download_button(
        label="📥 Download Province-Level Upgrade Summary (CSV)",
        data=province_summary.to_csv(index=False),
        file_name="province_upgrade_summary.csv",
        mime="text/csv"
    )

elif page == "🌐 ISP Performance":
    st.title("🌐 ISP Performance Analysis")

    
    st.markdown("""
This analysis shows the **average internet performance** by Internet Service Provider (ISP),  
based on available download and upload speeds across Indonesia.
    """)

    # Merge speed data and ISP data
    df_merged = pd.merge(df_pop, df, on=["provinsi", "kab/kota"], how="inner")

    # Compute average speeds per ISP
    isp_summary = df_merged.groupby("nama_penyelenggara").agg(
        avg_download=("download", "mean"),
        avg_upload=("upload", "mean"),
        region_count=("kab/kota", "nunique")
    ).reset_index().sort_values(by="avg_download", ascending=False)

    # Optional filter: min number of regions
    min_regions = st.sidebar.slider("Minimum Regions per ISP", 1, 10, 1)
    filtered_isp = isp_summary[isp_summary["region_count"] >= min_regions]

    # Show table
    st.subheader("📶 ISP Speed Rankings")
    st.dataframe(filtered_isp)

    # Chart: Top 10 ISPs
    st.subheader("🏆 Top 10 ISPs by Download Speed")
    chart = alt.Chart(filtered_isp.head(10)).mark_bar().encode(
        x=alt.X("avg_download:Q", title="Avg Download Speed (Mbps)"),
        y=alt.Y("nama_penyelenggara:N", sort='-x', title="ISP"),
        color=alt.Color("avg_upload:Q", title="Avg Upload Speed"),
        tooltip=["nama_penyelenggara", "avg_download", "avg_upload", "region_count"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)
    
    st.subheader("📊 ISP Speed Distribution")

    selected_isp = st.selectbox("Choose ISP", sorted(df_merged['nama_penyelenggara'].unique()))
    speed_dist = df_merged[df_merged['nama_penyelenggara'] == selected_isp]

    st.altair_chart(
        alt.Chart(speed_dist).transform_fold(
            ["download", "upload"],
            as_=["Speed Type", "Mbps"]
        ).mark_boxplot().encode(
            x="Speed Type:N",
            y="Mbps:Q",
            tooltip=["provinsi", "kab/kota"]
        ).properties(width=300, height=300),
        use_container_width=True
    )

    best_isp_by_region = df_merged.groupby(["provinsi", "kab/kota"]).apply(
    lambda g: g.sort_values(by="download", ascending=False).iloc[0]
    ).reset_index(drop=True)

    st.subheader("🏅 Best ISP by Region")
    st.dataframe(best_isp_by_region[["provinsi", "kab/kota", "nama_penyelenggara", "download", "upload"]])
    
    st.title("⚡ ISP Efficiency Score")

    st.markdown("""
The **Efficiency Score** balances internet speed and regional coverage.  
ISPs with high download/upload speeds and wide coverage score higher.
    """)

    import numpy as np

    # Reuse earlier ISP summary or create fresh
    df_merged = pd.merge(df_pop, df, on=["provinsi", "kab/kota"], how="inner")
    isp_summary = df_merged.groupby("nama_penyelenggara").agg(
        avg_download=("download", "mean"),
        avg_upload=("upload", "mean"),
        region_count=("kab/kota", "nunique")
    ).reset_index()

    # Efficiency score formula
    isp_summary["efficiency_score"] = (
        (isp_summary["avg_download"] + isp_summary["avg_upload"]) *
        np.log1p(isp_summary["region_count"])  # log1p handles log(0) safely
    )

    # Filter by region presence if desired
    min_regions = st.slider("Minimum Regions per ISP", 1, 10, 2)
    filtered = isp_summary[isp_summary["region_count"] >= min_regions]

    # Show table
    st.subheader("📈 ISP Efficiency Table")
    st.dataframe(filtered.sort_values(by="efficiency_score", ascending=False))

    # Chart
    st.subheader("🏆 Top 10 Efficient ISPs")
    chart = alt.Chart(filtered.sort_values(by="efficiency_score", ascending=False).head(10)).mark_bar().encode(
        x=alt.X("efficiency_score:Q", title="Efficiency Score"),
        y=alt.Y("nama_penyelenggara:N", sort='-x', title="ISP"),
        color="region_count:Q",
        tooltip=["nama_penyelenggara", "avg_download", "avg_upload", "region_count", "efficiency_score"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)
    
    st.subheader("🚨 Underperforming Regions per ISP")
    st.markdown("""
    This analysis highlights **underperforming regions** where internet speeds fall below an acceptable threshold, 
    based on reported download and upload performance by Internet Service Providers (ISPs).
    """)
        # Merge speed data with ISP data
    # Merge speed data with ISP data
    df_merged = pd.merge(df_pop, df, on=["provinsi", "kab/kota"], how="inner")

    # Calculate composite score
    df_merged["composite_score"] = (df_merged["download"] + df_merged["upload"]) / 2

    # Select threshold
    speed_threshold = st.sidebar.slider("Max Acceptable Composite Speed (Mbps)", 10.0, 100.0, 20.0)

    # Filter underperforming
    underperforming = df_merged[df_merged["composite_score"] < speed_threshold]

    st.subheader(f"📉 Underperforming Regions (Composite < {speed_threshold} Mbps)")

    # Region-level summary
    region_summary = underperforming.groupby(["provinsi", "kab/kota"]).agg(
        isps=("nama_penyelenggara", "nunique"),
        avg_download=("download", "mean"),
        avg_upload=("upload", "mean"),
        avg_score=("composite_score", "mean"),
    ).reset_index().sort_values(by="avg_score")

    st.markdown("### 📍 Regions with Low Overall ISP Performance")
    st.dataframe(region_summary)

    st.markdown("### 🏢 ISP-Level Details (in Underperforming Regions)")
    st.dataframe(underperforming[[
        "provinsi", "kab/kota", "nama_penyelenggara", "download", "upload", "composite_score"
    ]].sort_values(by="composite_score"))

    # ISP Bar Chart
    isp_count = underperforming["nama_penyelenggara"].value_counts().reset_index()
    isp_count.columns = ["ISP", "Underperforming Region Count"]

    st.subheader("🏆 ISPs with Most Underperforming Regions")
    chart = alt.Chart(isp_count.head(10)).mark_bar().encode(
        x=alt.X("Underperforming Region Count:Q", title="Region Count"),
        y=alt.Y("ISP:N", sort='-x'),
        color=alt.value("#FF4136"),
        tooltip=["ISP", "Underperforming Region Count"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

# --- Page: BTS Coverage ---
elif page == "🏗️ BTS Coverage":
    st.title("🏗️ BTS Infrastructure Coverage")

    st.markdown("""
This section shows the number of **BTS (Base Transceiver Station)** sites deployed in each region.  
You can use the filter to analyze BTS rollout by type or region.
    """)

    
    # Filter by category (e.g., NEW SITE 4G, SITE EXISTING)
    bts_categories = df_bts['kategori'].unique().tolist()
    selected_category = st.sidebar.selectbox("BTS Category", ["All"] + bts_categories)

    if selected_category != "All":
        filtered_bts = df_bts[df_bts['kategori'] == selected_category]
    else:
        filtered_bts = df_bts

    # Aggregate by province/kabupaten
    bts_count = (
        filtered_bts.groupby(["provinsi", "kab/kota"])
        .size()
        .reset_index(name="bts_site_count")
        .sort_values(by="bts_site_count", ascending=False)
    )

    # Display table
    st.subheader("📍 BTS Site Count per Region")
    st.dataframe(bts_count)

    # Plot top 15 kabupaten/kota
    st.subheader("🏙️ Top 15 Regions by BTS Count")
    top_bts = bts_count.head(15)
    chart = alt.Chart(top_bts).mark_bar().encode(
        x=alt.X("bts_site_count:Q", title="Number of BTS Sites"),
        y=alt.Y("kab/kota:N", sort='-x', title="Kabupaten/Kota"),
        color="provinsi:N",
        tooltip=["provinsi", "kab/kota", "bts_site_count"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)
    
    # --- BTS vs Internet Speed Correlation ---
    st.subheader("📉 BTS Count vs Internet Performance")

    # Prepare average speed data per region
    speed_by_region = (
        df.groupby(["provinsi", "kab/kota"])
        .agg(composite_score=("composite_score", "mean"))
        .reset_index()
    )

    # Merge with BTS count
    merged_bts_speed = pd.merge(bts_count, speed_by_region, on=["provinsi", "kab/kota"], how="left")

    # Display correlation table
    st.markdown("Average internet speed vs. BTS infrastructure by region.")
    st.dataframe(merged_bts_speed)

    # Scatter plot
    scatter = alt.Chart(merged_bts_speed).mark_circle(size=80).encode(
        x=alt.X("bts_site_count:Q", title="Number of BTS Sites"),
        y=alt.Y("composite_score:Q", title="Composite Internet Speed (Mbps)"),
        color="provinsi:N",
        tooltip=["provinsi", "kab/kota", "bts_site_count", "composite_score"]
    ).properties(
        height=400,
        width=600,
        title="📊 BTS Sites vs Internet Speed by Region"
    ).interactive()

    st.altair_chart(scatter, use_container_width=True)
    
    st.subheader("🔍 Regional Infrastructure Classification")

    # Set threshold interactively
    min_bts = st.sidebar.slider("Min BTS (for high coverage)", 5, 100, 20)
    min_speed = st.sidebar.slider("Min Composite Speed (Mbps)", 1.0, 100.0, 20.0)

    # Classify each region
    def classify(row):
        if row['bts_site_count'] < min_bts and row['composite_score'] < min_speed:
            return "🚨 Underserved"
        elif row['bts_site_count'] >= min_bts and row['composite_score'] < min_speed:
            return "⚠️ Overbuilt"
        elif row['bts_site_count'] < min_bts and row['composite_score'] >= min_speed:
            return "⚡ Efficient"
        else:
            return "✅ Well-served"

    merged_bts_speed["category"] = merged_bts_speed.apply(classify, axis=1)

    # Show result table
    st.dataframe(merged_bts_speed.sort_values(by="category"))

    # Plot: Color-coded scatter
    scatter_class = alt.Chart(merged_bts_speed).mark_circle(size=80).encode(
        x=alt.X("bts_site_count:Q", title="Number of BTS Sites"),
        y=alt.Y("composite_score:Q", title="Composite Speed (Mbps)"),
        color=alt.Color("category:N", legend=alt.Legend(title="Region Category")),
        tooltip=["provinsi", "kab/kota", "bts_site_count", "composite_score", "category"]
    ).properties(
        height=400,
        width=600,
        title="📍 Region Classification by BTS & Speed"
    ).interactive()

    st.altair_chart(scatter_class, use_container_width=True)

    st.subheader("🗺️ Map: Regional BTS vs Speed Classification")

    # Compute mean coordinates per region from BTS data
    coords = df_bts.groupby(["provinsi", "kab/kota"]).agg(
        lat=("lat", "mean"),
        long=("long", "mean")
    ).reset_index()

    # Merge with classified data
    map_df = pd.merge(merged_bts_speed, coords, on=["provinsi", "kab/kota"], how="left").dropna(subset=["lat", "long"])

    # Map using Streamlit's built-in map (color-coded by category)


    color_map = {
        "🚨 Underserved": [255, 0, 0],
        "⚠️ Overbuilt": [255, 165, 0],
        "⚡ Efficient": [0, 128, 255],
        "✅ Well-served": [0, 200, 0]
    }

    map_df["color"] = map_df["category"].map(color_map)

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=map_df["lat"].mean(),
            longitude=map_df["long"].mean(),
            zoom=5,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[long, lat]',
                get_fill_color="color",
                get_radius=8000,
                pickable=True,
                opacity=0.8,
            )
        ],
        tooltip={"text": "{provinsi} - {kab/kota}\nCategory: {category}\nBTS: {bts_site_count}\nSpeed: {composite_score} Mbps"}
    ))
    
    st.subheader("🗺️ BTS vs Speed Classification")
    
    # Merge BTS + speed data
    merged_corr = pd.merge(bts_count, df, on=["provinsi", "kab/kota"], how="inner")

    # Show summary stats
    corr_value = merged_corr["bts_site_count"].corr(merged_corr["composite_score"])
    st.metric("📈 Correlation", f"{corr_value:.2f}", help="Pearson correlation between BTS count and composite speed")

    st.dataframe(merged_corr[["provinsi", "kab/kota", "bts_site_count", "composite_score"]].sort_values(by="composite_score", ascending=False))

    # Scatterplot
    scatter = alt.Chart(merged_corr).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X("bts_site_count:Q", title="BTS Site Count"),
        y=alt.Y("composite_score:Q", title="Composite Speed (Mbps)"),
        color="provinsi:N",
        tooltip=["provinsi", "kab/kota", "bts_site_count", "composite_score"]
    ).interactive().properties(height=450)

    # Optional: Regression line
    regression = scatter + scatter.transform_regression("bts_site_count", "composite_score").mark_line(color="red")

    st.altair_chart(regression, use_container_width=True)

# ------------------- PAGE: CS Coverage ----------------------
elif page == "📶 CS Coverage Analysis":
    st.title("📶 Province-Level Mobile Network Coverage")

    coverage_cols = ['2g', '3g', '4g', '5g']
    df_cs[coverage_cols] = df_cs[coverage_cols].clip(0, 1)
    df_cs['coverage_score'] = (
        df_cs['2g'] * 0.1 + df_cs['3g'] * 0.2 + df_cs['4g'] * 0.5 + df_cs['5g'] * 0.2
    )

    cs_summary = df_cs.groupby('original_provinsi').agg({
        '2g': 'mean', '3g': 'mean', '4g': 'mean', '5g': 'mean', 'coverage_score': 'mean'
    }).reset_index()
    cs_summary.columns = ['Provinsi', 'Avg 2G', 'Avg 3G', 'Avg 4G', 'Avg 5G', 'Coverage Score']

    selected = st.multiselect("Filter by Province", cs_summary['Provinsi'].unique())
    filtered = cs_summary[cs_summary['Provinsi'].isin(selected)] if selected else cs_summary

    st.dataframe(filtered.style.format("{:.2%}", subset=cs_summary.columns[1:]), use_container_width=True)

    st.subheader("📊 Coverage Score by Province")
    chart = alt.Chart(filtered).mark_bar().encode(
        x=alt.X("Coverage Score:Q"),
        y=alt.Y("Provinsi:N", sort='-x'),
        tooltip=[
        alt.Tooltip('Avg 2G', format='.2%'),
        alt.Tooltip('Avg 3G', format='.2%'),
        alt.Tooltip('Avg 4G', format='.2%'),
        alt.Tooltip('Avg 5G', format='.2%'),
        alt.Tooltip('Coverage Score', format='.2%')
    ]
    ).properties(height=600)

    st.altair_chart(chart, use_container_width=True)

# ------------------- PAGE: Combined Analysis ----------------------
elif page == "🔗 Combined Coverage & Speed":
    st.title("🔗 Combined CS Coverage & Internet Speed")

    # --- Preprocess CS ---
    coverage_cols = ['2g', '3g', '4g', '5g']
    df_cs[coverage_cols] = df_cs[coverage_cols].clip(0, 1)
    df_cs['coverage_score'] = (
        df_cs['2g'] * 0.1 + df_cs['3g'] * 0.2 + df_cs['4g'] * 0.5 + df_cs['5g'] * 0.2
    )
    
    
    cs_summary = df_cs.groupby('provinsi').agg({
        '2g': 'mean', '3g': 'mean', '4g': 'mean', '5g': 'mean', 'coverage_score': 'mean'
    }).reset_index()
    cs_summary.columns = ['Provinsi', 'Avg 2G', 'Avg 3G', 'Avg 4G', 'Avg 5G', 'Coverage Score']

    # --- Preprocess Speed ---
    df['composite_score'] = (df['download'] + df['upload']) / 2
    speed_summary = df.groupby('provinsi').agg({
        'download': 'mean', 'upload': 'mean', 'composite_score': 'mean'
    }).reset_index()
    speed_summary.columns = ['Provinsi', 'Avg Download', 'Avg Upload', 'Avg Composite Speed']
    
    combined = pd.merge(cs_summary, speed_summary, on='Provinsi', how='left')

    selected = st.multiselect("Filter by Province", combined['Provinsi'].unique())
    filtered = combined[combined['Provinsi'].isin(selected)] if selected else combined

    st.dataframe(filtered.style.format("{:.2%}", subset=['Avg 2G','Avg 3G','Avg 4G','Avg 5G','Coverage Score']), use_container_width=True)

    st.subheader("📈 4G Coverage vs. Download Speed")
    scatter = alt.Chart(filtered).mark_circle(size=80).encode(
        x=alt.X("Avg 4G", type='quantitative'),
        y=alt.Y("Avg Download", type='quantitative'),
        tooltip=[
                    alt.Tooltip('Provinsi', type='nominal'),
                    alt.Tooltip('Avg 4G', format='.2%', type='quantitative'),
                    alt.Tooltip('Avg Download', format='.2f', type='quantitative')
                ]
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

    st.subheader("📈 Coverage Score vs. Composite Speed")
    scatter2 = alt.Chart(filtered).mark_circle(size=80, color='green').encode(
        x=alt.X("Coverage Score", type='quantitative'),
        y=alt.Y("Avg Composite Speed", type='quantitative'),
        tooltip=[
                    alt.Tooltip('Provinsi', type='nominal'),
                    alt.Tooltip('Avg 4G', format='.2%', type='quantitative'),
                    alt.Tooltip('Avg Download', format='.2f', type='quantitative')
                ]
    ).interactive()
    st.altair_chart(scatter2, use_container_width=True)

# --- New Sidebar Page ---
elif page == "🧵 ODP Utilization":
    st.title("🧵 Optical Distribution Point Utilization Overview")

    # Rename columns to simplify access
    df_odp.columns = df_odp.columns.str.lower().str.replace(" ", "_")
    df_odp["utilization_pct"] = (df_odp["kapasitas_fo_terpakai_(core)"] / df_odp["kapasitas_fo_max_(core)"]) * 100

    # Aggregate by region (provinsi or kab/kota)
    region_level = st.sidebar.radio("Group By", ["provinsi", "kab/kota"])
    odp_summary = df_odp.groupby(region_level).agg(
        total_odp=("kode_odp", "count"),
        total_capacity=("kapasitas_fo_max_(core)", "sum"),
        used_capacity=("kapasitas_fo_terpakai_(core)", "sum")
    ).reset_index()

    odp_summary["utilization_pct"] = (odp_summary["used_capacity"] / odp_summary["total_capacity"]) * 100

    # Filter for minimum number of ODPs
    min_odps = st.sidebar.slider("Minimum ODPs per Region", 1, 100, 10)
    odp_filtered = odp_summary[odp_summary["total_odp"] >= min_odps]

    # Display table
    st.subheader("📋 Regional ODP Utilization Summary")
    st.dataframe(odp_filtered.sort_values(by="utilization_pct", ascending=False))

    # Bar chart
    st.subheader("📊 Top 10 Regions by ODP Utilization")
    chart = alt.Chart(odp_filtered.sort_values(by="utilization_pct", ascending=False).head(10)).mark_bar().encode(
        x=alt.X("utilization_pct:Q", title="Utilization (%)"),
        y=alt.Y(f"{region_level}:N", sort='-x', title="Region"),
        color=alt.value("#4CAF50"),
        tooltip=[region_level, "total_odp", "used_capacity", "total_capacity", "utilization_pct"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)
    
    st.subheader("🚨 Low Capacity ODPs")

    df_odp.columns = df_odp.columns.str.lower().str.replace(" ", "_")
    
    # Calculate utilization %
    df_odp["utilization_pct"] = (df_odp["kapasitas_fo_terpakai_(core)"] / df_odp["kapasitas_fo_max_(core)"]) * 100
    
    # Filter ODPs near capacity
    low_capacity = df_odp[
        (df_odp["utilization_pct"] > 90) |
        (df_odp["sisa_kapasitas_fo_(core)"] < 2)
    ]

    st.markdown("""
    These are ODP units that are either **over 90% utilized** or have **less than 2 FO cores** remaining.  
    Such regions may require urgent upgrades or expansion.
    """)

    st.dataframe(low_capacity[[
        "provinsi", "kab/kota", "kecamatan", "kelurahan/desa",
        "kode_odp", "kapasitas_fo_max_(core)", "kapasitas_fo_terpakai_(core)",
        "sisa_kapasitas_fo_(core)", "utilization_pct"
    ]].sort_values(by="utilization_pct", ascending=False))

    # Bar chart by kab/kota (optional)
    top_kota = (
        low_capacity.groupby("kab/kota")
        .size().reset_index(name="overutilized_odp")
        .sort_values(by="overutilized_odp", ascending=False)
        .head(10)
    )

    st.subheader("📊 Top 10 Kab/Kota with Low Capacity ODPs")
    chart = alt.Chart(top_kota).mark_bar().encode(
        x=alt.X("overutilized_odp:Q", title="Low Capacity ODP Count"),
        y=alt.Y("kab/kota:N", sort="-x"),
        color=alt.value("#e74c3c")
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

    st.subheader("⚠️ ODP Saturation Detection by Region")
    # Sidebar filters
    selected_province = st.sidebar.selectbox("Filter by Province for Saturation Detection", ["All"] + sorted(df_odp["provinsi"].unique()))
    if selected_province != "All":
        df_odp = df_odp[df_odp["provinsi"] == selected_province]

    # Aggregate by kab/kota
    agg = df_odp.groupby(["provinsi", "kab/kota"]).agg(
        total_odp=("kode_odp", "count"),
        avg_utilization_pct=("utilization_pct", "mean"),
        overutilized_odp=("utilization_pct", lambda x: (x > 90).sum())
    ).reset_index()

    agg["overutilized_ratio"] = agg["overutilized_odp"] / agg["total_odp"] * 100

    st.markdown("""
This chart shows which regions have high numbers of **overutilized ODPs**  
(e.g., >90% fiber usage), indicating the **need for upgrades or additional deployment**.
""")

    # Table
    st.dataframe(agg.sort_values(by="overutilized_ratio", ascending=False))

    # Bar chart of top 15 kab/kota by overutilization ratio
    st.subheader("📊 Top 15 Regions by ODP Saturation Ratio")
    chart = alt.Chart(agg.sort_values("overutilized_ratio", ascending=False).head(15)).mark_bar().encode(
        x=alt.X("overutilized_ratio:Q", title="ODPs >90% Utilization (%)"),
        y=alt.Y("kab/kota:N", sort='-x', title="Kabupaten/Kota"),
        color=alt.value("#e67e22"),
        tooltip=["provinsi", "kab/kota", "total_odp", "overutilized_odp", "overutilized_ratio"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)
    
    st.subheader("🗺️ ODP Deployment Map")
    
    selected_prov = st.sidebar.selectbox("Filter by Province for Mapping", sorted(df_odp["provinsi"].unique()))
    if selected_prov != "All":
        df_odp = df_odp[df_odp["provinsi"] == selected_prov]
        
    # Filter invalid coordinates
    map_data = df_odp.dropna(subset=["lat", "long"])

    st.markdown("""
This map shows the geographic distribution of **ODP infrastructure**.  
Dot color represents utilization level.
""")

    # Color scale
    import pydeck as pdk

    def get_color(util):
        if util >= 90:
            return [255, 0, 0]     # red
        elif util >= 70:
            return [255, 165, 0]   # orange
        elif util >= 50:
            return [255, 255, 0]   # yellow
        else:
            return [0, 200, 0]     # green

    map_data["color"] = map_data["utilization_pct"].apply(get_color)

    # Map render
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=map_data["lat"].mean(),
            longitude=map_data["long"].mean(),
            zoom=5,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position='[long, lat]',
                get_fill_color="color",
                get_radius=500,
                pickable=True,
                opacity=0.7,
            )
        ],
        tooltip={
            "text": "{provinsi} - {kab/kota}\nODP: {kode_odp}\nUtilization: {utilization_pct:.1f}%"
        }
    ))

    if selected_prov != "All":
        df_filtered = df_odp[df_odp["provinsi"] == selected_prov]
    else:
        df_filtered = df_odp.copy()
    
    # Classify utilization status
    def classify_utilization(pct):
        if pct >= 90:
            return "🔴 Overutilized"
        elif pct <= 30:
            return "🟢 Underutilized"
        else:
            return "⚪ Normal"

    df_filtered["utilization_status"] = df_filtered["utilization_pct"].apply(classify_utilization)

    # Filter only flagged ODPs
    flagged_odps = df_filtered[df_filtered["utilization_status"] != "⚪ Normal"]

    # Display
    st.subheader("🚦 Flagged ODPs (Overutilized & Underutilized)")
    st.markdown(f"Showing **{len(flagged_odps)}** ODPs that are either overutilized (≥90%) or underutilized (≤30%).")
    st.dataframe(flagged_odps[[
        "provinsi", "kab/kota", "kapasitas_fo_max_(core)", 
        "kapasitas_fo_terpakai_(core)", "utilization_pct", 
        "utilization_status", "lat", "long"
    ]])

    # Map
    st.subheader("🗺️ Map of Flagged ODPs")

    color_map = {
        "🔴 Overutilized": [255, 0, 0],
        "🟢 Underutilized": [0, 200, 0],
        "⚪ Normal": [200, 200, 200]
    }
    flagged_odps["color"] = flagged_odps["utilization_status"].map(color_map)

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=flagged_odps["lat"].mean(),
            longitude=flagged_odps["long"].mean(),
            zoom=5,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=flagged_odps,
                get_position='[long, lat]',
                get_fill_color="color",
                get_radius=5000,
                pickable=True,
                opacity=0.8,
            )
        ],
        tooltip={"text": "{provinsi} - {kab/kota}\nStatus: {utilization_status}\nUtilization: {utilization_pct}%"}
    ))
    
    st.title("📈 Forecasting ODP Exhaustion")
    st.markdown("""
This analysis estimates **how long until each ODP runs out of fiber capacity**,  
based on a simple monthly growth assumption in usage.
""")

    if selected_prov != "All":
        df_forecast = df_odp[df_odp["provinsi"] == selected_prov].copy()
    else:
        df_forecast = df_odp.copy()

    # User defines monthly usage growth
    monthly_growth = st.sidebar.slider("Assumed Monthly Port Usage Growth", 0.1, 5.0, 0.5)

    # Calculate forecast
    df_forecast["remaining_ports"] = df_forecast["kapasitas_fo_max_(core)"] - df_forecast["kapasitas_fo_terpakai_(core)"]
    df_forecast["months_until_full"] = df_forecast["remaining_ports"] / monthly_growth
    df_forecast["months_until_full"] = df_forecast["months_until_full"].replace([np.inf, -np.inf], np.nan)

    # Filter exhausted within 6 months
    alert_df = df_forecast[df_forecast["months_until_full"] <= 6].dropna(subset=["months_until_full"])

    st.subheader("⚠️ ODPs Expected to Run Out in ≤ 6 Months")
    st.dataframe(alert_df[[
        "provinsi", "kab/kota", "kapasitas_fo_max_(core)", "kapasitas_fo_terpakai_(core)",
        "sisa_kapasitas_fo_(core)", "months_until_full", "lat", "long"
    ]])

    # Map
    if not alert_df.empty:
        st.subheader("🗺️ Map: At-Risk ODPs (≤ 6 Months)")
        alert_df["color"] = [[255, 0, 0]] * len(alert_df)
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=alert_df["lat"].mean(),
                longitude=alert_df["long"].mean(),
                zoom=5,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=alert_df,
                    get_position='[long, lat]',
                    get_fill_color="color",
                    get_radius=5000,
                    pickable=True,
                    opacity=0.8,
                )
            ],
            tooltip={"text": "{provinsi} - {kab/kota}\n⚠️ Full in {months_until_full} months"}
        ))
    else:
        st.info("✅ No ODPs expected to be full within 6 months based on current growth.")

elif page == "📍 Combined ODP & BTS":  
    st.title("📈 Combined Infrastructure ODP & BTS Analysis")
    st.markdown("""
This gives us a combined infrastructure readiness view, identifying areas where:
\n - There’s fiber (ODP) but no BTS → last-mile not yet wireless.
\n - There’s BTS but no nearby ODP → wireless infra without fiber backhaul.
\n - Both exist → potentially well-connected regions.
\n - Neither exist → infrastructure gap areas.
""")
        # Count BTS per region
    bts_count = df_bts.groupby(["provinsi", "kab/kota"]).size().reset_index(name="bts_count")

    # Count ODP per region
    odp_count = df_odp.groupby(["provinsi", "kab/kota"]).size().reset_index(name="odp_count")

    # Merge them
    infra_merged = pd.merge(bts_count, odp_count, on=["provinsi", "kab/kota"], how="outer").fillna(0)

    # Classify readiness
    def classify(row):
        if row["bts_count"] > 0 and row["odp_count"] > 0:
            return "✅ BTS + ODP"
        elif row["bts_count"] > 0:
            return "📡 BTS only"
        elif row["odp_count"] > 0:
            return "🧵 ODP only"
        else:
            return "❌ No Infrastructure"

    infra_merged["infra_status"] = infra_merged.apply(classify, axis=1)
    
    # Get mean lat/long by region
    bts_coords = df_bts.groupby(["provinsi", "kab/kota"]).agg(
        lat=("lat", "mean"),
        long=("long", "mean")
    ).reset_index()

    odp_coords = df_odp.groupby(["provinsi", "kab/kota"]).agg(
        lat=("lat", "mean"),
        long=("long", "mean")
    ).reset_index()

    # Prefer BTS coordinates first, fallback to ODP
    coords = pd.merge(bts_coords, odp_coords, on=["provinsi", "kab/kota"], how="outer", suffixes=("_bts", "_odp"))

    # Choose lat/long from BTS first, fallback to ODP
    coords["lat"] = coords["lat_bts"].combine_first(coords["lat_odp"])
    coords["long"] = coords["long_bts"].combine_first(coords["long_odp"])

    coords = coords[["provinsi", "kab/kota", "lat", "long"]]

    infra_merged = pd.merge(infra_merged, coords, on=["provinsi", "kab/kota"], how="left")
    
    st.subheader("📍 Combined Infrastructure Readiness by Region")
    st.dataframe(infra_merged.sort_values(by=["infra_status"]))
    
    chart = alt.Chart(infra_merged).mark_bar().encode(
    x=alt.X("odp_count:Q", title="ODP Count"),
    y=alt.Y("kab/kota:N", sort='-x', title="Kab/Kota"),
    color=alt.Color("infra_status:N", title="Infrastructure Status"),
    tooltip=["provinsi", "kab/kota", "bts_count", "odp_count", "infra_status"]
    ).properties(height=500)

    st.altair_chart(chart, use_container_width=True)
    
    # Color mapping for each class
    color_map = {
        "✅ BTS + ODP": [0, 200, 0],
        "📡 BTS only": [255, 165, 0],
        "🧵 ODP only": [0, 128, 255],
        "❌ No Infrastructure": [255, 0, 0]
    }

    infra_merged["color"] = infra_merged["infra_status"].map(color_map)

    st.subheader("🗺️ Infrastructure Status  Map (BTS + ODP)")

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=infra_merged["lat"].mean(),
            longitude=infra_merged["long"].mean(),
            zoom=5,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=infra_merged.dropna(subset=["lat", "long"]),
                get_position='[long, lat]',
                get_fill_color="color",
                get_radius=8000,
                pickable=True,
                opacity=0.8,
            )
        ],
        tooltip={"text": "{provinsi} - {kab/kota}\nStatus: {infra_status}"}
    ))
    
    st.title("📡 ODP-to-BTS Load Ratio Analysis")

    # Step 1: Count ODPs and BTS per region
    odp_count = df_odp.groupby(["provinsi", "kab/kota"]).size().reset_index(name="odp_count")
    bts_count = df_bts.groupby(["provinsi", "kab/kota"]).size().reset_index(name="bts_count")

    # Step 2: Merge counts
    ratio_df = pd.merge(odp_count, bts_count, on=["provinsi", "kab/kota"], how="inner")

    # Step 3: Compute ratio
    ratio_df["odp_per_bts"] = ratio_df["odp_count"] / ratio_df["bts_count"]

    # Optional: Classify regions
    def classify_ratio(r):
        if r < 5:
            return "⚠️ BTS-constrained"
        elif r > 20:
            return "🧵 Fiber-heavy"
        else:
            return "✅ Balanced"

    ratio_df["classification"] = ratio_df["odp_per_bts"].apply(classify_ratio)

    # Sidebar filter
    min_bts = st.sidebar.slider("Minimum BTS per Region", 1, 10, 3)
    filtered = ratio_df[ratio_df["bts_count"] >= min_bts]

    # Show table
    st.subheader("📊 ODP-to-BTS Ratio by Region")
    st.dataframe(filtered.sort_values("odp_per_bts", ascending=False))

    # Bar chart
    st.subheader("🏙️ Top 15 Regions by ODP-to-BTS Ratio")
    import altair as alt
    chart = alt.Chart(filtered.sort_values("odp_per_bts", ascending=False).head(15)).mark_bar().encode(
        x=alt.X("odp_per_bts:Q", title="ODP per BTS"),
        y=alt.Y("kab/kota:N", sort='-x', title="Region"),
        color="classification:N",
        tooltip=["provinsi", "kab/kota", "odp_count", "bts_count", "odp_per_bts"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

    # Explanation
    st.markdown("""
    ### 📘 Interpretation
    - **BTS-constrained**: More BTS infrastructure may be needed to handle ODP demand.
    - **Fiber-heavy**: ODPs may be underutilizing available BTS backhaul.
    - **Balanced**: Infrastructure appears proportionally deployed.
    """)

    st.subheader("🗺️ Map: ODP-to-BTS Load Ratio by Region")

    # Estimate coordinates per region using BTS locations
    region_coords = df_bts.groupby(["provinsi", "kab/kota"]).agg(
        lat=("lat", "mean"),
        long=("long", "mean")
    ).reset_index()

    # Merge with ratio classification
    map_data = pd.merge(ratio_df, region_coords, on=["provinsi", "kab/kota"], how="left").dropna(subset=["lat", "long"])

    # Define color mapping
    color_map = {
        "⚠️ BTS-constrained": [255, 99, 71],     # Red
        "🧵 Fiber-heavy": [100, 149, 237],        # Blue
        "✅ Balanced": [60, 179, 113]             # Green
    }
    map_data["color"] = map_data["classification"].map(color_map)

    # Create pydeck map
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=map_data["lat"].mean(),
            longitude=map_data["long"].mean(),
            zoom=5,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position='[long, lat]',
                get_fill_color="color",
                get_radius=8000,
                pickable=True,
                opacity=0.8,
            )
        ],
        tooltip={
            "text": "{provinsi} - {kab/kota}\nODP: {odp_count}\nBTS: {bts_count}\nRatio: {odp_per_bts}\nType: {classification}"
        }
    ))

elif page == "⚙️ Deployment Infrastructure":
    st.title("⚙️ Speed vs Infrastructure Quantity Correlation")

    st.markdown("""
    This analysis explores whether the **quantity of physical infrastructure** (ODPs and BTS) in a region is correlated with better internet speeds.

    - **ODP (Optical Distribution Point)** and **BTS (Base Transceiver Station)** counts are aggregated per region.
    - The data is merged with average internet speed data (download/upload).
    - Scatter plots with regression lines show whether more infra = higher speed.
    """)

    # --- Step 1: Clean & Normalize ---
    df['kab/kota'] = df['kab/kota'].str.strip().str.lower()
    df['provinsi'] = df['provinsi'].str.strip().str.title()
    df_odp['kab/kota'] = df_odp['kab/kota'].str.strip().str.lower()
    df_bts['kab/kota'] = df_bts['kab/kota'].str.strip().str.lower()

    # --- Step 2: Count ODP & BTS per kab/kota ---
    odp_count = df_odp.groupby('kab/kota').size().reset_index(name='odp_count')
    bts_count = df_bts.groupby('kab/kota').size().reset_index(name='bts_count')

    # --- Step 3: Merge into speed data ---
    df_speed = df.copy()
    df_speed['composite_speed'] = (df_speed['download'] + df_speed['upload']) / 2
    merged = df_speed.merge(odp_count, on='kab/kota', how='left')
    merged = merged.merge(bts_count, on='kab/kota', how='left')

    # Drop missing
    merged = merged.dropna(subset=['composite_speed', 'odp_count', 'bts_count'])

    # --- Step 4: Sidebar controls ---
    st.sidebar.markdown("## Filter Options")
    prov_options = ["All"] + sorted(merged['provinsi'].dropna().unique())
    selected_prov = st.sidebar.selectbox("Select Province", prov_options)
    log_scale = st.sidebar.checkbox("Use log scale for BTS/ODP")

    if selected_prov != "All":
        merged = merged[merged['provinsi'] == selected_prov]

    # --- Step 5: Correlation Table ---
    st.subheader("📈 Correlation Matrix")
    corr = merged[['download', 'upload', 'composite_speed', 'odp_count', 'bts_count']].corr()
    st.dataframe(corr.style.background_gradient(cmap="RdBu", axis=None).format("{:.2f}"))

    # --- Step 6: Scatter Plot Function ---
    def scatter(x, y, label, color):
        base = alt.Chart(merged).mark_circle(size=70, color=color).encode(
            x=alt.X(x, scale=alt.Scale(type='log') if log_scale else alt.Scale(), title=label),
            y=alt.Y(y, title="Composite Speed (Mbps)"),
            tooltip=["kab/kota", x, y]
        )
        reg = base.transform_regression(x, y, method="linear").mark_line(color="red")
        return (base + reg).interactive()

    # --- Step 7: Visualizations ---
    st.subheader("📊 Composite Speed vs BTS Count")
    st.markdown("Do regions with more BTS stations actually achieve faster internet?")
    st.altair_chart(scatter("bts_count", "composite_speed", "Number of BTS", "steelblue"), use_container_width=True)

    st.subheader("📊 Composite Speed vs ODP Count")
    st.markdown("Are more fiber connection points (ODPs) associated with faster speeds?")
    st.altair_chart(scatter("odp_count", "composite_speed", "Number of ODP", "orange"), use_container_width=True)

    # --- Step 8: Automatic Insight Summary ---
    st.subheader("🔎 Summary Insights")
    top_outliers = merged[(merged['bts_count'] > merged['bts_count'].quantile(0.75)) & (merged['composite_speed'] < merged['composite_speed'].quantile(0.25))]
    bts_corr = corr.loc['bts_count', 'composite_speed']
    odp_corr = corr.loc['odp_count', 'composite_speed']

    st.markdown(f"""
    - **BTS vs Speed Correlation**: A coefficient of **{bts_corr:.2f}** indicates 
      {'a strong' if abs(bts_corr) > 0.6 else 'a moderate' if abs(bts_corr) > 0.3 else 'a weak'} 
      {'positive' if bts_corr > 0 else 'negative'} relationship between the number of BTS and composite internet speed.

    - **ODP vs Speed Correlation**: A coefficient of **{odp_corr:.2f}** suggests 
      {'a strong' if abs(odp_corr) > 0.6 else 'a moderate' if abs(odp_corr) > 0.3 else 'a weak'} 
      {'positive' if odp_corr > 0 else 'negative'} correlation between fiber endpoints and internet speed.

    - There are **{len(top_outliers)} regions** with **high BTS count but low internet speed**, indicating potential issues with performance, capacity, or backhaul limitations.
    """)

    st.title("🧭 Optimal BTS/ODP Deployment Targeting")

    st.markdown("""
This model prioritizes regions for **new BTS or ODP deployments** using:
- 🐌 Low composite internet speed
- 🏗️ Low existing infrastructure (BTS + ODP count)

Each region is scored and ranked. The top 100 are recommended for infrastructure expansion.
""")

    # --- Aggregate BTS + ODP Counts ---
    bts_count = df_bts.groupby(["provinsi", "kab/kota"]).size().reset_index(name="bts_count")
    odp_count = df_odp.groupby(["provinsi", "kab/kota"]).size().reset_index(name="odp_count")

    # --- Merge with Speed Data ---
    merged = df.copy()
    merged = merged.merge(bts_count, on=["provinsi", "kab/kota"], how="left")
    merged = merged.merge(odp_count, on=["provinsi", "kab/kota"], how="left")
    merged = merged.fillna(0)

    # --- Total Infra Count ---
    merged["infra_count"] = merged["bts_count"] + merged["odp_count"]

    # --- Compute Scores ---
    merged["speed_score"] = 1 - (merged["composite_score"] / merged["composite_score"].max())
    merged["infra_score"] = 1 - (merged["infra_count"] / merged["infra_count"].max())
    merged["targeting_score"] = 0.6 * merged["speed_score"] + 0.4 * merged["infra_score"]

    # --- Top 100 Target Regions ---
    top_100 = merged.sort_values("targeting_score", ascending=False).head(100)

    st.subheader("📋 Top 100 Regions Recommended for Deployment")
    st.dataframe(top_100[[
        "provinsi", "kab/kota", "composite_score", "bts_count", "odp_count", "infra_count", "targeting_score"
    ]])

    # # --- Map Coordinates ---
    # bts_coords = df_bts.groupby(["provinsi", "kab/kota"]).agg(
    #     lat=("lat", "mean"),
    #     long=("long", "mean")
    # ).reset_index()

    # odp_coords = df_odp.groupby(["provinsi", "kab/kota"]).agg(
    #     lat=("lat", "mean"),
    #     long=("long", "mean")
    # ).reset_index()

    # coords = pd.merge(bts_coords, odp_coords, on=["provinsi", "kab/kota"], how="outer", suffixes=("_bts", "_odp"))
    # coords["lat"] = coords["lat_bts"].combine_first(coords["lat_odp"])
    # coords["long"] = coords["long_bts"].combine_first(coords["long_odp"])
    # coords = coords[["provinsi", "kab/kota", "lat", "long"]]
    
    # # Split top 2 and the rest
    # top_2 = top_100.sort_values("targeting_score", ascending=False).head(2)
    # top_rest = top_100.sort_values("targeting_score", ascending=False).iloc[2:]

    # # Get coordinates for both
    # top_2_map = pd.merge(top_2, coords, on=["provinsi", "kab/kota"], how="left").dropna(subset=["lat", "long"])
    # top_rest_map = pd.merge(top_rest, coords, on=["provinsi", "kab/kota"], how="left").dropna(subset=["lat", "long"])
    # st.dataframe(top_rest)
    # st.dataframe(coords)
    # # # Add color and radius
    # # top_2_map["color"] = [[255, 0, 0]] * len(top_2_map)         # red
    # # top_2_map["radius"] = 10000

    # # top_rest_map["color"] = [[120, 120, 120]] * len(top_rest_map)  # gray
    # # top_rest_map["radius"] = 5000

    # # # Combine both for display
    # # combined_map = pd.concat([top_2_map, top_rest_map])

    # # st.subheader("🗺️ Top 100 Deployment Priority Regions (Top 2 Highlighted)")

    # # st.pydeck_chart(pdk.Deck(
    # #     map_style="mapbox://styles/mapbox/light-v9",
    # #     initial_view_state=pdk.ViewState(
    # #         latitude=combined_map["lat"].mean(),
    # #         longitude=combined_map["long"].mean(),
    # #         zoom=5,
    # #         pitch=0,
    # #     ),
    # #     layers=[
    # #         pdk.Layer(
    # #             "ScatterplotLayer",
    # #             data=combined_map,
    # #             get_position='[long, lat]',
    # #             get_fill_color="color",
    # #             get_radius="radius",
    # #             pickable=True,
    # #             opacity=0.8,
    # #         )
    # #     ],
    # #     tooltip={
    # #         "text": "{provinsi} - {kab/kota}\nTargeting Score: {targeting_score:.2f}\nInfra: {infra_count}\nSpeed: {composite_score:.1f} Mbps"
    # #     }
    # # ))


    # Optional CSV export
    st.download_button(
        label="📥 Download Top 100 Regions (CSV)",
        data=top_100.to_csv(index=False),
        file_name="optimal_target_regions.csv",
        mime="text/csv"
    )
# --- Page: Other Analysis ---
elif page == "📎 Other Analysis":
    st.title("📎 Other Analysis (Coming Soon)")
    st.markdown("You can add new analysis pages here using more sheets or new datasets.")
