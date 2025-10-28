# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import base64
import io
from datetime import datetime

# sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))                   
sys.path.append(str(ROOT / "src"))  

from fund_pipeline import analytics as an


# Try importing PDF generation libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from PIL import Image as PILImage
    import plotly.io as pio
    import kaleido
    HAS_REPORTLAB = True
    HAS_KALEIDO = True
except ImportError:
    HAS_REPORTLAB = False
    try:
        import kaleido
        HAS_KALEIDO = True
    except ImportError:
        HAS_KALEIDO = False
    st.warning("⚠️ ReportLab not installed. PDF generation will be limited.")

st.set_page_config(page_title="TooSharpe – Institutional Dashboard", layout="wide")

# Custom CSS for tabs
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def convert_plotly_to_image(fig, width=800, height=400):
    """Convert Plotly figure to PNG image bytes."""
    if not HAS_KALEIDO:
        return None
    try:
        img_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=2)
        return io.BytesIO(img_bytes)
    except Exception as e:
        # If conversion fails, try to export as static image without kaleido
        try:
            # Save to temporary buffer and read back
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.write_image(tmp.name, format='png', width=width, height=height)
                with open(tmp.name, 'rb') as f:
                    img_bytes = f.read()
                os.unlink(tmp.name)
                return io.BytesIO(img_bytes)
        except:
            return None

def generate_pdf_report(df, benchmark, as_of_date, analysis_level, sel_accounts, 
                       account_kpis_df, risk_kpis, liq_kpis, daily_aum_value, bench_df=None,
                       charts=None, position_liquidity_summary=None):
    """Generate PDF report for TooSharpe institutional investors."""
    if not HAS_REPORTLAB:
        return None
    
    # Create a BytesIO buffer for PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=0.5*inch, leftMargin=0.5*inch,
                           topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle',
                                 parent=styles['Heading1'],
                                 fontSize=24,
                                 textColor=colors.HexColor('#1f77b4'),
                                 spaceAfter=30,
                                 alignment=TA_CENTER)
    
    heading_style = ParagraphStyle('CustomHeading',
                                   parent=styles['Heading2'],
                                   fontSize=16,
                                   textColor=colors.HexColor('#1f77b4'),
                                   spaceAfter=12,
                                   spaceBefore=12)
    
    # Determine report title based on analysis level
    if analysis_level == "Account Level" and sel_accounts:
        if len(sel_accounts) == 1:
            report_title = f"TooSharpe – {sel_accounts[0]} Account Report"
        else:
            report_title = f"TooSharpe – {len(sel_accounts)} Accounts Report"
    else:
        report_title = "TooSharpe Fund Report"
    
    # Title
    elements.append(Paragraph(report_title, title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Format AUM properly
    aum_value = daily_aum_value / 1e6 if daily_aum_value >= 1e6 else daily_aum_value
    aum_formatted = f"${aum_value:,.2f}m" if daily_aum_value >= 1e6 else f"${daily_aum_value:,.0f}"
    
    # Format date properly
    if isinstance(as_of_date, pd.Timestamp):
        date_str = as_of_date.strftime("%B %d, %Y")
    elif hasattr(as_of_date, 'strftime'):
        date_str = as_of_date.strftime("%B %d, %Y")
    else:
        date_str = str(as_of_date)
    
    # Report Details
    details_data = [
        ["Analysis Level:", analysis_level],
        ["As of Date:", date_str],
        ["Benchmark:", benchmark],
        ["Fund AUM:", aum_formatted]
    ]
    
    details_table = Table(details_data, colWidths=[2*inch, 4*inch])
    details_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#333333')),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(details_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Calculate KPIs for comparison
    daily_ret = df.groupby("Date")["RET_TWR"].mean() if "RET_TWR" in df.columns else pd.Series(dtype=float)
    ann_ret = daily_ret.mean() * 252 if len(daily_ret) > 0 else 0
    vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 0 else 0
    sharpe = (ann_ret / vol) if vol > 0 else 0
    cum_ret = (1 + daily_ret).prod() - 1 if len(daily_ret) > 0 else 0
    
    # Calculate benchmark returns for comparison
    # Filter benchmark data to match portfolio's date range
    bench_ann_ret = 0
    if bench_df is not None and not bench_df.empty and benchmark in bench_df.columns and "Date" in bench_df.columns:
        # Convert Date to datetime if needed
        bench_df_temp = bench_df.copy()
        bench_df_temp["Date"] = pd.to_datetime(bench_df_temp["Date"], errors='coerce')
        bench_df_temp = bench_df_temp.dropna(subset=["Date"])
        
        # Get date range from portfolio df
        if "Date" in df.columns and len(daily_ret) > 0:
            date_min = pd.to_datetime(df["Date"]).min()
            date_max = pd.to_datetime(df["Date"]).max()
            
            # Filter benchmark to same date range
            bench_filtered = bench_df_temp[
                (bench_df_temp["Date"] >= date_min) & 
                (bench_df_temp["Date"] <= date_max)
            ]
            
            if not bench_filtered.empty and benchmark in bench_filtered.columns:
                bench_daily = bench_filtered.set_index("Date")[benchmark].dropna()
                if len(bench_daily) > 0:
                    bench_ann_ret = bench_daily.mean() * 252
    
    # Determine return strength based on comparison with benchmark
    excess_ret = ann_ret - bench_ann_ret if bench_ann_ret != 0 else ann_ret
    if excess_ret > 0.10 or (bench_ann_ret == 0 and ann_ret > 0.15):
        perf_adjective = "strong"
    elif excess_ret > 0.03 or (bench_ann_ret == 0 and ann_ret > 0.08):
        perf_adjective = "solid"
    elif excess_ret > -0.05 or (bench_ann_ret == 0 and ann_ret > 0):
        perf_adjective = "moderate"
    else:
        perf_adjective = "weak"
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    if bench_ann_ret != 0:
        exec_summary_text = f"""
        <para align="justify">
        This report presents a comprehensive analysis of the portfolio as of {date_str}, 
        analyzed at the {analysis_level.lower()} level. The portfolio has demonstrated {perf_adjective} performance characteristics, 
        generating an annualized return of {ann_ret:.2%} compared to the benchmark's {bench_ann_ret:.2%}, representing 
        an excess return of {excess_ret:.2%}. Key risk-adjusted metrics have been calculated using industry-standard methodologies 
        to provide institutional-level insights into portfolio positioning and performance attribution.
        </para>
        """
    else:
        exec_summary_text = f"""
        <para align="justify">
        This report presents a comprehensive analysis of the portfolio as of {date_str}, 
        analyzed at the {analysis_level.lower()} level. The portfolio has demonstrated {perf_adjective} performance characteristics, 
        generating an annualized return of {ann_ret:.2%}. Key risk-adjusted metrics have been calculated using industry-standard 
        methodologies to provide institutional-level insights into portfolio positioning and performance attribution.
        </para>
        """
    elements.append(Paragraph(exec_summary_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Key Performance Indicators
    elements.append(Paragraph("Key Performance Indicators", heading_style))
    
    # Calculate KPIs
    daily_ret = df.groupby("Date")["RET_TWR"].mean() if "RET_TWR" in df.columns else pd.Series(dtype=float)
    ann_ret = daily_ret.mean() * 252 if len(daily_ret) > 0 else 0
    vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 0 else 0
    sharpe = (ann_ret / vol) if vol > 0 else 0
    cum_ret = (1 + daily_ret).prod() - 1 if len(daily_ret) > 0 else 0
    
    # MDD calculation
    if len(daily_ret) > 0:
        eq = (1 + daily_ret).cumprod()
        peak = eq.cummax()
        mdd = (eq / peak - 1).min()
    else:
        mdd = 0
    
    kpi_data = [
        ["Metric", "Value"],
        ["Annualized Return", f"{ann_ret:.2%}"],
        ["Volatility", f"{vol:.2%}"],
        ["Sharpe Ratio (rf=3.8%)", f"{sharpe:.2f}"],
        ["Cumulative Return", f"{cum_ret:.2%}"],
        ["Max Drawdown", f"{mdd:.2%}"],
        ["Fund AUM", aum_formatted]
    ]
    
    kpi_table = Table(kpi_data, colWidths=[3.5*inch, 2.5*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Performance Commentary
    elements.append(Paragraph("Performance Analysis", heading_style))
    perf_text = f"""
    <para align="justify">
    The portfolio has generated an annualized return of {ann_ret:.2%} with a volatility of {vol:.2%}, 
    resulting in a Sharpe ratio of {sharpe:.2f}. The cumulative return since inception stands at {cum_ret:.2%}, 
    with a maximum drawdown of {mdd:.2%} observed during the analysis period. These metrics demonstrate 
    the risk-adjusted performance characteristics of the portfolio relative to institutional standards.
    </para>
    """
    elements.append(Paragraph(perf_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Account Performance Summary
    if not account_kpis_df.empty and "Account" in account_kpis_df.columns:
        elements.append(Paragraph("Account Performance Summary", heading_style))
        
        # Format account data for table
        account_data = [["Account", "Ann. Return (12M)", "Ann. Vol (12M)", "Sharpe (12M)"]]
        for _, row in account_kpis_df.head(10).iterrows():
            account_data.append([
                str(row.get("Account", "")),
                f"{row.get('AnnReturn12M', 0):.2%}" if not pd.isna(row.get("AnnReturn12M")) else "N/A",
                f"{row.get('AnnVol12M', 0):.2%}" if not pd.isna(row.get("AnnVol12M")) else "N/A",
                f"{row.get('Sharpe12M', 0):.2f}" if not pd.isna(row.get("Sharpe12M")) else "N/A"
            ])
        
        account_table = Table(account_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        account_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(account_table)
        elements.append(Spacer(1, 0.3*inch))
    
    # Risk & Attribution Section
    elements.append(Paragraph("Risk Exposure & Return Attribution", heading_style))
    risk_text = f"""
    <para align="justify">
    The portfolio exhibits diversified exposure across sectors and regions, with concentrated positions reflecting 
    the fund's investment strategy. Key risk metrics indicate prudent risk management practices, with gross leverage 
    and net exposure levels monitored on an ongoing basis to ensure alignment with target risk parameters.
    </para>
    """
    elements.append(Paragraph(risk_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Add Risk Charts
    if charts is not None and HAS_KALEIDO:
        try:
            # Add sector exposures chart
            if 'sector_exposures' in charts:
                img_buffer = convert_plotly_to_image(charts['sector_exposures'], width=700, height=300)
                if img_buffer:
                    img = Image(img_buffer, width=6*inch, height=2.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
            
            # Add additional risk charts if available
            if 'sector_nav' in charts:
                img_buffer = convert_plotly_to_image(charts['sector_nav'], width=700, height=300)
                if img_buffer:
                    img = Image(img_buffer, width=6*inch, height=2.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
            
            if 'market_cap_nav' in charts:
                img_buffer = convert_plotly_to_image(charts['market_cap_nav'], width=700, height=300)
                if img_buffer:
                    img = Image(img_buffer, width=6*inch, height=2.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            elements.append(Paragraph(f"<i>Risk charts unavailable: {str(e)}</i>", styles['Normal']))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Liquidity Commentary
    elements.append(Paragraph("Liquidity Analysis", heading_style))
    liq_text = f"""
    <para align="justify">
    Portfolio liquidity metrics are monitored to ensure sufficient market depth for position sizing and exit strategies. 
    The weighted average days to liquidate coverage is carefully managed to maintain operational flexibility while 
    preserving investment opportunities across various liquidity regimes.
    </para>
    """
    elements.append(Paragraph(liq_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Add Liquidity Charts
    if charts is not None and HAS_KALEIDO:
        try:
            # Add liquidity buckets chart
            if 'liquidity_buckets' in charts:
                img_buffer = convert_plotly_to_image(charts['liquidity_buckets'], width=700, height=300)
                if img_buffer:
                    img = Image(img_buffer, width=6*inch, height=2.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            elements.append(Paragraph(f"<i>Liquidity charts unavailable: {str(e)}</i>", styles['Normal']))
    
    # Add Position Liquidity Summary table if available
    if position_liquidity_summary is not None and isinstance(position_liquidity_summary, pd.DataFrame) and not position_liquidity_summary.empty:
        elements.append(Paragraph("Position Liquidity Summary", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Format the table data
        summary_data = [list(position_liquidity_summary.columns)]
        for _, row in position_liquidity_summary.head(10).iterrows():
            row_data = []
            for col, val in zip(position_liquidity_summary.columns, row.values):
                if col == "GMV" and isinstance(val, (int, float)):
                    row_data.append(f"${val:,.0f}")
                elif col == "ADV Cover" and isinstance(val, (int, float)):
                    row_data.append(f"{val:.2f}")
                else:
                    row_data.append(str(val))
            summary_data.append(row_data)
        
        summary_table = Table(summary_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer
    elements.append(Spacer(1, 0.2*inch))
    footer_style = ParagraphStyle('Footer',
                                 parent=styles['Normal'],
                                 fontSize=8,
                                 textColor=colors.grey,
                                 alignment=TA_CENTER)
    elements.append(Paragraph("Generated by TooSharpe Institutional Dashboard", footer_style))
    elements.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

@st.cache_data(show_spinner=True)
def _load_intermediary():
    """Load intermediary dataset from local files or Google Drive."""
    # Try to load from local files (prefer Parquet for deployment)
    intermediary_path = Path(__file__).resolve().parents[1] / "outputs" / "intermediary" / "latest"

    pq_path = intermediary_path / "intermediary.parquet"
    csv_path = intermediary_path / "intermediary.csv"

    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        df = an.coerce(df)
        return df
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = an.coerce(df)
        return df
    st.error("❌ No local intermediary file found (parquet/csv). Please run the data pipeline or include the file in the repo.")
    return pd.DataFrame()

@st.cache_data(show_spinner=True)
def _load_aum():
    """Load AUM data."""
    aum_path = Path(__file__).resolve().parents[1] / "data" / "AUM.csv"
    
    if aum_path.exists():
        aum_df = pd.read_csv(aum_path)
        # Convert date column
        if "AsOfDate" in aum_df.columns:
            aum_df["Date"] = pd.to_datetime(aum_df["AsOfDate"])
        elif "Date" in aum_df.columns:
            aum_df["Date"] = pd.to_datetime(aum_df["Date"])
        return aum_df
    else:
        return pd.DataFrame()

# ============================================================================
# Sidebar
# ============================================================================

# Load initial data to get date range for date picker
# Use local files only
_initial_df = _load_intermediary()

with st.sidebar:
    st.markdown("### Data Source")
    st.info("Using local files in repo: outputs/intermediary/latest/intermediary.parquet (fallback CSV)")
    
    st.markdown("---")
    st.markdown("### Analysis Level")
    
    # Analysis level selection
    analysis_level = st.radio(
        "Analysis Level",
        ["Fund Level (TooSharpe)", "Account Level"],
        index=0,
        help="Fund Level: Aggregate analysis across all accounts\nAccount Level: Detailed analysis for selected account(s)"
    )
    
    st.markdown("---")
    st.markdown("### Global Controls")
    
    # As of Date - user only selects ending date, default to earliest available
    if not _initial_df.empty and "Date" in _initial_df.columns:
        dmin, dmax = _initial_df["Date"].min(), _initial_df["Date"].max()
    else:
        # Default dates if no data available
        import datetime
        dmin = datetime.date(2022, 1, 1)
        dmax = datetime.date(2022, 12, 30)
    
    as_of_date = st.date_input("As of Date", value=dmax, min_value=dmin, max_value=dmax)
    
    # Benchmark
    benchmark_path = Path(__file__).resolve().parents[1] / "outputs" / "cleaned" / "DailyIndexReturns_returns.csv"
    if benchmark_path.exists():
        bench_df = pd.read_csv(benchmark_path)
        bm_choices = [c for c in bench_df.columns if c.lower() != "date"]
        benchmark = st.selectbox("Benchmark", bm_choices, index=0 if bm_choices else 0)
    else:
        benchmark = "ACWI"
        st.info("ℹ️ Benchmark file not found")
    
    # Account filter - only show for Account Level analysis
    # Note: df will be loaded after sidebar based on selected data source
    # We'll use _initial_df to get account list for now
    if analysis_level == "Account Level":
        st.markdown("---")
        st.markdown("#### Account Selection")
        if "Account" in _initial_df.columns:
            accs = sorted(_initial_df["Account"].dropna().unique().tolist())
            sel_accounts = st.multiselect("Select Account(s)", accs, default=accs[0] if accs else [])
        else:
            st.warning("No accounts available in data")
            sel_accounts = []
    elif analysis_level == "Fund Level":
        # For Fund Level, show which accounts are included but don't filter
        if "Account" in _initial_df.columns:
            accs = sorted(_initial_df["Account"].dropna().unique().tolist())
            st.markdown("---")
            st.markdown(f"#### Included Accounts ({len(accs)} total)")
            with st.expander("View all accounts"):
                for acc in accs:
                    st.text(f"• {acc}")
            # Keep all accounts for fund-level analysis
            sel_accounts = accs
        else:
            sel_accounts = []
    else:
        sel_accounts = []

# Load data (local only)
df = _load_intermediary()
aum_df = _load_aum()

if df.empty:
    st.warning("⚠️ No data loaded.")
    st.stop()

# Calculate daily AUM from initial AUM + cumulative PNL
def _calculate_daily_aum(df_data, aum_data):
    """Calculate daily AUM from initial AUM plus cumulative PNL."""
    if df_data.empty or aum_data.empty or "PNL" not in df_data.columns or "Date" not in df_data.columns:
        return pd.DataFrame()
    
    # Get initial AUM from AUM.csv (date 1/1/2022)
    initial_aum = aum_data[aum_data["Date"].dt.date == pd.to_datetime("2022-01-01").date()]
    if initial_aum.empty:
        return pd.DataFrame()
    
    initial_aum_value = float(initial_aum.iloc[0]["AUM"])
    
    # Calculate daily PNL (aggregate across all positions)
    daily_pnl = df_data.groupby("Date", observed=True)["PNL"].sum().sort_index()
    
    # Calculate cumulative AUM: AUM[t] = Initial AUM + cumulative PNL
    daily_aum = (initial_aum_value + daily_pnl.cumsum()).to_frame("AUM")
    daily_aum.reset_index(inplace=True)
    
    return daily_aum

# Calculate daily AUM for all dates
daily_aum_df = _calculate_daily_aum(df, aum_df)

# Apply date filter - from earliest date to selected as_of_date
dmin, dmax = df["Date"].min(), df["Date"].max()
df = df[(df["Date"] >= pd.to_datetime(dmin)) & (df["Date"] <= pd.to_datetime(as_of_date))]

# Store analysis level for use in display
st.session_state['analysis_level'] = analysis_level
st.session_state['sel_accounts'] = sel_accounts

# ============================================================================
# Main Dashboard
# ============================================================================

# Display title based on analysis level
if analysis_level == "Fund Level (TooSharpe)":
    st.title("TooSharpe – Institutional Dashboard")
    subtitle_text = "Fund-Level Analysis"
elif analysis_level == "Account Level" and sel_accounts:
    if len(sel_accounts) == 1:
        st.title(f"{sel_accounts[0]} – Institutional Dashboard")
        subtitle_text = f"Account-Level Analysis"
    else:
        st.title(f"{len(sel_accounts)} Accounts – Institutional Dashboard")
        subtitle_text = f"Multi-Account Analysis: {', '.join(sel_accounts[:2])}{'...' if len(sel_accounts) > 2 else ''}"
else:
    st.title("TooSharpe – Institutional Dashboard")
    subtitle_text = "Dashboard"

# Add subtitle
st.caption(subtitle_text)

# --- Overview KPIs ---
# Calculate basic KPIs
if "RET_TWR" in df.columns:
    daily_ret = df.groupby("Date")["RET_TWR"].mean()
else:
    daily_ret = pd.Series(dtype=float)

cum_ret = (1 + daily_ret).prod() - 1 if len(daily_ret) > 0 else 0
ann_ret = daily_ret.mean() * 252 if len(daily_ret) > 0 else 0
vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 0 else 0
sharpe = (ann_ret / vol) if vol > 0 else np.nan

# Max drawdown
if len(daily_ret) > 0:
    eq = (1 + daily_ret).cumprod()
    peak = eq.cummax()
    mdd = (eq / peak - 1).min()
else:
    mdd = 0

# Get Fund-level AUM (always fund-level, regardless of analysis level)
current_aum = None

if not daily_aum_df.empty and "Date" in daily_aum_df.columns and "AUM" in daily_aum_df.columns:
    # Filter AUM up to and including the selected date
    aum_filtered = daily_aum_df[daily_aum_df["Date"] <= pd.to_datetime(as_of_date)]
    if not aum_filtered.empty:
        # Get the most recent calculated AUM value
        current_aum = aum_filtered.sort_values("Date")["AUM"].iloc[-1] / 1e6  # Convert to millions
    else:
        # Fallback to latest available calculated AUM
        current_aum = daily_aum_df.sort_values("Date")["AUM"].iloc[-1] / 1e6
elif "NMV" in df.columns:
    # Fallback to using NMV if calculated AUM data not available
    current_aum = df["NMV"].abs().sum() / 1e6

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Ann. Return", f"{ann_ret:.2%}")
col2.metric("Volatility", f"{vol:.2%}")
col3.metric("Sharpe (rf=3.8%)", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
col4.metric("Cum. Return", f"{cum_ret:.2%}")
col5.metric("Max Drawdown", f"{mdd:.2%}")
col6.metric("Fund AUM", f"${current_aum:.2f}m" if current_aum is not None else "N/A")

# ============================================================================
# Tabs
# ============================================================================

tab_perf, tab_liq, tab_risk, tab_reports = st.tabs(
    ["Performance", "Liquidity", "Risk & Attribution", "Reports & Downloads"]
)

with tab_perf:
    st.header("Performance Analysis")
    
    # --- Performance-Specific KPI Row (non-duplicating with global KPIs) ---
    # Load benchmarks
    bench_df = an.load_bench(str(benchmark_path)) if benchmark_path.exists() else pd.DataFrame()
    
    # Use user-selected benchmark for calculations
    te_ir = an.tracking_error_and_info_ratio(df, bench_df, benchmark_col=benchmark)
    tracking_error = te_ir.get('tracking_error', 0)
    info_ratio = te_ir.get('info_ratio', 0)
    beta = te_ir.get('beta', 0)
    alpha = te_ir.get('alpha', 0)
    
    # Focus on benchmark-relative metrics for Performance tab
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Tracking Error", f"{tracking_error:.2%}" if not np.isnan(tracking_error) else "N/A")
    kpi_col2.metric("Info Ratio", f"{info_ratio:.2f}" if not np.isnan(info_ratio) else "N/A")
    kpi_col3.metric("Beta", f"{beta:.2f}" if not np.isnan(beta) else "N/A")
    kpi_col4.metric("Alpha (Ann.)", f"{alpha:.2%}" if not np.isnan(alpha) else "N/A")
    
    st.markdown("---")
    
    # --- Cumulative Return vs Benchmarks ---
    st.markdown("### Cumulative Return vs Benchmarks")
    if benchmark_path.exists() and "RET_TWR" in df.columns:
        # Get daily returns
        daily_rets = df.groupby("Date", observed=True)["RET_TWR"].mean().reset_index()
        daily_rets.columns = ["Date", "Ret"]
        daily_rets["Ret"] = daily_rets["Ret"].clip(-0.50, 0.50)
        
        # Time series chart
        ts_data = an.prepare_benchmark_timeseries(daily_rets, str(benchmark_path))
        if not ts_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_data["Date"], y=ts_data["Account_CumReturn"] * 100,
                                    mode='lines', name='Portfolio', line=dict(color='#1f77b4', width=2)))
            
            # Only show the selected benchmark
            bench_cols = [c for c in ts_data.columns if c.endswith("_CumReturn") and c != "Account_CumReturn"]
            selected_bench_col = f"{benchmark}_CumReturn"
            
            # Check if selected benchmark exists in the data
            if selected_bench_col in bench_cols:
                # Only show selected benchmark
                bench_name = selected_bench_col.replace("_CumReturn", "")
                fig.add_trace(go.Scatter(x=ts_data["Date"], y=ts_data[selected_bench_col] * 100,
                                        mode='lines', name=bench_name,
                                        line=dict(color='#ff7f0e', width=2, dash='dash')))
            else:
                # Fallback to showing first available benchmark
                if bench_cols:
                    bench_name = bench_cols[0].replace("_CumReturn", "")
                    fig.add_trace(go.Scatter(x=ts_data["Date"], y=ts_data[bench_cols[0]] * 100,
                                            mode='lines', name=bench_name,
                                            line=dict(color='#ff7f0e', width=2, dash='dash')))
            
            fig.update_layout(title=f"Cumulative Returns: Portfolio vs {benchmark}",
                            xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                            hovermode='x unified', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Rolling Excess Return ---
    st.markdown("### Rolling Excess Return (90D)")
    if benchmark_path.exists() and not bench_df.empty:
        excess_data = an.rolling_excess_return(df, bench_df, window=90, benchmark_col=benchmark)
        if not excess_data.empty and "RollingExcess" in excess_data.columns:
            fig = px.line(excess_data, x="Date", y="RollingExcess",
                         title="Rolling 90-Day Excess Return vs Benchmark",
                         labels={"RollingExcess": "Excess Return (bps)"})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Excess return data not available")
    
    st.markdown("---")
    
    # --- Account Performance ---
    if analysis_level == "Fund Level (TooSharpe)":
        st.markdown("### Account-Level Performance Summary")
    else:
        st.markdown("### Selected Account(s) Performance")
    
    if "Account" in df.columns and "RET_TWR" in df.columns:
        # Bar chart: Annualized Returns by Account
        account_ret_data = an.account_annualized_returns(df)
        
        if not account_ret_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Annualized Return by Account**")
                fig_bar = px.bar(account_ret_data, x="Account", y="AnnReturn",
                                title="Annualized Returns (%)",
                                labels={"AnnReturn": "Return (%)"},
                                color="AnnReturn",
                                color_continuous_scale="RdYlGn")
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                st.markdown("**Return vs Volatility**")
                # Use absolute return values for size to make dots bigger
                fig_scatter = px.scatter(account_ret_data, x="AnnVol", y="AnnReturn",
                                        size=account_ret_data["AnnReturn"].abs() * 3,  # Scale up the size
                                        color="Account",  # Different color per account
                                        color_discrete_sequence=px.colors.qualitative.Set3,
                                        hover_name="Account",
                                        title="Return vs Volatility by Account",
                                        labels={"AnnReturn": "Annualized Return (%)",
                                               "AnnVol": "Annualized Volatility (%)"},
                                        size_max=50)  # Larger max size
                fig_scatter.update_traces(marker=dict(line=dict(width=1.5, color='DarkSlateGrey')))
                fig_scatter.update_layout(height=400, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Account KPIs table
    account_kpis_df = an.perf_account_kpis(df)
    if not account_kpis_df.empty:
        if analysis_level == "Fund Level (TooSharpe)":
            st.markdown("### Account Performance Summary")
        else:
            st.markdown("### Performance Summary")
        # Format the dataframe for display (removed MaxDrawdown column)
        display_cols = ["Account", "AnnReturn12M", "AnnVol12M", "Sharpe12M"]
        available_cols = [c for c in display_cols if c in account_kpis_df.columns]
        display_df = account_kpis_df[available_cols].round(4).copy()
        # Format other metrics
        if "AnnReturn12M" in display_df.columns:
            display_df["AnnReturn12M"] = display_df["AnnReturn12M"].apply(
                lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A"
            )
        if "AnnVol12M" in display_df.columns:
            display_df["AnnVol12M"] = display_df["AnnVol12M"].apply(
                lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A"
            )
        if "Sharpe12M" in display_df.columns:
            display_df["Sharpe12M"] = display_df["Sharpe12M"].apply(
                lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
            )
        
        # Reset index to remove the index column
        display_df = display_df.reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab_liq:
    st.header("Liquidity Profile")
    
    # Liquidity KPIs
    liq_kpis = an.liquidity_kpis(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Weighted ADV Cover", f"{liq_kpis['weighted_adv_cover']:.1f}" if not np.isnan(liq_kpis['weighted_adv_cover']) else "N/A")
    col2.metric("% GMV Unwindable <5d", f"{liq_kpis['pct_unwind_lt5d']:.1%}" if not np.isnan(liq_kpis['pct_unwind_lt5d']) else "N/A")
    col3.metric("% GMV Illiquid >20d", f"{liq_kpis['pct_illiquid_gt20d']:.1%}" if not np.isnan(liq_kpis['pct_illiquid_gt20d']) else "N/A")
    col4.metric("Liquidity Trend", liq_kpis['liquidity_trend'])
    
    st.markdown("---")
    
    # Visuals
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Liquidity Trend Over Time")
        trend_data = an.liquidity_trend_over_time(df)
        if not trend_data.empty:
            fig = px.line(trend_data, x="Date", y="Weighted_ADV_Cover", 
                         title="Weighted ADV Cover Over Time")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No liquidity trend data available")
    
    with col2:
        st.markdown("### Liquidity Buckets (Current)")
        bucket_data = an.liquidity_buckets(df, pd.DataFrame())
        if not bucket_data.empty:
            fig = px.bar(bucket_data, x="Bucket", y="Pct", 
                        title="GMV by Liquidity Bucket (%)",
                        labels={"Pct": "% of GMV"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No bucket data available")
    
    # Summary table
    st.markdown("### Position Liquidity Summary")
    if "ADV_COVER_GMV" in df.columns and "Date" in df.columns:
        latest = df[df["Date"] == df["Date"].max()].copy()
        
        # Add liquidity bucket
        if "ADV_COVER_GMV" in latest.columns:
            days = latest["ADV_COVER_GMV"].fillna(9999)
            latest["Liquidity_Bucket"] = pd.cut(days, bins=[-1, 1, 5, 20, 99999], 
                                                labels=["≤1d", "1–5d", "5–20d", ">20d"])
        
        # Aggregate by Account
        if "Account" in latest.columns:
            summary = latest.groupby(["Account", "Liquidity_Bucket"], observed=True).agg({
                "GMV": "sum",
                "ADV_COVER_GMV": "mean"
            }).reset_index()
            # Format GMV column with $ and thousands
            summary["GMV"] = summary["GMV"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(summary.sort_values("Account"), use_container_width=True, hide_index=True)
        else:
            latest["GMV"] = latest["GMV"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(latest[["Date", "GMV", "ADV_COVER_GMV", "Liquidity_Bucket"]].head(50), use_container_width=True, hide_index=True)
    else:
            st.info("Liquidity metrics (ADV_COVER_GMV) not available in this dataset")

# --- Risk & Attribution Tab ---
with tab_risk:
    st.header("Risk Exposure & Return Attribution")
    
    # --- Risk Exposure KPIs ---
    risk_kpis = an.risk_exposure_kpis(df)
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Top Region", f"{risk_kpis['top_region']}", f"{risk_kpis['top_region_pct']:.1f}% of NAV")
    kpi_col2.metric("Top Sector", f"{risk_kpis['top_sector']}", f"{risk_kpis['top_sector_pct']:.1f}% of NAV")
    kpi_col3.metric("Gross Leverage", f"{risk_kpis['gross_leverage']:.2f}x")
    kpi_col4.metric("Net Exposure", f"{risk_kpis['net_exposure']:.2f}x")
    
    st.markdown("---")
    
    # --- Risk Exposure Charts ---
    st.markdown("### Risk Exposures")
    
    # Sector bar chart
    sector_data = an.top_sector_exposures(df, top_n=6)
    if not sector_data.empty:
        # Sort by Abs_NMV value (descending) to match pie chart ordering
        sector_data_sorted = sector_data.sort_values("Abs_NMV", ascending=False).reset_index(drop=True)
        
        fig_sector = px.bar(sector_data_sorted, x="Sector", y="Net_NMV", 
                           title="Top 6 Sector Exposures (Net NMV, Sorted by Absolute Exposure)",
                           labels={"Net_NMV": "Net Exposure ($)"},
                           color="Net_NMV",
                           color_continuous_scale="RdBu",
                           color_continuous_midpoint=0)
        fig_sector.update_layout(height=350)
        st.plotly_chart(fig_sector, use_container_width=True)
    
    # Sector and Market Cap exposure pie charts
    col_pie1, col_pie2 = st.columns(2)
    
    with col_pie1:
        sector_pie_data = an.sector_exposure_pie(df)
        if not sector_pie_data.empty:
            # Use Sector_Label if available, otherwise use Sector
            names_col = "Sector_Label" if "Sector_Label" in sector_pie_data.columns else "Sector"
            fig_sector_pie = px.pie(sector_pie_data, values="Pct", names=names_col,
                                   title="Sector Exposure (% of NAV)")
            fig_sector_pie.update_layout(height=500)
            st.plotly_chart(fig_sector_pie, use_container_width=True)
        else:
            st.info("Sector exposure data not available")
    
    with col_pie2:
        market_cap_data = an.market_cap_exposure_pie(df)
        if not market_cap_data.empty:
            fig_market_cap = px.pie(market_cap_data, values="Pct", names="MarketCap",
                                   title="Market Cap Exposure (% of NAV)")
            fig_market_cap.update_layout(height=500)
            st.plotly_chart(fig_market_cap, use_container_width=True)
        else:
            st.info("Market cap exposure data not available")
    
    st.markdown("---")
    
    # --- Return Attribution ---
    st.markdown("### Return Attribution")
    
    # Top contributors by sector
    top_neg, top_pos = an.top_contributors(df, pd.DataFrame(), by_dimension="GICS_SECTOR_CLEAN", top_n=5)
    
    if not top_neg.empty and not top_pos.empty:
        col_top1, col_top2 = st.columns(2)
        
        with col_top1:
            st.markdown("**Top 5 Negative Contributors**")
            fig_neg = px.bar(top_neg, x="Bucket", y="Contribution",
                            title="By Sector (PNL)",
                            labels={"Contribution": "PnL ($)"},
                            color="Contribution",
                            color_continuous_scale="reds")
            fig_neg.update_layout(height=300)
            st.plotly_chart(fig_neg, use_container_width=True)
        
        with col_top2:
            st.markdown("**Top 5 Positive Contributors**")
            fig_pos = px.bar(top_pos, x="Bucket", y="Contribution",
                            title="By Sector (PNL)",
                            labels={"Contribution": "PnL ($)"},
                            color="Contribution",
                            color_continuous_scale="greens")
            fig_pos.update_layout(height=300)
            st.plotly_chart(fig_pos, use_container_width=True)
    
    # Contributors by market cap
    if "MARKET_CAP" in df.columns and "PNL" in df.columns and "Date" in df.columns:
        # Add market cap bucket column
        df_with_cap = df.copy()
        df_with_cap["MarketCap_Bucket"] = df_with_cap["MARKET_CAP"].apply(an.market_cap_classify)
        
        # Aggregate PNL by market cap bucket (for latest date)
        latest_cap = df_with_cap[df_with_cap["Date"] == df_with_cap["Date"].max()]
        cap_pnl = latest_cap.groupby("MarketCap_Bucket", observed=True)["PNL"].sum().reset_index()
        cap_pnl.columns = ["MarketCap", "PNL"]
        
        if not cap_pnl.empty:
            # Sort by PNL
            cap_pnl = cap_pnl.sort_values("PNL", ascending=True)
            
            fig_cap = px.bar(cap_pnl, x="MarketCap", y="PNL",
                           title="PNL by Market Cap Category",
                           labels={"PNL": "PnL ($)", "MarketCap": "Market Cap"},
                           color="PNL",
                           color_continuous_scale="RdBu",
                           color_continuous_midpoint=0)
            fig_cap.update_layout(height=300)
            st.plotly_chart(fig_cap, use_container_width=True)
    
    # Return by Sector (moved here from Return by Dimension section)
    if "GICS_SECTOR_CLEAN" in df.columns:
        sector_ret = an.return_by_dimension(df, "GICS_SECTOR_CLEAN")
        if not sector_ret.empty:
            fig_sector_ret = px.bar(sector_ret, x=sector_ret.columns[0], y="Return",
                                   title="Return by Sector (bps)",
                                   labels={"Return": "Return (bps)"},
                                   color="Return",
                                   color_continuous_scale="RdBu",
                                   color_continuous_midpoint=0)
            fig_sector_ret.update_layout(height=300)
            st.plotly_chart(fig_sector_ret, use_container_width=True)
    
    st.markdown("---")
    
    # Attribution table
    st.markdown("### Attribution Details")
    if "GICS_SECTOR_CLEAN" in df.columns and "PNL" in df.columns and "Date" in df.columns:
        latest_df = df[df["Date"] == df["Date"].max()]
        
        # Use same method as return_by_dimension
        sector_pnl = latest_df.groupby("GICS_SECTOR_CLEAN", observed=True).agg({
            "PNL": "sum",
            "NMV": lambda x: x.abs().sum()
        }).reset_index()
        
        total_nav = latest_df["NMV"].abs().sum()
        sector_pnl["Port_Wt"] = (sector_pnl["NMV"] / total_nav * 100).round(2)
        
        # Calculate return consistently with return_by_dimension: (PNL / abs(NMV_sum)) * 10000
        sector_pnl["Return"] = ((sector_pnl["PNL"] / sector_pnl["NMV"].clip(lower=1e-6)) * 10000).round(2)
        
        sector_pnl = sector_pnl.rename(columns={"GICS_SECTOR_CLEAN": "Sector", "PNL": "PnL"})
        sector_pnl = sector_pnl[["Sector", "Port_Wt", "Return", "PnL"]]
        
        # Format columns: Port_Wt with %, Return with bps, PnL with $
        sector_pnl["Port_Wt"] = sector_pnl["Port_Wt"].apply(lambda x: f"{x:.2f}%")
        sector_pnl["Return"] = sector_pnl["Return"].apply(lambda x: f"{x:.2f} bps")
        sector_pnl["PnL"] = sector_pnl["PnL"].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(sector_pnl.sort_values("PnL", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Attribution details not available")

with tab_reports:
    st.header("Reports & Downloads")
    
    # Executive Summary
    st.markdown("### Executive Summary")
    st.info("""
    This dashboard provides institutional-quality analytics for the TooSharpe fund portfolio. 
    All metrics are calculated using industry-standard methodologies and are updated in real-time based on your selected date range and analysis level.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Key Performance Metrics")
        st.markdown("""
        **Annualized Return**: Mean daily return multiplied by 252 trading days.
        
        **Volatility**: Annualized standard deviation of returns (√252 × daily std).
        
        **Sharpe Ratio**: Risk-adjusted return metric, comparing excess returns to volatility.
        
        **Max Drawdown**: Maximum peak-to-trough decline from any high point.
        
        **Cumulative Return**: Total return since the start of the selected period.
        """)
    
    with col2:
        st.markdown("#### Risk & Attribution")
        st.markdown("""
        **Tracking Error**: Measures deviation from benchmark performance.
        
        **Information Ratio**: Excess return per unit of tracking error.
        
        **Beta**: Sensitivity of portfolio returns to benchmark movements.
        
        **Alpha**: Excess return after adjusting for market risk (beta).
        """)
    
    st.markdown("---")
    
    # Download Section
    st.markdown("### Download Reports & Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Portfolio Summary Report")
        
        # Generate PDF report
        if HAS_REPORTLAB and not df.empty:
            has_data = len(df) > 0
            
            def generate_pdf_with_charts(_df, _benchmark, _as_of_date, _analysis_level, _sel_accounts, 
                                       _current_aum, _benchmark_path_str):
                """Generate PDF with all charts."""
                try:
                    # Get required data
                    account_kpis_for_pdf = an.perf_account_kpis(_df) if "Account" in _df.columns else pd.DataFrame()
                    risk_kpis_for_pdf = an.risk_exposure_kpis(_df)
                    liq_kpis_for_pdf = an.liquidity_kpis(_df)
                    bench_df_for_pdf = pd.read_csv(_benchmark_path_str) if Path(_benchmark_path_str).exists() else pd.DataFrame()
                    
                    # Generate charts
                    charts_dict = {}
                    if "RET_TWR" in _df.columns and Path(_benchmark_path_str).exists():
                        # Cumulative Returns - Use same method as app
                        try:
                            # Get daily returns in same format as app
                            daily_rets = _df.groupby("Date", observed=True)["RET_TWR"].mean().reset_index()
                            daily_rets.columns = ["Date", "Ret"]
                            daily_rets["Ret"] = daily_rets["Ret"].clip(-0.50, 0.50)
                            
                            # Use the same prepare_benchmark_timeseries function as app
                            ts_data = an.prepare_benchmark_timeseries(daily_rets, _benchmark_path_str)
                            
                            if not ts_data.empty:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=ts_data["Date"], y=ts_data["Account_CumReturn"] * 100,
                                                        mode='lines', name='Portfolio', line=dict(color='#1f77b4', width=2)))
                                
                                # Find the selected benchmark column
                                bench_cols = [c for c in ts_data.columns if c.endswith("_CumReturn") and c != "Account_CumReturn"]
                                selected_bench_col = f"{_benchmark}_CumReturn"
                                
                                # Check if selected benchmark exists
                                if selected_bench_col in ts_data.columns:
                                    bench_name = selected_bench_col.replace("_CumReturn", "")
                                    fig.add_trace(go.Scatter(x=ts_data["Date"], y=ts_data[selected_bench_col] * 100,
                                                            mode='lines', name=bench_name,
                                                            line=dict(color='#ff7f0e', width=2, dash='dash')))
                                elif bench_cols:
                                    # Fallback to first available benchmark
                                    bench_name = bench_cols[0].replace("_CumReturn", "")
                                    fig.add_trace(go.Scatter(x=ts_data["Date"], y=ts_data[bench_cols[0]] * 100,
                                                            mode='lines', name=bench_name,
                                                            line=dict(color='#ff7f0e', width=2, dash='dash')))
                                
                                fig.update_layout(
                                    title=f"Cumulative Returns: Portfolio vs {_benchmark}",
                                    xaxis_title="Date",
                                    yaxis_title="Cumulative Return (%)",
                                    height=400,
                                    showlegend=True,
                                    xaxis=dict(type='date', tickformat='%b %Y')
                                )
                                charts_dict['cumulative_returns'] = fig
                        except Exception as e:
                            pass
                    
                    # Generate Rolling 90-Day Excess Return chart
                    if "RET_TWR" in _df.columns and Path(_benchmark_path_str).exists():
                        _df_temp = _df.copy()
                        if "Date" in _df_temp.columns:
                            _df_temp["Date"] = pd.to_datetime(_df_temp["Date"], errors='coerce')
                            _df_temp = _df_temp.dropna(subset=["Date", "RET_TWR"])
                        
                        if not _df_temp.empty:
                            try:
                                bench_df = pd.read_csv(_benchmark_path_str)
                                if "Date" in bench_df.columns:
                                    bench_df["Date"] = pd.to_datetime(bench_df["Date"], errors='coerce')
                                
                                # Get excess return data using the analytics function
                                excess_data = an.rolling_excess_return(_df_temp, bench_df, window=90, benchmark_col=_benchmark)
                                
                                if not excess_data.empty and "RollingExcess" in excess_data.columns:
                                    # Ensure Date column is properly formatted
                                    if "Date" in excess_data.columns:
                                        excess_data["Date"] = pd.to_datetime(excess_data["Date"], errors='coerce')
                                        excess_data = excess_data.dropna(subset=["Date", "RollingExcess"])
                                        
                                        if not excess_data.empty:
                                            fig_excess = go.Figure()
                                            fig_excess.add_trace(go.Scatter(x=excess_data["Date"], y=excess_data["RollingExcess"],
                                                                            mode='lines', line=dict(color='#d62728', width=2)))
                                            fig_excess.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)
                                            fig_excess.update_layout(
                                                title="Rolling 90-Day Excess Return vs Benchmark",
                                                xaxis_title="Date",
                                                yaxis_title="Excess Return (bps)",
                                                height=350,
                                                showlegend=False,
                                                xaxis=dict(
                                                    type='date',
                                                    tickformat='%b %Y'
                                                )
                                            )
                                            charts_dict['rolling_excess_return'] = fig_excess
                            except Exception as e:
                                pass
                    
                    if "RET_TWR" in _df.columns:
                        # Rolling Volatility
                        _df_temp = _df.copy()
                        if "Date" in _df_temp.columns:
                            _df_temp["Date"] = pd.to_datetime(_df_temp["Date"])
                        
                        daily_rets = _df_temp.groupby("Date")["RET_TWR"].mean()
                        rolling_vol = daily_rets.rolling(90).std() * np.sqrt(252)
                        rolling_vol = rolling_vol.dropna()
                        
                        if not rolling_vol.empty and len(rolling_vol) > 0:
                            fig_vol = go.Figure()
                            fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                                                        mode='lines', line=dict(color='#2ca02c', width=2)))
                            fig_vol.update_layout(
                                title="Rolling 90-Day Volatility",
                                xaxis_title="Date",
                                yaxis_title="Rolling Volatility (90D)",
                                height=350,
                                showlegend=False,
                                xaxis=dict(
                                    type='date',
                                    tickformat='%b %Y'
                                )
                            )
                            charts_dict['rolling_volatility'] = fig_vol
                    
                    if "Account" in _df.columns:
                        # Return vs Volatility
                        account_ret_data = an.account_annualized_returns(_df)
                        if not account_ret_data.empty:
                            fig_scatter = px.scatter(account_ret_data, x="AnnVol", y="AnnReturn",
                                                    size=account_ret_data["AnnReturn"].abs() * 3,
                                                    color="Account",
                                                    color_discrete_sequence=px.colors.qualitative.Set3,
                                                    hover_name="Account",
                                                    title="Return vs Volatility by Account",
                                                    labels={"AnnReturn": "Annualized Return (%)",
                                                           "AnnVol": "Annualized Volatility (%)"},
                                                    size_max=50)
                            fig_scatter.update_traces(marker=dict(line=dict(width=1.5, color='DarkSlateGrey')))
                            fig_scatter.update_layout(height=400, showlegend=False)
                            charts_dict['return_vs_volatility'] = fig_scatter
                    
                    # 4. Sector Exposures (Risk)
                    if "GICS_SECTOR_CLEAN" in _df.columns and "NMV" in _df.columns:
                        sector_data = an.top_sector_exposures(_df, top_n=6)
                        if not sector_data.empty:
                            sector_data_sorted = sector_data.sort_values("Abs_NMV", ascending=False).reset_index(drop=True)
                            fig_sector = px.bar(sector_data_sorted, x="Sector", y="Net_NMV", 
                                             title="Top 6 Sector Exposures",
                                             labels={"Net_NMV": "Net Exposure ($)"},
                                             color="Net_NMV",
                                             color_continuous_scale="RdBu",
                                             color_continuous_midpoint=0)
                            fig_sector.update_layout(height=350, showlegend=False)
                            charts_dict['sector_exposures'] = fig_sector
                        
                        # Sector Exposure % of NAV
                        if "NMV" in _df.columns:
                            sector_nav = _df.groupby("GICS_SECTOR_CLEAN")["NMV"].sum().reset_index()
                            total_nav = _df["NMV"].abs().sum()
                            sector_nav["Pct"] = (sector_nav["NMV"] / total_nav * 100) if total_nav > 0 else 0
                            sector_nav = sector_nav.sort_values("Pct", ascending=False).head(6)
                            fig_sector_nav = px.bar(sector_nav, x="GICS_SECTOR_CLEAN", y="Pct",
                                                   title="Sector Exposure (% of NAV)",
                                                   labels={"GICS_SECTOR_CLEAN": "Sector", "Pct": "% of NAV"})
                            fig_sector_nav.update_layout(height=300, showlegend=False)
                            charts_dict['sector_nav'] = fig_sector_nav
                    
                    # Market Cap Exposure % of NAV
                    if "MARKET_CAP_GROUPS" in _df.columns and "NMV" in _df.columns:
                        cap_nav = _df.groupby("MARKET_CAP_GROUPS")["NMV"].sum().reset_index()
                        total_nav = _df["NMV"].abs().sum()
                        cap_nav["Pct"] = (cap_nav["NMV"] / total_nav * 100) if total_nav > 0 else 0
                        fig_cap_nav = px.bar(cap_nav, x="MARKET_CAP_GROUPS", y="Pct",
                                           title="Market Cap Exposure (% of NAV)",
                                           labels={"MARKET_CAP_GROUPS": "Market Cap", "Pct": "% of NAV"})
                        fig_cap_nav.update_layout(height=300, showlegend=False)
                        charts_dict['market_cap_nav'] = fig_cap_nav
                    
                    # 5. Liquidity Trend
                    if "ADV_COVER_GMV" in _df.columns and "GMV" in _df.columns and "Date" in _df.columns:
                        # Ensure Date is datetime before calling function
                        _df_liq = _df.copy()
                        if "Date" in _df_liq.columns:
                            _df_liq["Date"] = pd.to_datetime(_df_liq["Date"])
                        
                        trend_data = an.liquidity_trend_over_time(_df_liq)
                        if not trend_data.empty:
                            # Force convert Date to datetime
                            if "Date" in trend_data.columns and "Weighted_ADV_Cover" in trend_data.columns:
                                trend_data["Date"] = pd.to_datetime(trend_data["Date"], errors='coerce')
                                trend_data = trend_data.dropna(subset=["Date", "Weighted_ADV_Cover"])
                            
                                if not trend_data.empty:
                                    fig_liq = go.Figure()
                                    fig_liq.add_trace(go.Scatter(x=trend_data["Date"], y=trend_data["Weighted_ADV_Cover"],
                                                                mode='lines', line=dict(color='#9467bd', width=2)))
                                    fig_liq.update_layout(
                                        title="Liquidity Trend Over Time",
                                        xaxis_title="Date",
                                        yaxis_title="Weighted ADV Cover",
                                        height=350,
                                        showlegend=False,
                                        xaxis=dict(
                                            type='date',
                                            tickformat='%b %Y'
                                        )
                                    )
                                    charts_dict['liquidity_trend'] = fig_liq
                    
                    # 6. Return by Sector (Return Attribution)
                    if "GICS_SECTOR_CLEAN" in _df.columns and "RET_TWR" in _df.columns and "NMV" in _df.columns:
                        # Calculate sector returns
                        sector_returns = []
                        for sector in _df["GICS_SECTOR_CLEAN"].dropna().unique():
                            sector_df = _df[_df["GICS_SECTOR_CLEAN"] == sector]
                            sector_cum_ret = (1 + sector_df["RET_TWR"]).prod() - 1 if len(sector_df) > 0 else 0
                            sector_returns.append({"Sector": sector, "Return": sector_cum_ret * 10000})  # in bps
                        
                        if sector_returns:
                            sector_returns_df = pd.DataFrame(sector_returns).sort_values("Return", ascending=False).head(6)
                            fig_return_sector = px.bar(sector_returns_df, x="Sector", y="Return",
                                                      title="Return by Sector (bps)",
                                                      labels={"Return": "Return (bps)"})
                            fig_return_sector.update_layout(height=300, showlegend=False)
                            charts_dict['return_by_sector'] = fig_return_sector
                    
                    # 7. Liquidity Buckets
                    if "ADV_COVER_GMV" in _df.columns and "GMV" in _df.columns:
                        bucket_data = an.liquidity_buckets(_df, pd.DataFrame())
                        if not bucket_data.empty:
                            fig_buckets = px.bar(bucket_data, x="Bucket", y="Pct", 
                                                title="GMV by Liquidity Bucket",
                                                labels={"Pct": "% of GMV"})
                            fig_buckets.update_layout(height=300, showlegend=False)
                            charts_dict['liquidity_buckets'] = fig_buckets
                    
                    # Prepare Position Liquidity Summary
                    position_liquidity_summary_for_pdf = pd.DataFrame()
                    if "Account" in _df.columns and "Liquidity_Bucket" in _df.columns and "GMV" in _df.columns:
                        position_liquidity_summary_for_pdf = _df.groupby(["Account", "Liquidity_Bucket"]).agg({
                            "GMV": "sum",
                            "ADV_COVER_GMV": "mean"
                        }).reset_index()
                        position_liquidity_summary_for_pdf.columns = ["Account", "Liquidity Bucket", "GMV", "ADV Cover"]
                    
                    # Generate PDF
                    return generate_pdf_report(
                        df=_df,
                        benchmark=_benchmark,
                        as_of_date=_as_of_date,
                        analysis_level=_analysis_level,
                        sel_accounts=_sel_accounts,
                        account_kpis_df=account_kpis_for_pdf,
                        risk_kpis=risk_kpis_for_pdf,
                        liq_kpis=liq_kpis_for_pdf,
                        daily_aum_value=_current_aum * 1e6 if _current_aum is not None else 0,
                        bench_df=bench_df_for_pdf,
                        charts=charts_dict if charts_dict else None,
                        position_liquidity_summary=position_liquidity_summary_for_pdf
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return None
            
            # Initialize session state for PDF
            if 'pdf_generated' not in st.session_state:
                st.session_state['pdf_generated'] = False
            if 'pdf_bytes' not in st.session_state:
                st.session_state['pdf_bytes'] = b""
            if 'pdf_filename' not in st.session_state:
                st.session_state['pdf_filename'] = ""
            
            # Generate filename based on analysis level
            if has_data:
                if analysis_level == "Account Level" and sel_accounts:
                    if len(sel_accounts) == 1:
                        filename = f"TooSharpe_{sel_accounts[0]}_{as_of_date.strftime('%Y%m%d')}.pdf"
                    else:
                        filename = f"TooSharpe_{len(sel_accounts)}Accounts_{as_of_date.strftime('%Y%m%d')}.pdf"
                else:
                    filename = f"TooSharpe_Fund_{as_of_date.strftime('%Y%m%d')}.pdf"
            else:
                filename = f"toosharpe_report_{as_of_date.strftime('%Y%m%d')}.pdf"
            
            # Button to generate PDF
            if st.button("📄 Generate PDF Report", disabled=not has_data, key="generate_pdf_button"):
                st.session_state['pdf_generated'] = False
                with st.spinner("Generating PDF report..."):
                    pdf_data = generate_pdf_with_charts(
                        df, benchmark, as_of_date, analysis_level, sel_accounts,
                        current_aum, str(benchmark_path) if benchmark_path.exists() else ""
                    )
                    if pdf_data:
                        st.session_state['pdf_bytes'] = pdf_data
                        st.session_state['pdf_filename'] = filename
                        st.session_state['pdf_generated'] = True
                        st.success("PDF report generated successfully!")
                    else:
                        st.error("Failed to generate PDF report.")
                        st.session_state['pdf_generated'] = False
            
            # Download button (shown if PDF is generated)
            if st.session_state['pdf_generated'] and st.session_state['pdf_bytes']:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state['pdf_bytes'],
                    file_name=st.session_state['pdf_filename'],
                    mime="application/pdf",
                    help="Click to download the generated PDF report"
                )
            else:
                st.info("Click 'Generate PDF Report' button above to create your PDF report.")
        else:
            st.download_button(
                label="📄 Download PDF Report",
                data=b"",
                file_name=f"toosharpe_report_{as_of_date.strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                disabled=True,
                help="ReportLab library required for PDF generation. Please install: pip install reportlab"
            )
        
        # Account Performance Summary CSV
        st.markdown("#### Account Performance Data")
        if "Account" in df.columns:
            account_summary = an.perf_account_kpis(df)
            if not account_summary.empty:
                csv_account = account_summary.to_csv(index=False)
                st.download_button(
                    label="📊 Download Account Summary CSV",
                    data=csv_account,
                    file_name=f"account_performance_{as_of_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        st.markdown("#### Position Data")
        
        # Full intermediary dataset
        csv_full = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Position Data CSV",
            data=csv_full,
            file_name=f"position_data_{as_of_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("#### Attribution Details")
        # Download attribution details if available
        if "GICS_SECTOR_CLEAN" in df.columns and "PNL" in df.columns and "Date" in df.columns:
            latest_df = df[df["Date"] == df["Date"].max()]
            sector_pnl = latest_df.groupby("GICS_SECTOR_CLEAN", observed=True).agg({
                "PNL": "sum",
                "NMV": lambda x: x.abs().sum()
            }).reset_index()
            total_nav = latest_df["NMV"].abs().sum()
            sector_pnl["Port_Wt"] = (sector_pnl["NMV"] / total_nav * 100).round(2)
            sector_pnl["Return"] = ((sector_pnl["PNL"] / sector_pnl["NMV"].clip(lower=1e-6)) * 10000).round(2)
            sector_pnl = sector_pnl.rename(columns={"GICS_SECTOR_CLEAN": "Sector", "PNL": "PnL"})
            sector_pnl = sector_pnl[["Sector", "Port_Wt", "Return", "PnL"]]
            csv_attrib = sector_pnl.to_csv(index=False)
            st.download_button(
                label="📋 Download Attribution CSV",
                data=csv_attrib,
                file_name=f"attribution_{as_of_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    
    # Data Information
    st.markdown("### Data Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Date Range")
        if not df.empty and "Date" in df.columns:
            st.text(f"Analysis Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
            st.text(f"Total Observations: {len(df):,}")
        
    with col2:
        st.markdown("#### Portfolio Composition")
        if "Account" in df.columns:
            num_accounts = df["Account"].nunique()
            st.text(f"Number of Accounts: {num_accounts}")
        if "SEDOL" in df.columns:
            num_positions = df["SEDOL"].nunique()
            st.text(f"Unique Positions: {num_positions}")
