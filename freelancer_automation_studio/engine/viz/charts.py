import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pandas as pd
from typing import List
from loguru import logger

class ChartGenerator:
    """Generate interactive Plotly visualizations"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Plotly
        
    def distribution_plot(self, df: pl.DataFrame, column: str):
        """Create distribution plot"""
        try:
            pdf = df.to_pandas()
            
            if df[column].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                # Numeric: histogram
                fig = px.histogram(
                    pdf,
                    x=column,
                    title=f"Distribution of {column}",
                    nbins=50,
                    color_discrete_sequence=[self.color_palette[0]]
                )
                fig.update_traces(marker_line_width=1, marker_line_color='white')
            else:
                # Categorical: bar chart
                value_counts = df[column].value_counts().to_pandas()
                fig = px.bar(
                    value_counts,
                    x=column,
                    y='counts',
                    title=f"Distribution of {column}",
                    color_discrete_sequence=[self.color_palette[1]]
                )
            
            fig.update_layout(
                template='plotly_white',
                title_font_size=18,
                height=500
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating distribution plot: {e}")
            return go.Figure()
    
    def correlation_heatmap(self, df: pl.DataFrame):
        """Create correlation heatmap"""
        try:
            # Select numeric columns
            numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
            
            if len(numeric_cols) < 2:
                logger.warning("Not enough numeric columns for correlation")
                return go.Figure()
            
            # Calculate correlation
            pdf = df.select(numeric_cols).to_pandas()
            corr_matrix = pdf.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Correlation Heatmap",
                template='plotly_white',
                height=600,
                width=700
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return go.Figure()
    
    def time_series_plot(self, df: pl.DataFrame, date_col: str, value_col: str):
        """Create time series plot"""
        try:
            pdf = df.select([date_col, value_col]).to_pandas()
            
            # Try to convert to datetime
            try:
                pdf[date_col] = pd.to_datetime(pdf[date_col])
            except:
                logger.warning(f"Could not convert {date_col} to datetime")
            
            pdf = pdf.sort_values(date_col)
            
            fig = px.line(
                pdf,
                x=date_col,
                y=value_col,
                title=f"Time Series: {value_col}",
                color_discrete_sequence=[self.color_palette[2]]
            )
            
            fig.update_traces(line_width=2)
            fig.update_layout(
                template='plotly_white',
                title_font_size=18,
                height=500,
                hovermode='x unified'
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            return go.Figure()
    
    def scatter_matrix(self, df: pl.DataFrame, columns: List[str]):
        """Create scatter matrix"""
        try:
            pdf = df.select(columns).to_pandas()
            
            fig = px.scatter_matrix(
                pdf,
                dimensions=columns,
                title="Scatter Matrix",
                color_discrete_sequence=[self.color_palette[3]]
            )
            
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            fig.update_layout(
                template='plotly_white',
                height=700,
                width=800
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating scatter matrix: {e}")
            return go.Figure()
    
    def box_plot(self, df: pl.DataFrame, column: str):
        """Create box plot"""
        try:
            pdf = df.to_pandas()
            
            fig = px.box(
                pdf,
                y=column,
                title=f"Box Plot: {column}",
                color_discrete_sequence=[self.color_palette[4]]
            )
            
            fig.update_layout(
                template='plotly_white',
                title_font_size=18,
                height=500
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating box plot: {e}")
            return go.Figure()