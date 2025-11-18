# Synthetic Population Visualization Dashboard

This document describes the visualization capabilities available in the voxpop-crew repository for analyzing and validating synthetic population data.

**See [example_plots.md](example_plots.md) for actual example visualizations.**

## Overview

The `geo_tools/plot_synthetic.py` module provides comprehensive visualization tools for synthetic personas. These visualizations help validate the distribution and characteristics of synthetic populations compared to original census data.

## Available Visualization Functions

### 1. Selected Regions Plot

**Function**: `plot_selected_regions()`

**Purpose**: Visualize Australian Statistical Area Level 2 (SA2) regions on a map.

**Use Cases**:
- Validate geographic coverage of synthetic data
- Understand regional boundaries
- Plan persona distribution

**Output**:
- Map showing SA2 boundaries
- Region labels with names
- Legend showing number of selected regions

**Example Usage**:
```python
from geo_tools.plot_synthetic import plot_selected_regions

sa2_codes = ['115011290', '115011291', '115011292']
plot = plot_selected_regions(sa2_codes, title="Selected SA2 Regions")
plot.save('selected_regions.png')
```

**Example Output**: See [example_plots.md](example_plots.md#1-selected-regions-plot) for a visual example.

### 2. Personas on Map

**Function**: `plot_personas_on_map()`

**Purpose**: Plot synthetic personas as points on a map with optional SA2 boundaries.

**Use Cases**:
- Visualize persona distribution across geographic areas
- Validate spatial distribution matches expectations
- Identify coverage gaps or clustering

**Output**:
- Scatter plot of persona locations (lat/lon)
- Optional SA2 boundary overlays
- Legend showing persona count

**Key Features**:
- Configurable marker size and color
- Optional boundary visualization
- Geographic coordinate display

**Example Usage**:
```python
from geo_tools.plot_synthetic import plot_personas_on_map
import pandas as pd

personas_df = pd.DataFrame({
    'lat': [-33.8688, -33.8705, -33.8650],
    'lon': [151.2093, 151.2110, 151.2070],
    'sa2_code': ['115011290', '115011290', '115011291']
})

plot = plot_personas_on_map(
    personas_df,
    sa2_codes=['115011290', '115011291'],
    title="Synthetic Personas Distribution"
)
plot.save('personas_map.png')
```

**Example Output**: See [example_plots.md](example_plots.md#2-personas-on-map) for a visual example.

### 3. Multi-Region Comparison

**Function**: `plot_multi_region_comparison()`

**Purpose**: Compare persona distributions across multiple regions side-by-side.

**Use Cases**:
- Compare synthetic population characteristics across regions
- Validate consistency of generation methodology
- Analyze regional differences

**Output**:
- Grid of subplots (up to 3 columns)
- Each subplot shows personas for one region
- Color-coded by region

**Example Usage**:
```python
from geo_tools.plot_synthetic import plot_multi_region_comparison

region_data = {
    'Region A': personas_df_a,
    'Region B': personas_df_b,
    'Region C': personas_df_c
}

plot = plot_multi_region_comparison(
    region_data,
    title="Multi-Region Persona Comparison"
)
plot.save('multi_region_comparison.png')
```

**Example Output**: See [example_plots.md](example_plots.md#3-multi-region-comparison) for a visual example.

### 4. Demographic Heatmap

**Function**: `create_demographic_heatmap()`

**Purpose**: Create a heatmap showing demographic distribution across SA2 regions.

**Use Cases**:
- Validate demographic distributions match census data
- Identify demographic patterns across regions
- Compare synthetic vs. original distributions

**Output**:
- Choropleth map colored by demographic variable
- Color scale showing value ranges
- Statistical aggregation by SA2 region

**Demographic Variables**:
- Age distributions
- Income levels
- Household sizes
- Any numeric persona attribute

**Example Usage**:
```python
from geo_tools.plot_synthetic import create_demographic_heatmap

plot = create_demographic_heatmap(
    personas_df,
    sa2_codes=['115011290', '115011291'],
    demographic_column='age_band',
    title="Age Distribution Across Regions"
)
plot.save('age_heatmap.png')
```

### 5. Regions with Highlighted Personas

**Function**: `plot_regions_with_highlighted_personas()`

**Purpose**: Show all SA2 regions with highlighted regions that contain personas.

**Use Cases**:
- Visualize coverage of synthetic population
- Identify regions with and without personas
- Understand geographic distribution patterns

**Output**:
- Background map showing all SA2 regions (light shading)
- Highlighted regions containing personas (darker shading)
- Persona points overlaid on map
- Legend distinguishing region types

**Example Usage**:
```python
from geo_tools.plot_synthetic import plot_regions_with_highlighted_personas

all_sa2_codes = ['115011290', '115011291', '115011292', '115011293']
persona_sa2_codes = ['115011290', '115011291']

plot = plot_regions_with_highlighted_personas(
    all_sa2_codes,
    persona_sa2_codes,
    personas_df,
    title="SA2 Regions with Personas Highlighted"
)
plot.save('highlighted_regions.png')
```

**Example Output**: See [example_plots.md](example_plots.md#4-regions-with-highlighted-personas) for a visual example.

## Validation Use Cases

### Comparing Synthetic vs. Original Data

While the visualization functions don't directly compare synthetic vs. original data, they enable validation through:

1. **Distribution Validation**: Compare persona distributions across regions to expected census distributions
2. **Geographic Coverage**: Verify personas are distributed across expected geographic areas
3. **Demographic Patterns**: Validate demographic heatmaps match census patterns
4. **Spatial Clustering**: Check for unrealistic clustering or gaps

### Diagnostic Plots

These visualizations serve as diagnostic tools to:

- **Validate Generation Quality**: Ensure synthetic personas are distributed realistically
- **Identify Issues**: Spot anomalies in distribution or demographics
- **Document Coverage**: Show which regions have synthetic data
- **Support Analysis**: Provide visual context for statistical analysis

## Technical Details

### Dependencies

- **geopandas**: Geographic data processing
- **matplotlib**: Plotting and visualization
- **pandas**: Data manipulation
- **numpy**: Numerical operations

### Data Requirements

- **SA2 Shapefiles**: Australian Bureau of Statistics SA2 boundary files
- **Persona Data**: DataFrame with columns:
  - `lat`: Latitude
  - `lon`: Longitude
  - `sa2_code`: SA2 region code
  - Additional demographic columns for heatmaps

### Configuration

Plot styling can be configured through:
- `PLOT_STYLE_CONFIG`: Default styling settings
- Function parameters: Override defaults per plot
- Matplotlib customization: Direct matplotlib access

## Integration with Analysis Workflow

These visualizations integrate into the broader synthetic population analysis workflow:

1. **Generation**: Generate synthetic personas
2. **Visualization**: Create diagnostic plots
3. **Validation**: Compare visual patterns to expectations
4. **Refinement**: Adjust generation parameters if needed
5. **Documentation**: Include plots in analysis reports

## Privacy Considerations

- All visualizations use synthetic data only
- No real personal information is displayed
- Geographic boundaries are public ABS data
- Aggregated statistics only (no individual identification)

## Future Enhancements

Potential additions to the visualization suite:

- Interactive maps (Plotly/Bokeh)
- Statistical comparison overlays
- Time-series visualizations
- Export to web dashboards
- Automated validation reports

---

**Note**: These visualizations are diagnostic tools for validating synthetic population generation. They help ensure synthetic data maintains statistical properties similar to original census data while preserving privacy.

