#!/usr/bin/env python3
"""
Script to generate example visualization plots for AUSTRAC work sample documentation.

This script creates example plots demonstrating the visualization capabilities
described in visualization_overview.md.

Requirements:
- geopandas
- matplotlib
- pandas
- numpy

Note: This script requires SA2 shapefiles and may need to be run from the
voxpop-crew repository directory where geo_tools is available.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to access voxpop-crew if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "voxpop-crew"))

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    print("✓ Basic dependencies available")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("Please install: pip install matplotlib pandas numpy")
    sys.exit(1)

try:
    import geopandas as gpd
    print("✓ Geopandas available")
    GEOPANDAS_AVAILABLE = True
except ImportError:
    print("⚠ Geopandas not available - geographic plots will be limited")
    GEOPANDAS_AVAILABLE = False

def create_simple_example_plots():
    """Create simple example plots that don't require shapefiles."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Simple scatter plot (personas on map - simplified)
    print("\nGenerating example plots...")
    
    # Create sample persona data
    np.random.seed(42)
    n_personas = 50
    
    # Sydney area coordinates
    base_lat = -33.8688
    base_lon = 151.2093
    
    personas_df = pd.DataFrame({
        'lat': base_lat + np.random.normal(0, 0.05, n_personas),
        'lon': base_lon + np.random.normal(0, 0.05, n_personas),
        'age': np.random.choice(['25-34', '35-44', '45-54', '55-64'], n_personas),
        'income': np.random.choice(['Low', 'Medium', 'High'], n_personas)
    })
    
    # Plot 1: Personas scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        personas_df['lon'],
        personas_df['lat'],
        c=range(len(personas_df)),
        cmap='viridis',
        s=50,
        alpha=0.6,
        edgecolors='white',
        linewidth=0.5
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Example: Synthetic Personas Distribution\n(50 personas in Sydney area)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Persona Index')
    plt.tight_layout()
    plt.savefig(output_dir / 'example_personas_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: example_personas_scatter.png")
    
    # Plot 2: Demographic distribution bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Age distribution
    age_counts = personas_df['age'].value_counts().sort_index()
    ax1.bar(age_counts.index, age_counts.values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Count')
    ax1.set_title('Age Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Income distribution
    income_counts = personas_df['income'].value_counts()
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    ax2.bar(income_counts.index, income_counts.values, color=colors[:len(income_counts)], alpha=0.7)
    ax2.set_xlabel('Income Level')
    ax2.set_ylabel('Count')
    ax2.set_title('Income Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Example: Demographic Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'example_demographic_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: example_demographic_distributions.png")
    
    # Plot 3: Multi-region comparison (simplified)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    regions = ['Region A', 'Region B', 'Region C']
    colors_region = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (ax, region, color) in enumerate(zip(axes, regions, colors_region)):
        # Generate sample data for each region
        n = np.random.randint(15, 25)
        lat = base_lat + np.random.normal(0, 0.03, n) + idx * 0.1
        lon = base_lon + np.random.normal(0, 0.03, n) + idx * 0.1
        
        ax.scatter(lon, lat, c=color, s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.set_title(f'{region}\n({n} personas)', fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Example: Multi-Region Persona Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'example_multi_region_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created: example_multi_region_comparison.png")
    
    print(f"\n✓ Generated example plots in: {output_dir}")
    print("\nNote: These are simplified examples. Full geographic visualizations")
    print("with SA2 boundaries require shapefiles and can be generated using")
    print("the geo_tools/plot_synthetic.py functions in the voxpop-crew repository.")

if __name__ == "__main__":
    create_simple_example_plots()

