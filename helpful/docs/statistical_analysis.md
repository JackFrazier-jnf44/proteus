# Statistical Analysis and Visualization

This document describes the statistical analysis and visualization capabilities of the multi-model analysis framework.

## Effect Size Measures

The framework supports multiple effect size measures for comparing model versions:

1. **Cohen's d**: Standardized mean difference using pooled standard deviation
2. **Hedges' g**: Corrected Cohen's d for small sample sizes
3. **Glass's delta**: Standardized mean difference using control group standard deviation
4. **Cliff's delta**: Non-parametric effect size measure

## Statistical Tests

The following statistical tests are performed for each metric:

1. **Parametric Tests**:
   - Independent samples t-test

2. **Non-parametric Tests**:
   - Wilcoxon signed-rank test
   - Mann-Whitney U test
   - Kolmogorov-Smirnov test
   - Chi-square test (for categorical data)

## Meta-Analysis

The framework performs meta-analysis across different effect size measures:

1. **Fixed Effects**: Combines effect sizes assuming homogeneity
2. **Random Effects**: Accounts for heterogeneity between studies
3. **Heterogeneity Analysis**: Measures variation between effect sizes
4. **Publication Bias Assessment**: Evaluates potential bias in effect size estimates

## Aggregated Metrics

The following aggregated metrics are calculated:

1. **Mean Difference**: Average difference between versions
2. **Median Difference**: Median difference between versions
3. **Standard Deviation Difference**: Difference in spread
4. **IQR Difference**: Difference in interquartile range
5. **Effect Size Consistency**: Agreement between different effect size measures

## Visualization

The framework provides several visualization options:

1. **Effect Size Forest Plot**:
   - Compares different effect size measures across metrics
   - Shows confidence intervals and significance

2. **Meta-Analysis Forest Plot**:
   - Displays fixed and random effects
   - Shows heterogeneity between studies

3. **Statistical Significance Heatmap**:
   - Visualizes p-values across different tests
   - Uses -log10(p-value) for better visualization

4. **Aggregated Metrics Plot**:
   - Shows multiple aggregated metrics
   - Facilitates comparison across different measures

5. **Confidence Intervals Plot**:
   - Displays confidence intervals for each metric
   - Shows point estimates and uncertainty

## Usage Example

```python
from src.utils.model_versioning import ModelVersionManager
from src.utils.statistical_visualization import (
    plot_effect_sizes,
    plot_meta_analysis,
    plot_statistical_significance,
    plot_aggregated_metrics,
    plot_confidence_intervals,
    VisualizationConfig
)

# Initialize version manager
version_manager = ModelVersionManager("path/to/models")

# Compare versions
comparison = version_manager.compare_versions("version1", "version2")

# Create visualizations
config = VisualizationConfig(
    figure_size=(15, 10),
    dpi=300,
    style='seaborn',
    color_palette='Set2'
)

# Generate plots
plot_effect_sizes(comparison, "output_dir", config)
plot_meta_analysis(comparison, "output_dir", config)
plot_statistical_significance(comparison, "output_dir", config)
plot_aggregated_metrics(comparison, "output_dir", config)
plot_confidence_intervals(comparison, "output_dir", config)
```

## Interpretation Guidelines

1. **Effect Sizes**:
   - Cohen's d/Hedges' g: |d| < 0.2 (small), |d| < 0.5 (medium), |d| < 0.8 (large)
   - Glass's delta: Similar interpretation to Cohen's d
   - Cliff's delta: |d| < 0.147 (small), |d| < 0.33 (medium), |d| < 0.474 (large)

2. **Statistical Significance**:
   - p < 0.05: Statistically significant
   - p < 0.01: Highly significant
   - p < 0.001: Very highly significant

3. **Meta-Analysis**:
   - Fixed vs Random effects: Choose based on heterogeneity
   - Publication bias: Consider if effect sizes differ significantly

4. **Confidence Intervals**:
   - Wider intervals indicate more uncertainty
   - Intervals crossing zero suggest non-significant differences

## Best Practices

1. **Sample Size**:
   - Use appropriate sample sizes for statistical power
   - Consider effect size when planning comparisons

2. **Multiple Testing**:
   - Consider using multiple testing corrections
   - Interpret results in context of all comparisons

3. **Visualization**:
   - Choose appropriate plot types for your data
   - Use consistent scales and colors
   - Include clear labels and legends

4. **Reporting**:
   - Report all effect sizes and tests performed
   - Include confidence intervals
   - Document any assumptions or limitations
