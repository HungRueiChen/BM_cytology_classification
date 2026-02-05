"""
Compare Ablation Study Results
Compares performance between quality-based and random tile selection approaches.
Generates comparison tables, visualizations, and statistical significance tests.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats

def load_metrics(results_dir):
    """
    Load test metrics from a results directory.
    
    Args:
        results_dir: Path to directory containing test_metrics.json and test_metrics.csv
        
    Returns:
        Dictionary with metrics data
    """
    results_dir = Path(results_dir)
    
    # Load JSON with detailed metrics
    with open(results_dir / 'test_metrics.json', 'r') as f:
        metrics_json = json.load(f)
    
    # Load CSV with summary metrics
    metrics_df = pd.read_csv(results_dir / 'test_metrics.csv', index_col=0)
    
    return {
        'json': metrics_json,
        'df': metrics_df,
        'dir': results_dir
    }

def create_comparison_table(quality_metrics, random_metrics, output_dir):
    """
    Create a comparison table showing metrics side-by-side.
    """
    quality_df = quality_metrics['df']
    random_df = random_metrics['df']
    
    # Create comparison dataframe
    comparison_data = []
    
    for method in quality_df.index:
        if method in random_df.index:
            for metric in quality_df.columns:
                quality_val = quality_df.loc[method, metric]
                random_val = random_df.loc[method, metric]
                diff = quality_val - random_val
                pct_diff = (diff / random_val * 100) if random_val != 0 else 0
                
                comparison_data.append({
                    'Method': method,
                    'Metric': metric,
                    'Quality-Based': quality_val,
                    'Random': random_val,
                    'Difference': diff,
                    'Improvement (%)': pct_diff
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    output_path = Path(output_dir) / 'comparison_table.csv'
    comparison_df.to_csv(output_path, index=False)
    print(f"Saved comparison table to {output_path}")
    
    return comparison_df

def plot_metric_comparison(comparison_df, output_dir):
    """
    Create bar plots comparing metrics between approaches.
    """
    methods = comparison_df['Method'].unique()
    metrics = comparison_df['Metric'].unique()
    
    # Create subplots for each method
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 4 * n_methods))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, method in enumerate(methods):
        method_data = comparison_df[comparison_df['Method'] == method]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        quality_vals = method_data['Quality-Based'].values
        random_vals = method_data['Random'].values
        
        axes[idx].bar(x - width/2, quality_vals, width, label='Quality-Based', color='steelblue')
        axes[idx].bar(x + width/2, random_vals, width, label='Random', color='coral')
        
        axes[idx].set_xlabel('Metrics', fontsize=12, fontfamily='serif')
        axes[idx].set_ylabel('Score', fontsize=12, fontfamily='serif')
        axes[idx].set_title(f'{method} - Metric Comparison', fontsize=14, fontfamily='serif')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(metrics, rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'metric_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metric comparison plot to {output_path}")

def plot_improvement_heatmap(comparison_df, output_dir):
    """
    Create a heatmap showing percentage improvement of quality-based over random.
    """
    # Pivot the data
    pivot_data = comparison_df.pivot(index='Method', columns='Metric', values='Improvement (%)')
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Improvement (%)'})
    plt.title('Quality-Based vs Random: Percentage Improvement', fontsize=14, fontfamily='serif')
    plt.xlabel('Metric', fontsize=12, fontfamily='serif')
    plt.ylabel('Method', fontsize=12, fontfamily='serif')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'improvement_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved improvement heatmap to {output_path}")

def compare_roc_curves(quality_metrics, random_metrics, output_dir):
    """
    Compare ROC curves between quality-based and random approaches.
    """
    methods = ['Tile', 'Patient_weighted_average', 'Patient_weighted_sum', 'Patient_mode']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, method in enumerate(methods):
        if method not in quality_metrics['json'] or method not in random_metrics['json']:
            continue
        
        quality_roc = quality_metrics['json'][method]['roc_data']['micro']
        random_roc = random_metrics['json'][method]['roc_data']['micro']
        
        # Plot ROC curves
        axes[idx].plot(quality_roc['fpr'], quality_roc['tpr'], 
                      label=f"Quality-Based (AUC={quality_roc['auc']:.3f})",
                      color='steelblue', linewidth=2)
        axes[idx].plot(random_roc['fpr'], random_roc['tpr'],
                      label=f"Random (AUC={random_roc['auc']:.3f})",
                      color='coral', linewidth=2)
        axes[idx].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        axes[idx].set_xlim([-0.05, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel('False Positive Rate', fontsize=11, fontfamily='serif')
        axes[idx].set_ylabel('True Positive Rate', fontsize=11, fontfamily='serif')
        axes[idx].set_title(f'{method}', fontsize=12, fontfamily='serif')
        axes[idx].legend(loc='lower right', fontsize=9, prop={'family':'serif'})
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('ROC Curve Comparison: Quality-Based vs Random', fontsize=14, fontfamily='serif')
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'roc_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC comparison to {output_path}")

def statistical_significance_test(comparison_df, output_dir):
    """
    Perform statistical tests to determine if differences are significant.
    Uses paired t-test for metrics.
    """
    results = []
    
    for method in comparison_df['Method'].unique():
        method_data = comparison_df[comparison_df['Method'] == method]
        
        quality_vals = method_data['Quality-Based'].values
        random_vals = method_data['Random'].values
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(quality_vals, random_vals)
        
        # Effect size (Cohen's d)
        diff = quality_vals - random_vals
        pooled_std = np.sqrt((np.std(quality_vals)**2 + np.std(random_vals)**2) / 2)
        cohens_d = np.mean(diff) / pooled_std if pooled_std != 0 else 0
        
        results.append({
            'Method': method,
            'T-statistic': t_stat,
            'P-value': p_value,
            'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No',
            'Cohens_d': cohens_d,
            'Effect_Size': 'Large' if abs(cohens_d) > 0.8 else ('Medium' if abs(cohens_d) > 0.5 else 'Small')
        })
    
    stats_df = pd.DataFrame(results)
    
    output_path = Path(output_dir) / 'statistical_tests.csv'
    stats_df.to_csv(output_path, index=False)
    print(f"Saved statistical test results to {output_path}")
    
    return stats_df

def generate_summary_report(comparison_df, stats_df, output_dir):
    """
    Generate a text summary report of the ablation study.
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ABLATION STUDY SUMMARY REPORT")
    report_lines.append("Quality-Based Tile Selection vs Random Tile Selection")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall findings
    report_lines.append("OVERALL FINDINGS:")
    report_lines.append("-" * 80)
    
    for method in comparison_df['Method'].unique():
        method_data = comparison_df[comparison_df['Method'] == method]
        avg_improvement = method_data['Improvement (%)'].mean()
        
        report_lines.append(f"\n{method}:")
        report_lines.append(f"  Average Improvement: {avg_improvement:.2f}%")
        
        # Best and worst metrics
        best_metric = method_data.loc[method_data['Improvement (%)'].idxmax()]
        worst_metric = method_data.loc[method_data['Improvement (%)'].idxmin()]
        
        report_lines.append(f"  Best Improvement: {best_metric['Metric']} (+{best_metric['Improvement (%)']:.2f}%)")
        report_lines.append(f"  Worst Improvement: {worst_metric['Metric']} ({worst_metric['Improvement (%)']:.2f}%)")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("STATISTICAL SIGNIFICANCE:")
    report_lines.append("-" * 80)
    
    for _, row in stats_df.iterrows():
        report_lines.append(f"\n{row['Method']}:")
        report_lines.append(f"  P-value: {row['P-value']:.4f}")
        report_lines.append(f"  Significant: {row['Significant (p<0.05)']}")
        report_lines.append(f"  Effect Size: {row['Effect_Size']} (Cohen's d = {row['Cohens_d']:.3f})")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("CONCLUSION:")
    report_lines.append("-" * 80)
    
    # Determine overall conclusion
    avg_overall_improvement = comparison_df['Improvement (%)'].mean()
    significant_count = (stats_df['Significant (p<0.05)'] == 'Yes').sum()
    
    if avg_overall_improvement > 0 and significant_count > 0:
        report_lines.append(f"\nQuality-based tile selection shows an average improvement of {avg_overall_improvement:.2f}%")
        report_lines.append(f"over random selection, with {significant_count}/{len(stats_df)} methods showing")
        report_lines.append("statistically significant differences (p < 0.05).")
        report_lines.append("\nThis demonstrates the VALUE of Stage 1 (quality scoring) in the pipeline.")
    else:
        report_lines.append("\nThe results do not show consistent significant improvement from quality-based")
        report_lines.append("selection over random selection.")
    
    report_lines.append("\n" + "=" * 80)
    
    # Write report
    report_text = "\n".join(report_lines)
    output_path = Path(output_dir) / 'ablation_summary_report.txt'
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"\nSaved summary report to {output_path}")
    print("\n" + report_text)

def main():
    parser = argparse.ArgumentParser(
        description="Compare ablation study results between quality-based and random tile selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--quality_dir", required=True,
                       help="Directory with quality-based selection test results")
    parser.add_argument("--random_dir", required=True,
                       help="Directory with random selection test results")
    parser.add_argument("--output_dir", default="./ablation_comparison",
                       help="Directory to save comparison results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Loading Results...")
    print(f"{'='*80}\n")
    
    # Load metrics
    quality_metrics = load_metrics(args.quality_dir)
    random_metrics = load_metrics(args.random_dir)
    
    print(f"Quality-based results loaded from: {args.quality_dir}")
    print(f"Random selection results loaded from: {args.random_dir}")
    
    print(f"\n{'='*80}")
    print("Generating Comparisons...")
    print(f"{'='*80}\n")
    
    # Create comparison table
    comparison_df = create_comparison_table(quality_metrics, random_metrics, output_dir)
    
    # Generate visualizations
    plot_metric_comparison(comparison_df, output_dir)
    plot_improvement_heatmap(comparison_df, output_dir)
    compare_roc_curves(quality_metrics, random_metrics, output_dir)
    
    # Statistical tests
    stats_df = statistical_significance_test(comparison_df, output_dir)
    
    # Generate summary report
    generate_summary_report(comparison_df, stats_df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"All comparison results saved to: {output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
