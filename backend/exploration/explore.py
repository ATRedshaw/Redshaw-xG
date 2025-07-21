from preprocessing.preprocessing import main as run_preprocessing_pipeline
from analysis.data_aggregation import create_player_match_stats, create_lagged_features
from analysis.regression_models import run_linear_regressions
import matplotlib.pyplot as plt
import os
import pandas as pd

def main():
    df_with_xg = run_preprocessing_pipeline()
    if df_with_xg is not None:
        print("\nSuccessfully loaded DataFrame with xG predictions in explore.py:")

        # Ensure date is ordered oldest to newest.
        df_with_xg.sort_values(by=['date', 'minute'], inplace=True)

        print(f'df_with_xg.columns: {df_with_xg.columns}')
        print("\nDataFrame sorted by date:")
        print(df_with_xg.head())

        # Define window sizes for lagged features
        windows = [1, 2, 4, 8, 16, 32, 64]
        
        # To store correlation results for the summary table
        correlation_summary = []

        # 1. Create player match statistics
        player_match_df = create_player_match_stats(df_with_xg)
        print("\nPlayer Match Statistics DataFrame created:")
        print(player_match_df.head())
        print(f"Shape: {player_match_df.shape}")

        for window in windows:
            print(f"\n--- Processing for window size: {window} ---")
            
            # 2. Create lagged features
            lagged_analysis_df = create_lagged_features(player_match_df, window, window)
            print("\nLagged Analysis DataFrame created:")
            print(lagged_analysis_df.head())
            print(f"Shape: {lagged_analysis_df.shape}")

            # 3. Run linear regressions
            print("\n--- Running Linear Regressions ---")
            linear_results = run_linear_regressions(lagged_analysis_df)

            # Filter out results with notes
            valid_results = [r for r in linear_results if 'note' not in r]
            
            # Store results for this window
            for r in valid_results:
                correlation_summary.append({
                    'window_size': window,
                    'predictor': r['predictor'],
                    'correlation': r['Pearson Correlation']
                })

            if not valid_results:
                print(f"No valid regression results to plot for window size {window}.")
                continue

            # Sort results for consistent plot ordering
            desired_order_map = {
                'past_goals': 0,
                'past_xg_basic': 1,
                'past_xg_shottype': 2,
                'past_xg_situation': 3,
                'past_xg_advanced': 4
            }
            valid_results.sort(key=lambda r: desired_order_map.get(r['predictor'], 99))

            # 4. Create visualisations
            print("\n--- Creating Visualisations ---")
            
            # Determine shared axis limits for comparability
            x_min, x_max = float('inf'), float('-inf')
            y_min, y_max = float('inf'), float('-inf')
            for r in valid_results:
                x_min = min(x_min, r['data'][r['predictor']].min())
                x_max = max(x_max, r['data'][r['predictor']].max())
                y_min = min(y_min, r['data']['future_goals'].min(), r['predictions'].min())
                y_max = max(y_max, r['data']['future_goals'].max(), r['predictions'].max())
            
            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05
            x_lim = (x_min - x_pad, x_max + x_pad)
            y_lim = (y_min - y_pad, y_max + y_pad)

            # Use a professional style
            plt.style.use('seaborn-v0_8-whitegrid')
            num_plots = len(valid_results)
            n_cols = min(num_plots, 3)
            n_rows = (num_plots + n_cols - 1) // n_cols
                
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False)
            axes = axes.flatten()

            fig.suptitle(f'Linear Regression: Predicting Future Goals (Window: {window})', fontsize=18, fontweight='bold')

            for i, result in enumerate(valid_results):
                ax = axes[i]
                predictor = result['predictor']
                data = result['data']
                predictions = result['predictions']
                
                # Scatter plot
                ax.scatter(data[predictor], data['future_goals'], alpha=0.6, color='dodgerblue', edgecolors='w', s=60, label='Actual Data')
                
                # Regression line
                ax.plot(data[predictor], predictions, color='crimson', linewidth=2.5, label='Regression Line')
                
                # Titles and labels
                ax.set_title(f'{predictor} vs. Future Goals', fontsize=12, fontweight='medium')
                ax.set_xlabel(predictor, fontsize=10)
                ax.set_ylabel('Future Goals', fontsize=10)
                ax.legend(fontsize=9)
                
                # Set shared axes
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                
                # Add text for correlation and p-value
                stats_text = (
                    f"Corr: {result['Pearson Correlation']:.3f}\n"
                    f"P-val: {result['P-value (Coefficient)']:.3f}"
                )
                ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.4', fc='lightgrey', alpha=0.8, ec='black'))

            # Hide any unused subplots
            for j in range(num_plots, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the figure
            output_dir = 'exploration/figures'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'linear_regression_window_{window}.png')
            plt.savefig(output_path, dpi=300)
            print(f"\nFigure saved to {output_path}")

        # After the loop, create and save the correlation summary
        if correlation_summary:
            summary_df = pd.DataFrame(correlation_summary)
            pivot_df = summary_df.pivot(index='predictor', columns='window_size', values='correlation')
            
            # Reorder the predictors for better comparison
            desired_order = [
                'past_goals',
                'past_xg_basic',
                'past_xg_shottype',
                'past_xg_situation',
                'past_xg_advanced'
            ]
            pivot_df = pivot_df.reindex(desired_order)

            summary_output_path = os.path.join('figures', 'correlation_summary.csv')
            pivot_df.to_csv(summary_output_path, float_format='%.3f')
            print(f"\nCorrelation summary saved to {summary_output_path}")


if __name__ == '__main__':
    main()