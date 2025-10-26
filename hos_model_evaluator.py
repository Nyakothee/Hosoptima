"""
Advanced Model Evaluator for HOS Violation Prediction System
Comprehensive evaluation metrics, visualization, and performance analysis
Production-ready with detailed model assessment
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, matthews_corrcoef,
    cohen_kappa_score, log_loss, average_precision_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EvaluationConfig:
    """Configuration for model evaluation"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        config = config_dict or {}
        
        # Evaluation parameters
        self.classification_threshold = config.get('classification_threshold', 0.5)
        self.n_bootstrap_samples = config.get('n_bootstrap_samples', 1000)
        self.confidence_level = config.get('confidence_level', 0.95)
        
        # Visualization
        self.plot_size = config.get('plot_size', (12, 8))
        self.dpi = config.get('dpi', 300)
        self.save_plots = config.get('save_plots', True)
        self.plot_dir = config.get('plot_dir', './evaluation_plots')
        
        # Reporting
        self.generate_detailed_report = config.get('generate_detailed_report', True)
        self.report_dir = config.get('report_dir', './evaluation_reports')


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_pred_proba=None,
                             average='weighted') -> Dict:
        """Calculate all classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Advanced metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 2:
                    # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', average=average)
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                else:
                    # Binary
                    if len(y_pred_proba.shape) > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                    metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Error calculating probability-based metrics: {str(e)}")
        
        logger.info("Calculated all metrics")
        return metrics
    
    @staticmethod
    def calculate_class_wise_metrics(y_true, y_pred, class_names=None) -> pd.DataFrame:
        """Calculate per-class metrics"""
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Support (number of samples per class)
        unique, counts = np.unique(y_true, return_counts=True)
        support = dict(zip(unique, counts))
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(len(precision))]
        
        metrics_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': [support.get(i, 0) for i in range(len(precision))]
        })
        
        return metrics_df
    
    @staticmethod
    def calculate_confidence_intervals(y_true, y_pred, n_bootstraps=1000, 
                                      confidence_level=0.95) -> Dict:
        """Calculate confidence intervals using bootstrap"""
        np.random.seed(42)
        
        bootstrap_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        n_samples = len(y_true)
        
        for _ in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.randint(0, n_samples, n_samples)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics
            bootstrap_scores['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            bootstrap_scores['precision'].append(
                precision_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0)
            )
            bootstrap_scores['recall'].append(
                recall_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0)
            )
            bootstrap_scores['f1'].append(
                f1_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0)
            )
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        confidence_intervals = {}
        
        for metric, scores in bootstrap_scores.items():
            lower = np.percentile(scores, alpha/2 * 100)
            upper = np.percentile(scores, (1 - alpha/2) * 100)
            mean = np.mean(scores)
            
            confidence_intervals[metric] = {
                'mean': mean,
                'lower': lower,
                'upper': upper,
                'std': np.std(scores)
            }
        
        logger.info(f"Calculated {confidence_level*100}% confidence intervals")
        return confidence_intervals


class ConfusionMatrixAnalyzer:
    """Analyze and visualize confusion matrices"""
    
    @staticmethod
    def create_confusion_matrix(y_true, y_pred, normalize=None) -> np.ndarray:
        """Create confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        return cm
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix',
                            figsize=(10, 8), save_path=None):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=figsize)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='.2f' if cm.dtype == float else 'd',
                   cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def analyze_confusion_matrix(cm, class_names=None) -> Dict:
        """Extract insights from confusion matrix"""
        analysis = {}
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Most confused pairs
        np.fill_diagonal(cm, 0)
        confused_pairs = []
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': i,
                        'pred_class': j,
                        'count': int(cm[i, j])
                    })
        
        confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)
        
        analysis['per_class_accuracy'] = per_class_accuracy.tolist()
        analysis['top_confused_pairs'] = confused_pairs[:5]
        
        return analysis


class ROCAnalyzer:
    """Analyze and plot ROC curves"""
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, n_classes=None,
                      class_names=None, figsize=(10, 8), save_path=None):
        """Plot ROC curve(s)"""
        plt.figure(figsize=figsize)
        
        if len(y_pred_proba.shape) == 1 or (len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 1):
            # Binary classification
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            auc = roc_auc_score(y_true, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
        else:
            # Multi-class
            from sklearn.preprocessing import label_binarize
            
            if n_classes is None:
                n_classes = y_pred_proba.shape[1]
            
            if class_names is None:
                class_names = [f'Class {i}' for i in range(n_classes)]
            
            # Binarize labels
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            # Plot ROC for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})', linewidth=2)
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred_proba, 
                                    figsize=(10, 8), save_path=None):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=figsize)
        
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.plot(recall, precision, linewidth=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.close()


class ModelComparator:
    """Compare multiple models"""
    
    @staticmethod
    def compare_models(results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Compare metrics across multiple models"""
        comparison_data = []
        
        for model_name, metrics in results_dict.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by a primary metric (e.g., F1 score)
        if 'f1_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        logger.info("Model comparison completed")
        return comparison_df
    
    @staticmethod
    def plot_model_comparison(comparison_df: pd.DataFrame,
                             metrics_to_plot=None,
                             figsize=(14, 8), save_path=None):
        """Plot model comparison"""
        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Filter available metrics
        metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize)
        
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            models = comparison_df['Model']
            values = comparison_df[metric]
            
            bars = ax.bar(range(len(models)), values, alpha=0.7)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def statistical_significance_test(model1_scores, model2_scores) -> Dict:
        """Perform statistical significance test between two models"""
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Wilcoxon signed-rank test (non-parametric)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(model1_scores, model2_scores)
        
        result = {
            't_statistic': float(t_stat),
            't_test_p_value': float(p_value),
            'wilcoxon_statistic': float(wilcoxon_stat),
            'wilcoxon_p_value': float(wilcoxon_p),
            'significantly_different': p_value < 0.05
        }
        
        return result


class AdvancedModelEvaluator:
    """
    Comprehensive model evaluation with metrics, visualizations, and analysis
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.metrics_calculator = MetricsCalculator()
        self.cm_analyzer = ConfusionMatrixAnalyzer()
        self.roc_analyzer = ROCAnalyzer()
        self.comparator = ModelComparator()
        
        self.evaluation_results = {}
        
        # Create directories
        Path(self.config.plot_dir).mkdir(exist_ok=True, parents=True)
        Path(self.config.report_dir).mkdir(exist_ok=True, parents=True)
    
    def evaluate_model(self, model_name: str, y_true, y_pred, 
                      y_pred_proba=None, class_names=None) -> Dict:
        """Comprehensive evaluation of a single model"""
        logger.info(f"Evaluating model: {model_name}")
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true, y_pred, y_pred_proba
        )
        results['metrics'] = metrics
        
        # Per-class metrics
        if class_names:
            class_metrics_df = self.metrics_calculator.calculate_class_wise_metrics(
                y_true, y_pred, class_names
            )
            results['class_metrics'] = class_metrics_df.to_dict('records')
        
        # Confidence intervals
        confidence_intervals = self.metrics_calculator.calculate_confidence_intervals(
            y_true, y_pred,
            n_bootstraps=self.config.n_bootstrap_samples,
            confidence_level=self.config.confidence_level
        )
        results['confidence_intervals'] = confidence_intervals
        
        # Confusion matrix
        cm = self.cm_analyzer.create_confusion_matrix(y_true, y_pred)
        cm_normalized = self.cm_analyzer.create_confusion_matrix(y_true, y_pred, normalize='true')
        
        results['confusion_matrix'] = cm.tolist()
        results['confusion_matrix_normalized'] = cm_normalized.tolist()
        
        # Confusion matrix analysis
        cm_analysis = self.cm_analyzer.analyze_confusion_matrix(cm, class_names)
        results['confusion_matrix_analysis'] = cm_analysis
        
        # Generate visualizations
        if self.config.save_plots:
            # Confusion matrix plot
            cm_path = f"{self.config.plot_dir}/{model_name}_confusion_matrix.png"
            self.cm_analyzer.plot_confusion_matrix(
                cm_normalized, class_names, 
                title=f'Confusion Matrix - {model_name}',
                save_path=cm_path
            )
            
            # ROC curve
            if y_pred_proba is not None:
                roc_path = f"{self.config.plot_dir}/{model_name}_roc_curve.png"
                self.roc_analyzer.plot_roc_curve(
                    y_true, y_pred_proba, 
                    n_classes=len(np.unique(y_true)),
                    class_names=class_names,
                    save_path=roc_path
                )
                
                # Precision-Recall curve (for binary)
                if len(np.unique(y_true)) == 2:
                    pr_path = f"{self.config.plot_dir}/{model_name}_pr_curve.png"
                    self.roc_analyzer.plot_precision_recall_curve(
                        y_true, y_pred_proba,
                        save_path=pr_path
                    )
        
        # Store results
        self.evaluation_results[model_name] = results
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return results
    
    def evaluate_multiple_models(self, models_dict: Dict, 
                                 y_true, class_names=None) -> pd.DataFrame:
        """Evaluate multiple models and compare"""
        logger.info(f"Evaluating {len(models_dict)} models...")
        
        all_metrics = {}
        
        for model_name, predictions in models_dict.items():
            y_pred = predictions.get('y_pred')
            y_pred_proba = predictions.get('y_pred_proba')
            
            results = self.evaluate_model(
                model_name, y_true, y_pred, y_pred_proba, class_names
            )
            
            all_metrics[model_name] = results['metrics']
        
        # Compare models
        comparison_df = self.comparator.compare_models(all_metrics)
        
        # Plot comparison
        if self.config.save_plots:
            comparison_path = f"{self.config.plot_dir}/model_comparison.png"
            self.comparator.plot_model_comparison(comparison_df, save_path=comparison_path)
        
        logger.info("Multi-model evaluation completed")
        
        return comparison_df
    
    def generate_evaluation_report(self, model_name: str = None):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        if model_name:
            results_to_report = {model_name: self.evaluation_results[model_name]}
        else:
            results_to_report = self.evaluation_results
        
        # Create text report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for name, results in results_to_report.items():
            report_lines.append(f"\nModel: {name}")
            report_lines.append("-" * 80)
            
            # Metrics
            report_lines.append("\nOverall Metrics:")
            for metric, value in results['metrics'].items():
                report_lines.append(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            
            # Confidence intervals
            report_lines.append("\nConfidence Intervals (95%):")
            for metric, ci in results['confidence_intervals'].items():
                report_lines.append(
                    f"  {metric.title()}: {ci['mean']:.4f} "
                    f"[{ci['lower']:.4f}, {ci['upper']:.4f}]"
                )
            
            # Class-wise metrics
            if 'class_metrics' in results:
                report_lines.append("\nPer-Class Metrics:")
                for cm in results['class_metrics']:
                    report_lines.append(
                        f"  {cm['Class']}: "
                        f"Precision={cm['Precision']:.3f}, "
                        f"Recall={cm['Recall']:.3f}, "
                        f"F1={cm['F1-Score']:.3f}"
                    )
            
            # Confusion matrix insights
            if 'confusion_matrix_analysis' in results:
                report_lines.append("\nMost Confused Classes:")
                for pair in results['confusion_matrix_analysis']['top_confused_pairs'][:3]:
                    report_lines.append(
                        f"  True: {pair['true_class']} -> "
                        f"Predicted: {pair['pred_class']} "
                        f"(Count: {pair['count']})"
                    )
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = f"{self.config.report_dir}/evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        # Save JSON report
        json_path = f"{self.config.report_dir}/evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        logger.info(f"JSON results saved to {json_path}")
        
        return report_text
    
    def plot_metrics_distribution(self, save_path=None):
        """Plot distribution of metrics across models"""
        if len(self.evaluation_results) < 2:
            logger.warning("Need at least 2 models for distribution plot")
            return
        
        metrics_data = []
        for model_name, results in self.evaluation_results.items():
            for metric, value in results['metrics'].items():
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': value
                })
        
        df = pd.DataFrame(metrics_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        metrics = df['Metric'].unique()
        x = np.arange(len(metrics))
        width = 0.8 / len(df['Model'].unique())
        
        for idx, model in enumerate(df['Model'].unique()):
            model_data = df[df['Model'] == model]
            values = [model_data[model_data['Metric'] == m]['Value'].values[0] 
                     if m in model_data['Metric'].values else 0 
                     for m in metrics]
            
            ax.bar(x + idx * width, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Metrics Distribution Across Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(df['Model'].unique()) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.config.plot_dir}/metrics_distribution.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics distribution plot saved to {save_path}")
        plt.close()


# Main execution
if __name__ == "__main__":
    # Example usage
    config = EvaluationConfig({
        'save_plots': True,
        'generate_detailed_report': True
    })
    
    evaluator = AdvancedModelEvaluator(config)
    
    # Example: Evaluate single model
    # y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    # y_pred = np.array([0, 1, 2, 1, 0, 1, 1, 0])
    # y_pred_proba = np.random.rand(8, 3)
    # class_names = ['Class A', 'Class B', 'Class C']
    
    # results = evaluator.evaluate_model('LSTM', y_true, y_pred, y_pred_proba, class_names)
    
    # Generate report
    # report = evaluator.generate_evaluation_report()