"""
Advanced Solar Flare Analysis
- Multi-flare detection and segmentation
- Real-time anomaly detection
- Predictive modeling with uncertainty
- Comparative algorithm analysis (HMC vs Adaptive MH)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import norm, chi2
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


class AdvancedSolarFlareAnalyzer:
    """
    Next-level analysis tools for solar flare detection and prediction
    """
    
    def __init__(self, analyzer):
        """
        Initialize with base analyzer
        
        Parameters:
        -----------
        analyzer : SolarFlareAnalyzer
            Base analyzer with MCMC results
        """
        self.analyzer = analyzer
        self.changepoints = []
        self.anomaly_scores = []
        
    def detect_multiple_flares(self, prominence=0.5, distance=50):
        """
        Detect multiple flare events in the time series
        Uses peak detection on smoothed signal
        """
        # Smooth the signal
        smoothed = savgol_filter(self.analyzer.intensity, 
                                window_length=min(51, len(self.analyzer.intensity)//2*2-1), 
                                polyorder=3)
        
        # Find peaks
        peaks, properties = find_peaks(np.abs(smoothed), 
                                      prominence=prominence * np.max(np.abs(smoothed)),
                                      distance=distance)
        
        self.changepoints = peaks
        
        print(f"\nDetected {len(peaks)} potential flare events at indices:")
        for i, peak in enumerate(peaks):
            print(f"  Flare {i+1}: t = {self.analyzer.time[peak]:.3f}, "
                  f"intensity = {self.analyzer.intensity[peak]:.3f}")
        
        return peaks, properties
    
    def anomaly_detection(self, contamination=0.1):
        """
        Identify anomalous points using Isolation Forest
        Useful for detecting unusual flare behavior
        """
        # Prepare features
        X = np.column_stack([
            self.analyzer.time,
            self.analyzer.intensity,
            np.gradient(self.analyzer.intensity),  # First derivative
            np.gradient(np.gradient(self.analyzer.intensity))  # Second derivative
        ])
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        self.anomaly_scores = iso_forest.score_samples(X)
        
        anomalies = np.where(anomaly_labels == -1)[0]
        
        print(f"\nAnomaly Detection Results:")
        print(f"  Detected {len(anomalies)} anomalous points ({100*len(anomalies)/len(X):.2f}%)")
        
        return anomalies, self.anomaly_scores
    
    def bayesian_changepoint_detection(self, prior_scale=1.0):
        """
        Bayesian changepoint detection using cumulative sum methods
        Identifies regime changes in the time series
        """
        # Compute residuals from best fit
        summary = self.analyzer.get_posterior_summary()
        map_params = [summary['A']['MAP'], summary['tau']['MAP'], summary['omega']['MAP']]
        predicted = self.analyzer.flare_model(self.analyzer.time_norm, *map_params)
        
        residuals = self.analyzer.intensity - predicted
        
        # Compute cumulative sum of standardized residuals
        std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
        cusum = np.cumsum(std_residuals)
        
        # Detect significant deviations
        threshold = prior_scale * np.sqrt(len(residuals))
        changepoints = np.where(np.abs(cusum) > threshold)[0]
        
        return changepoints, cusum
    
    def hamiltonian_monte_carlo(self, n_iterations=10000, n_burn=2000, 
                               epsilon=0.01, L=20):
        """
        Hamiltonian Monte Carlo (HMC) for comparison with Adaptive MH
        Generally more efficient for complex posteriors
        
        Parameters:
        -----------
        n_iterations : int
            Number of MCMC iterations
        n_burn : int
            Burn-in period
        epsilon : float
            Step size for leapfrog integration
        L : int
            Number of leapfrog steps
        """
        print("\nRunning Hamiltonian Monte Carlo...")
        
        # Start from MAP estimate
        current_q = self.analyzer.find_map_estimate()
        
        # Storage
        n_samples = n_iterations - n_burn
        samples = np.zeros((n_samples, 3))
        acceptance_count = 0
        
        def grad_log_posterior(q):
            """Compute gradient using finite differences"""
            h = 1e-6
            grad = np.zeros_like(q)
            
            for i in range(len(q)):
                q_plus = q.copy()
                q_minus = q.copy()
                q_plus[i] += h
                q_minus[i] -= h
                
                grad[i] = (self.analyzer.log_posterior(q_plus) - 
                          self.analyzer.log_posterior(q_minus)) / (2 * h)
            
            return grad
        
        def leapfrog(q, p, epsilon, L):
            """Leapfrog integrator"""
            q_new = q.copy()
            p_new = p.copy()
            
            # Half step for momentum
            p_new = p_new + 0.5 * epsilon * grad_log_posterior(q_new)
            
            # Full steps
            for _ in range(L - 1):
                q_new = q_new + epsilon * p_new
                p_new = p_new + epsilon * grad_log_posterior(q_new)
            
            # Final position and half step for momentum
            q_new = q_new + epsilon * p_new
            p_new = p_new + 0.5 * epsilon * grad_log_posterior(q_new)
            
            return q_new, p_new
        
        sample_idx = 0
        
        for i in range(n_iterations):
            # Sample momentum
            current_p = np.random.randn(3)
            
            # Compute current Hamiltonian
            current_H = -self.analyzer.log_posterior(current_q) + 0.5 * np.sum(current_p**2)
            
            # Leapfrog integration
            proposed_q, proposed_p = leapfrog(current_q, current_p, epsilon, L)
            
            # Compute proposed Hamiltonian
            proposed_H = -self.analyzer.log_posterior(proposed_q) + 0.5 * np.sum(proposed_p**2)
            
            # Metropolis acceptance
            if np.log(np.random.uniform()) < (current_H - proposed_H):
                current_q = proposed_q
                if i >= n_burn:
                    acceptance_count += 1
            
            # Store sample
            if i >= n_burn:
                samples[sample_idx] = current_q
                sample_idx += 1
            
            if (i + 1) % 2000 == 0:
                print(f"  Iteration {i+1}/{n_iterations}")
        
        acceptance_rate = acceptance_count / n_samples
        print(f"HMC Acceptance rate: {acceptance_rate:.3f}")
        
        return samples, acceptance_rate
    
    def predictive_posterior(self, future_time, n_samples=5000):
        """
        Generate predictive posterior for future time points
        Includes full uncertainty propagation
        
        Parameters:
        -----------
        future_time : array
            Future time points to predict
        n_samples : int
            Number of posterior samples to use
        
        Returns:
        --------
        predictions : dict
            Mean, median, and credible intervals
        """
        if self.analyzer.chain is None:
            raise ValueError("Run MCMC first!")
        
        # Sample from posterior
        indices = np.random.choice(len(self.analyzer.chain), n_samples, replace=True)
        posterior_samples = self.analyzer.chain[indices]
        
        # Normalize future time
        future_time_norm = (future_time - self.analyzer.time_mean) / self.analyzer.time_std
        
        # Generate predictions
        predictions = np.zeros((n_samples, len(future_time)))
        
        for i, params in enumerate(posterior_samples):
            predictions[i] = self.analyzer.flare_model(future_time_norm, *params)
            
            # Add observational noise
            sigma = self.analyzer.compute_sigma(predictions[i])
            predictions[i] += np.random.normal(0, sigma)
        
        # Compute statistics
        result = {
            'mean': np.mean(predictions, axis=0),
            'median': np.median(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'CI_95_lower': np.percentile(predictions, 2.5, axis=0),
            'CI_95_upper': np.percentile(predictions, 97.5, axis=0),
            'CI_68_lower': np.percentile(predictions, 16, axis=0),
            'CI_68_upper': np.percentile(predictions, 84, axis=0),
        }
        
        return result
    
    def plot_multi_flare_detection(self, save_path='multi_flare_detection.png'):
        """Visualize multiple flare events"""
        peaks, properties = self.detect_multiple_flares()
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Original signal with detected peaks
        axes[0].plot(self.analyzer.time, self.analyzer.intensity, 
                    'b-', alpha=0.6, label='Observed Data')
        axes[0].plot(self.analyzer.time[peaks], self.analyzer.intensity[peaks], 
                    'r*', markersize=15, label='Detected Flares')
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Intensity', fontsize=12)
        axes[0].set_title('Multi-Flare Detection', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Smoothed signal
        smoothed = savgol_filter(self.analyzer.intensity, 
                                window_length=min(51, len(self.analyzer.intensity)//2*2-1),
                                polyorder=3)
        axes[1].plot(self.analyzer.time, smoothed, 'g-', linewidth=2, label='Smoothed Signal')
        axes[1].plot(self.analyzer.time[peaks], smoothed[peaks], 
                    'r*', markersize=15, label='Detected Peaks')
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('Smoothed Intensity', fontsize=12)
        axes[1].set_title('Smoothed Signal Analysis', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-flare detection plot saved to {save_path}")
        return fig
    
    def plot_anomaly_detection(self, save_path='anomaly_detection.png'):
        """Visualize anomaly detection results"""
        anomalies, scores = self.anomaly_detection()
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series with anomalies highlighted
        axes[0].plot(self.analyzer.time, self.analyzer.intensity, 
                    'b-', alpha=0.6, label='Normal Data')
        axes[0].scatter(self.analyzer.time[anomalies], 
                       self.analyzer.intensity[anomalies],
                       c='red', s=50, marker='x', linewidths=2,
                       label=f'Anomalies ({len(anomalies)})', zorder=5)
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Intensity', fontsize=12)
        axes[0].set_title('Anomaly Detection in Solar Flare Data', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Anomaly scores over time
        axes[1].plot(self.analyzer.time, scores, 'g-', linewidth=1.5)
        axes[1].axhline(np.percentile(scores, 10), color='red', 
                       linestyle='--', label='Anomaly Threshold')
        axes[1].fill_between(self.analyzer.time, 
                            np.min(scores), np.percentile(scores, 10),
                            alpha=0.3, color='red', label='Anomaly Region')
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('Anomaly Score', fontsize=12)
        axes[1].set_title('Anomaly Scores Over Time', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Anomaly detection plot saved to {save_path}")
        return fig
    
    def plot_predictive_forecast(self, forecast_horizon=2.0, 
                                save_path='predictive_forecast.png'):
        """Generate and plot predictive forecast"""
        # Create future time points
        t_max = np.max(self.analyzer.time)
        future_time = np.linspace(t_max, t_max + forecast_horizon, 500)
        
        # Generate predictions
        predictions = self.predictive_posterior(future_time)
        
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Historical data
        ax.plot(self.analyzer.time, self.analyzer.intensity, 
               'o', markersize=3, alpha=0.5, color='black', label='Historical Data')
        
        # Best fit on historical data
        summary = self.analyzer.get_posterior_summary()
        map_params = [summary['A']['MAP'], summary['tau']['MAP'], summary['omega']['MAP']]
        historical_fit = self.analyzer.flare_model(self.analyzer.time_norm, *map_params)
        ax.plot(self.analyzer.time, historical_fit, 'b-', 
               linewidth=2, label='Historical Fit')
        
        # Forecast
        ax.plot(future_time, predictions['mean'], 'r-', 
               linewidth=2.5, label='Forecast (Mean)', zorder=5)
        
        # Uncertainty bands
        ax.fill_between(future_time, 
                       predictions['CI_95_lower'], 
                       predictions['CI_95_upper'],
                       alpha=0.2, color='red', label='95% Prediction Interval')
        ax.fill_between(future_time,
                       predictions['CI_68_lower'],
                       predictions['CI_68_upper'],
                       alpha=0.3, color='red', label='68% Prediction Interval')
        
        # Vertical line separating historical and forecast
        ax.axvline(t_max, color='gray', linestyle='--', 
                  linewidth=2, label='Forecast Boundary')
        
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Intensity', fontsize=14)
        ax.set_title('Bayesian Predictive Forecast with Uncertainty', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictive forecast plot saved to {save_path}")
        return fig
    
    def compare_mcmc_methods(self, save_path='mcmc_comparison.png'):
        """Compare Adaptive MH and HMC performance"""
        print("\nComparing MCMC methods...")
        
        # Run HMC
        hmc_chain, hmc_acceptance = self.hamiltonian_monte_carlo(
            n_iterations=15000, n_burn=3000)
        
        # Compare posteriors
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        param_names = ['Amplitude (A)', 'Decay Time (τ)', 'Frequency (ω)']
        
        for i, param_name in enumerate(param_names):
            # Adaptive MH
            axes[i, 0].hist(self.analyzer.chain[:, i], bins=50, 
                           density=True, alpha=0.6, color='blue',
                           label='Adaptive MH', edgecolor='black')
            axes[i, 0].set_xlabel(param_name, fontsize=12)
            axes[i, 0].set_ylabel('Density', fontsize=11)
            axes[i, 0].set_title(f'{param_name} - Adaptive MH', fontsize=13)
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # HMC
            axes[i, 1].hist(hmc_chain[:, i], bins=50, 
                           density=True, alpha=0.6, color='red',
                           label='HMC', edgecolor='black')
            axes[i, 1].set_xlabel(param_name, fontsize=12)
            axes[i, 1].set_ylabel('Density', fontsize=11)
            axes[i, 1].set_title(f'{param_name} - HMC', fontsize=13)
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.suptitle('MCMC Method Comparison: Adaptive MH vs HMC', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MCMC comparison plot saved to {save_path}")
        
        # Print comparison statistics
        print("\nMCMC Method Comparison:")
        print(f"  Adaptive MH Acceptance Rate: {self.analyzer.acceptance_rate:.3f}")
        print(f"  HMC Acceptance Rate:         {hmc_acceptance:.3f}")
        
        return fig, hmc_chain


class RealTimeFlarePrediction:
    """
    Real-time flare prediction system
    Uses streaming data and online Bayesian updating
    """
    
    def __init__(self, baseline_analyzer):
        self.baseline = baseline_analyzer
        self.alert_threshold = 3.0  # Standard deviations
        
    def compute_prediction_score(self, new_data_point):
        """
        Compute anomaly score for new data point
        Returns probability of being a flare event
        """
        # Get posterior predictive distribution
        summary = self.baseline.analyzer.get_posterior_summary()
        
        # Compute expected value
        map_params = [summary['A']['MAP'], summary['tau']['MAP'], 
                     summary['omega']['MAP']]
        
        # This would use the actual time point in practice
        # For now, compute based on deviation from baseline
        expected = np.mean(self.baseline.analyzer.intensity)
        std = np.std(self.baseline.analyzer.intensity)
        
        # Z-score
        z_score = (new_data_point - expected) / std
        
        # Probability of being anomalous
        p_anomaly = 1 - norm.cdf(abs(z_score))
        
        return z_score, p_anomaly
    
    def trigger_alert(self, z_score):
        """Check if alert should be triggered"""
        return abs(z_score) > self.alert_threshold


# Example usage
if __name__ == "__main__":
    print("Advanced Features Module")
    print("Import this module and use with base SolarFlareAnalyzer")
    print("\nExample:")
    print("  advanced = AdvancedSolarFlareAnalyzer(analyzer)")
    print("  advanced.detect_multiple_flares()")
    print("  advanced.anomaly_detection()")
    print("  advanced.plot_predictive_forecast()")
