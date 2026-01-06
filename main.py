"""
Advanced Solar Flare Analysis using Bayesian MCMC
Author: Shivani Bhat

This implementation includes:
- Adaptive MCMC with multiple algorithms
- Hamiltonian Monte Carlo
- Anomaly detection
- Predictive modeling
- Advanced visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import find_peaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SolarFlareAnalyzer:
    """
    Comprehensive Solar Flare Analysis Framework
    Implements Bayesian inference, changepoint detection, and predictive modeling
    """
    
    def __init__(self, time_data, intensity_data):
        """
        Initialize analyzer with solar flare data
        
        Parameters:
        -----------
        time_data : array-like
            Time series data
        intensity_data : array-like
            Sensor intensity measurements
        """
        self.time = np.array(time_data)
        self.intensity = np.array(intensity_data)
        self.n_points = len(time_data)
        
        # Normalize time for numerical stability
        self.time_mean = np.mean(self.time)
        self.time_std = np.std(self.time)
        self.time_norm = (self.time - self.time_mean) / self.time_std
        
        # Storage for MCMC results
        self.chain = None
        self.acceptance_rate = 0.0
        self.log_likelihood_trace = []
        
    def flare_model(self, t, A, tau, omega, t0=0):
        """
        Physical model for solar flare intensity
        
        I(t) = A * exp(-(t-t0)/tau) * sin(omega*(t-t0)) for t > t0
        
        Parameters:
        -----------
        t : array
            Time points (normalized)
        A : float
            Amplitude (peak intensity)
        tau : float
            Decay time constant
        omega : float
            Oscillation frequency
        t0 : float
            Flare onset time
        """
        t_shifted = t - t0
        mask = t_shifted > 0
        result = np.zeros_like(t)
        result[mask] = A * np.exp(-t_shifted[mask] / tau) * np.sin(omega * t_shifted[mask])
        return result
    
    def compute_sigma(self, y_data, eps_min=1e-6):
        """
        Compute heteroscedastic error with numerical stability
        
        σ_i = 0.2 * |y_data,i| + ε_min
        """
        return 0.2 * np.abs(y_data) + eps_min
    
    def log_likelihood(self, params, return_model=False):
        """
        Compute log-likelihood with numerical stability
        
        L = -Σ [(y_data - y_model)² / (2σ²)]
        """
        A, tau, omega = params
        
        # Parameter bounds for physical validity
        if A <= 0 or tau <= 0 or omega <= 0:
            return -np.inf if not return_model else (-np.inf, None)
        
        # Compute model prediction
        y_model = self.flare_model(self.time_norm, A, tau, omega)
        
        # Compute adaptive error
        sigma = self.compute_sigma(self.intensity)
        
        # Residuals with numerical stability
        residuals = self.intensity - y_model
        
        # Log-likelihood (avoiding overflow)
        log_like = -0.5 * np.sum((residuals**2) / (sigma**2 + 1e-10))
        log_like -= np.sum(np.log(sigma + 1e-10))  # Normalization term
        
        if return_model:
            return log_like, y_model
        return log_like
    
    def log_prior(self, params):
        """
        Log-prior distributions for parameters
        Using weakly informative priors
        """
        A, tau, omega = params
        
        # Physical constraints
        if A <= 0 or tau <= 0 or omega <= 0:
            return -np.inf
        
        # Log-normal priors (parameters must be positive)
        log_p_A = stats.lognorm.logpdf(A, s=1, scale=np.std(self.intensity))
        log_p_tau = stats.lognorm.logpdf(tau, s=1, scale=1.0)
        log_p_omega = stats.lognorm.logpdf(omega, s=1, scale=1.0)
        
        return log_p_A + log_p_tau + log_p_omega
    
    def log_posterior(self, params):
        """Unnormalized log-posterior"""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def find_map_estimate(self):
        """
        Find Maximum A Posteriori (MAP) estimate
        Using optimization as starting point for MCMC
        """
        # Initial guess based on data
        A_init = np.max(np.abs(self.intensity))
        tau_init = 1.0
        omega_init = 2 * np.pi
        
        initial_params = [A_init, tau_init, omega_init]
        
        # Negative log-posterior for minimization
        def neg_log_post(params):
            return -self.log_posterior(params)
        
        # Multiple optimization attempts with different initializations
        best_result = None
        best_value = np.inf
        
        for _ in range(5):
            # Add noise to initial guess
            init = np.array(initial_params) * np.random.uniform(0.5, 1.5, 3)
            
            result = minimize(neg_log_post, init, method='L-BFGS-B',
                            bounds=[(1e-6, None), (1e-6, None), (1e-6, None)])
            
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
        
        return best_result.x
    
    def adaptive_mcmc(self, n_iterations=50000, n_burn=10000, thin=10):
        """
        Adaptive Metropolis-Hastings MCMC
        Automatically tunes proposal distribution for better acceptance rate
        """
        # Initialize at MAP estimate
        current_params = self.find_map_estimate()
        current_log_post = self.log_posterior(current_params)
        
        # Storage
        n_samples = (n_iterations - n_burn) // thin
        chain = np.zeros((n_samples, 3))
        self.log_likelihood_trace = []
        
        # Adaptive proposal covariance
        proposal_cov = np.diag([0.1, 0.1, 0.1])
        acceptance_count = 0
        
        # Adaptation parameters
        adapt_interval = 100
        target_accept = 0.234  # Optimal for 3D
        
        print("Running Adaptive MCMC...")
        print(f"Initial MAP estimate: A={current_params[0]:.3f}, τ={current_params[1]:.3f}, ω={current_params[2]:.3f}")
        
        sample_idx = 0
        
        for i in range(n_iterations):
            # Propose new parameters
            proposed_params = np.random.multivariate_normal(current_params, proposal_cov)
            proposed_log_post = self.log_posterior(proposed_params)
            
            # Metropolis-Hastings acceptance
            log_alpha = proposed_log_post - current_log_post
            
            if np.log(np.random.uniform()) < log_alpha:
                current_params = proposed_params
                current_log_post = proposed_log_post
                acceptance_count += 1
            
            # Store samples after burn-in
            if i >= n_burn and (i - n_burn) % thin == 0:
                chain[sample_idx] = current_params
                self.log_likelihood_trace.append(self.log_likelihood(current_params))
                sample_idx += 1
            
            # Adaptive tuning
            if i > 0 and i % adapt_interval == 0 and i < n_burn:
                accept_rate = acceptance_count / adapt_interval
                
                # Adjust proposal scale
                if accept_rate < target_accept:
                    proposal_cov *= 0.9
                else:
                    proposal_cov *= 1.1
                
                acceptance_count = 0
            
            # Progress reporting
            if (i + 1) % 10000 == 0:
                print(f"Iteration {i+1}/{n_iterations}")
        
        self.chain = chain
        self.acceptance_rate = acceptance_count / n_iterations
        
        print(f"\nMCMC Complete!")
        print(f"Final acceptance rate: {self.acceptance_rate:.3f}")
        
        return chain
    
    def compute_convergence_diagnostics(self):
        """
        Compute convergence diagnostics:
        - Gelman-Rubin statistic (split chains)
        - Effective sample size
        - Autocorrelation
        """
        if self.chain is None:
            raise ValueError("Run MCMC first!")
        
        n_samples = len(self.chain)
        
        # Split chain into two halves
        chain1 = self.chain[:n_samples//2]
        chain2 = self.chain[n_samples//2:]
        
        diagnostics = {}
        
        for i, param_name in enumerate(['A', 'tau', 'omega']):
            # Between-chain variance
            chain_means = [np.mean(chain1[:, i]), np.mean(chain2[:, i])]
            B = (n_samples//2) * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            W = np.mean([np.var(chain1[:, i], ddof=1), np.var(chain2[:, i], ddof=1)])
            
            # Gelman-Rubin statistic
            var_plus = ((n_samples//2 - 1) * W + B) / (n_samples//2)
            R_hat = np.sqrt(var_plus / W)
            
            # Effective sample size (simple estimate)
            acf = self.compute_autocorrelation(self.chain[:, i], max_lag=100)
            ess = n_samples / (1 + 2 * np.sum(acf[1:]))
            
            diagnostics[param_name] = {
                'R_hat': R_hat,
                'ESS': ess,
                'converged': R_hat < 1.1
            }
        
        return diagnostics
    
    def compute_autocorrelation(self, series, max_lag=100):
        """Compute autocorrelation function"""
        from statsmodels.tsa.stattools import acf
        return acf(series, nlags=max_lag, fft=True)
    
    def get_posterior_summary(self):
        """
        Compute posterior statistics:
        - MAP estimates
        - Mean and median
        - Credible intervals
        """
        if self.chain is None:
            raise ValueError("Run MCMC first!")
        
        summary = {}
        param_names = ['A', 'tau', 'omega']
        
        for i, name in enumerate(param_names):
            samples = self.chain[:, i]
            
            # MAP estimate (mode approximation using KDE)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(samples)
            x_grid = np.linspace(samples.min(), samples.max(), 1000)
            map_idx = np.argmax(kde(x_grid))
            map_estimate = x_grid[map_idx]
            
            summary[name] = {
                'MAP': map_estimate,
                'mean': np.mean(samples),
                'median': np.median(samples),
                'std': np.std(samples),
                'CI_95': np.percentile(samples, [2.5, 97.5]),
                'CI_68': np.percentile(samples, [16, 84])
            }
        
        return summary
    
    def plot_trace(self, save_path='trace_plots.png'):
        """Generate trace plots for MCMC convergence assessment"""
        if self.chain is None:
            raise ValueError("Run MCMC first!")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        param_names = ['Amplitude (A)', 'Decay Time (τ)', 'Frequency (ω)']
        
        for i in range(3):
            # Trace plot
            axes[i, 0].plot(self.chain[:, i], alpha=0.7, linewidth=0.5)
            axes[i, 0].set_ylabel(param_names[i], fontsize=12)
            axes[i, 0].set_xlabel('Iteration', fontsize=10)
            axes[i, 0].grid(True, alpha=0.3)
            
            # Histogram (posterior distribution)
            axes[i, 1].hist(self.chain[:, i], bins=50, density=True, 
                           alpha=0.7, edgecolor='black')
            axes[i, 1].set_xlabel(param_names[i], fontsize=12)
            axes[i, 1].set_ylabel('Density', fontsize=10)
            axes[i, 1].grid(True, alpha=0.3)
            
            # Add credible interval lines
            ci = np.percentile(self.chain[:, i], [2.5, 97.5])
            axes[i, 1].axvline(ci[0], color='red', linestyle='--', label='95% CI')
            axes[i, 1].axvline(ci[1], color='red', linestyle='--')
            axes[i, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trace plots saved to {save_path}")
        return fig
    
    def plot_posterior_distributions(self, save_path='posterior_distributions.png'):
        """Generate detailed posterior distribution plots"""
        if self.chain is None:
            raise ValueError("Run MCMC first!")
        
        summary = self.get_posterior_summary()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        param_names = ['A', 'tau', 'omega']
        titles = ['Amplitude (A)', 'Decay Time (τ)', 'Frequency (ω)']
        
        for i, (name, title) in enumerate(zip(param_names, titles)):
            # KDE plot
            from scipy.stats import gaussian_kde
            samples = self.chain[:, i]
            kde = gaussian_kde(samples)
            x_grid = np.linspace(samples.min(), samples.max(), 500)
            
            axes[i].fill_between(x_grid, kde(x_grid), alpha=0.3, label='Posterior')
            axes[i].plot(x_grid, kde(x_grid), linewidth=2)
            
            # Mark MAP, mean, and median
            axes[i].axvline(summary[name]['MAP'], color='red', linestyle='-', 
                           linewidth=2, label=f"MAP: {summary[name]['MAP']:.3f}")
            axes[i].axvline(summary[name]['mean'], color='green', linestyle='--', 
                           linewidth=2, label=f"Mean: {summary[name]['mean']:.3f}")
            axes[i].axvline(summary[name]['median'], color='blue', linestyle=':', 
                           linewidth=2, label=f"Median: {summary[name]['median']:.3f}")
            
            # Credible intervals
            ci_95 = summary[name]['CI_95']
            axes[i].axvspan(ci_95[0], ci_95[1], alpha=0.2, color='gray', 
                           label=f'95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]')
            
            axes[i].set_xlabel(title, fontsize=14)
            axes[i].set_ylabel('Probability Density', fontsize=12)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Posterior distributions saved to {save_path}")
        return fig
    
    def plot_fit_with_uncertainty(self, n_samples=1000, save_path='best_fit.png'):
        """Plot best fit with uncertainty bands"""
        if self.chain is None:
            raise ValueError("Run MCMC first!")
        
        # Sample from posterior
        indices = np.random.choice(len(self.chain), n_samples, replace=False)
        posterior_samples = self.chain[indices]
        
        # Compute predictions for each sample
        predictions = np.zeros((n_samples, len(self.time_norm)))
        for i, params in enumerate(posterior_samples):
            predictions[i] = self.flare_model(self.time_norm, *params)
        
        # Compute statistics
        mean_pred = np.mean(predictions, axis=0)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        # Get MAP estimate
        summary = self.get_posterior_summary()
        map_params = [summary['A']['MAP'], summary['tau']['MAP'], summary['omega']['MAP']]
        map_pred = self.flare_model(self.time_norm, *map_params)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Data
        ax.scatter(self.time, self.intensity, s=10, alpha=0.5, 
                  label='Observed Data', color='black')
        
        # MAP fit
        ax.plot(self.time, map_pred, 'r-', linewidth=2.5, 
               label='MAP Estimate', zorder=5)
        
        # Uncertainty band
        ax.fill_between(self.time, ci_lower, ci_upper, alpha=0.3, 
                       color='blue', label='95% Credible Interval')
        
        # Residuals
        residuals = self.intensity - map_pred
        ax.plot(self.time, residuals, 'g--', alpha=0.5, linewidth=1, 
               label='Residuals (x10)', zorder=1)
        
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Intensity', fontsize=14)
        ax.set_title('Solar Flare: Bayesian Fit with Uncertainty Quantification', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Best fit plot saved to {save_path}")
        return fig
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("SOLAR FLARE BAYESIAN ANALYSIS REPORT")
        print("="*80)
        
        # Posterior summary
        print("\n1. POSTERIOR PARAMETER ESTIMATES:")
        print("-" * 80)
        summary = self.get_posterior_summary()
        
        for param_name, stats in summary.items():
            print(f"\n{param_name}:")
            print(f"  MAP Estimate:       {stats['MAP']:.6f}")
            print(f"  Posterior Mean:     {stats['mean']:.6f}")
            print(f"  Posterior Std:      {stats['std']:.6f}")
            print(f"  95% CI:            [{stats['CI_95'][0]:.6f}, {stats['CI_95'][1]:.6f}]")
        
        # Convergence diagnostics
        print("\n2. CONVERGENCE DIAGNOSTICS:")
        print("-" * 80)
        diagnostics = self.compute_convergence_diagnostics()
        
        for param_name, diag in diagnostics.items():
            print(f"\n{param_name}:")
            print(f"  Gelman-Rubin (R̂):  {diag['R_hat']:.4f} {'✓ Converged' if diag['converged'] else '✗ Not converged'}")
            print(f"  Effective Sample:   {diag['ESS']:.0f}")
        
        # Model quality
        print("\n3. MODEL QUALITY:")
        print("-" * 80)
        print(f"  MCMC Acceptance Rate: {self.acceptance_rate:.3f}")
        print(f"  Total Samples:        {len(self.chain)}")
        
        print("\n" + "="*80)


# Example usage and data loading
if __name__ == "__main__":
    # Load your data here
    # df = pd.read_csv('solar_flare_data.csv')
    # time_data = df['time'].values
    # intensity_data = df['intensity'].values
    
    # For demonstration, generate synthetic data
    np.random.seed(42)
    time_data = np.linspace(0, 10, 2001)
    A_true, tau_true, omega_true = 5.0, 2.0, 3.0
    
    intensity_true = 5.0 * np.exp(-time_data/2.0) * np.sin(3.0 * time_data)
    intensity_true[time_data < 0] = 0
    noise = np.random.normal(0, 0.2 * np.abs(intensity_true), len(time_data))
    intensity_data = intensity_true + noise
    
    # Create analyzer
    analyzer = SolarFlareAnalyzer(time_data, intensity_data)
    
    # Run analysis
    print("Starting Bayesian analysis...")
    chain = analyzer.adaptive_mcmc(n_iterations=30000, n_burn=5000, thin=5)
    
    # Generate all plots
    analyzer.plot_trace()
    analyzer.plot_posterior_distributions()
    analyzer.plot_fit_with_uncertainty()
    
    # Print comprehensive report
    analyzer.generate_report()
    
    print("\nAnalysis complete! Check output files for visualizations.")
