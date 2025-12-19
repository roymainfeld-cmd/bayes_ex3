import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior
import os


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    mu = np.asarray(model.mu).reshape(-1)       
    sig = np.asarray(model.cov)
    sigma2 = float(model.sig)


    model.fit(X, y)
    mu_post = np.asarray(model.fit_mu).reshape(-1)     
    sig_post = np.asarray(model.fit_cov)              

    H = model.h(X)
    y = np.asarray(y).reshape(-1)
    N = int(y.shape[0])
    p = int(mu.shape[0]) 

    if sigma2 <= 0:
        return float("-inf")

    sign_prior, logdet_prior = np.linalg.slogdet(sig)
    sign_post, logdet_post = np.linalg.slogdet(sig_post)
    if sign_prior <= 0 or sign_post <= 0:
        return float("-inf")

    term_det = 0.5 * (logdet_post - logdet_prior)

    diff = mu_post - mu
    term_quad = diff.T @ np.linalg.solve(sig, diff)

    resid = y - (H @ mu_post)
    term_rss = resid.T @ resid

    log_ev = (
        term_det
        - 0.5 * (term_quad + (1.0 / sigma2) * term_rss + N * np.log(sigma2))
        - (p / 2.0) * np.log(2.0 * np.pi)
    )

    return float(log_ev)


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x ** 2 - 1
    f2 = lambda x: (-x ** 2 + 10 * x ** 3 + 50 * np.sin(x / 6) + 10) / 100
    f3 = lambda x: (.5 * x ** 6 - .75 * x ** 4 + 2.75 * x ** 2) / 50
    f4 = lambda x: 5 / (1 + np.exp(-4 * x)) - (x - 2 > 0) * x
    f5 = lambda x: 1 * (np.cos(x * 4) + 4 * np.abs(x - 2))
    functions = [f1, f2, f3, f4, f5]

    noise_var = .25
    x = np.linspace(-3, 3, 500)

    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alpha = 1

    # create folders for plots
    os.makedirs('Q3_plots', exist_ok=True)
    os.makedirs('Q3_models_plots', exist_ok=True)
    os.makedirs('Q5_plots', exist_ok=True)

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))
        
        evidence_values = []

        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            evidence_values.append(ev)

        plt.figure(figsize=(10, 6))
        plt.plot(degrees, evidence_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Degree of Polynomial', fontsize=12)
        plt.ylabel('Log-Evidence', fontsize=12)
        plt.title(f'Log-Evidence vs Polynomial Degree for Function f{i+1}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        max_idx = np.argmax(evidence_values)
        max_degree = degrees[max_idx]
        max_evidence = evidence_values[max_idx]
        plt.plot(max_degree, max_evidence, 'g*', markersize=20, label=f'Best at d={max_degree}')
        
        min_idx = np.argmin(evidence_values)
        min_degree = degrees[min_idx]
        min_evidence = evidence_values[min_idx]
        plt.plot(min_degree, min_evidence, 'rx', markersize=20, linewidth=3, label=f'Worst at d={min_degree}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'Q3_plots/log_evidence_function_f{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Function f{i+1}: Maximum log-evidence = {max_evidence:.2f} at degree d = {max_degree}")
        print(f"Function f{i+1}: Minimum log-evidence = {min_evidence:.2f} at degree d = {min_degree}")
        print(f"  Plot saved as: Q3_plots/log_evidence_function_f{i+1}.png")
        
        plt.figure(figsize=(12, 7))
        
        pbf_best = polynomial_basis_functions(max_degree)
        mean_best, cov_best = np.zeros(max_degree + 1), np.eye(max_degree + 1) * alpha
        model_best = BayesianLinearRegression(mean_best, cov_best, noise_var, pbf_best)
        model_best.fit(x, y)
        
        pbf_worst = polynomial_basis_functions(min_degree)
        mean_worst, cov_worst = np.zeros(min_degree + 1), np.eye(min_degree + 1) * alpha
        model_worst = BayesianLinearRegression(mean_worst, cov_worst, noise_var, pbf_worst)
        model_worst.fit(x, y)
        
        x_plot = np.linspace(-3, 3, 1000)
        y_pred_best = model_best.predict(x_plot)
        y_std_best = model_best.predict_std(x_plot)
        y_pred_worst = model_worst.predict(x_plot)
        y_std_worst = model_worst.predict_std(x_plot)
        
        plt.scatter(x, y, alpha=0.3, s=10, label='Data', color='gray', zorder=1)
        plt.plot(x_plot, f(x_plot), 'k--', linewidth=2.5, label='True function', alpha=0.8, zorder=5)
        
        plt.plot(x_plot, y_pred_best, 'g-', linewidth=2.5, label=f'Best model (d={max_degree})', zorder=4)
        plt.fill_between(x_plot, y_pred_best - 2*y_std_best, y_pred_best + 2*y_std_best, 
                         alpha=0.2, color='green', label=f'Best 95% confidence', zorder=2)
        
        plt.plot(x_plot, y_pred_worst, 'r-', linewidth=2.5, label=f'Worst model (d={min_degree})', zorder=3)
        plt.fill_between(x_plot, y_pred_worst - 2*y_std_worst, y_pred_worst + 2*y_std_worst, 
                         alpha=0.2, color='red', label=f'Worst 95% confidence', zorder=2)
        
        plt.xlabel('x', fontsize=13)
        plt.ylabel('y', fontsize=13)
        plt.title(f'Function f{i+1}: Best (d={max_degree}, log-ev={max_evidence:.2f}) vs Worst (d={min_degree}, log-ev={min_evidence:.2f})', fontsize=13)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'Q3_models_plots/best_worst_models_f{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Model comparison plot saved as: Q3_models_plots/best_worst_models_f{i+1}.png")

    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load('nov162024.npy')
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        ev = log_evidence(mdl, hours_train, train)
        evs[i] = ev

    # plot log-evidence versus amount of sample noise
    plt.figure(figsize=(10, 6))
    plt.plot(noise_vars, evs, linewidth=2)
    plt.xlabel('Noise Variance', fontsize=12)
    plt.ylabel('Log-Evidence', fontsize=12)
    plt.title('Log-Evidence vs Noise Variance', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    max_idx = np.argmax(evs)
    max_noise = noise_vars[max_idx]
    max_ev = evs[max_idx]
    plt.plot(max_noise, max_ev, 'r*', markersize=20, label=f'Max at σ²={max_noise:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Q5_plots/log_evidence_vs_noise_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSection 2.2 - Temperature Data:")
    print(f"Maximum log-evidence = {max_ev:.2f} at noise variance = {max_noise:.3f}")
    print(f"Plot saved as: Q5_plots/log_evidence_vs_noise_variance.png")


if __name__ == '__main__':
    main()



