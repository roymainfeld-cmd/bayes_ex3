import numpy as np
from matplotlib import pyplot as plt
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    sig = model.cov
    n = model.sig

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = np.asarray(model.fit_mu)       # posterior mean (d,)
    map_cov = np.asarray(model.fit_cov)  # posterior covariance (d,d)

    H = model.h(X)
    y = np.asarray(y).reshape(-1)
    N = y.shape[0]
    sign_prior, logdet_prior = np.linalg.slogdet(sig)
    sign_post, logdet_post = np.linalg.slogdet(map_cov)

    diff = map - mu
    quad = diff.T @ np.linalg.solve(sig, diff)

    resid = y - (H @ map)
    rss = resid.T @ resid

    log_ev = (
        - (N / 2) * np.log(2 * np.pi)
        - (N / 2) * np.log(n)
        - (1 / (2 * n)) * rss
        - 0.5 * logdet_prior
        + 0.5 * logdet_post
        - 0.5 * quad
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

        # plot evidence versus degree and predicted fit
        plt.figure(figsize=(10, 6))
        plt.plot(degrees, evidence_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Degree of Polynomial', fontsize=12)
        plt.ylabel('Log-Evidence', fontsize=12)
        plt.title(f'Log-Evidence vs Polynomial Degree for Function f{i+1}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # find and highlight the degree with highest evidence
        max_idx = np.argmax(evidence_values)
        max_degree = degrees[max_idx]
        max_evidence = evidence_values[max_idx]
        plt.plot(max_degree, max_evidence, 'r*', markersize=20, label=f'Max at d={max_degree}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"Function f{i+1}: Maximum log-evidence = {max_evidence:.2f} at degree d = {max_degree}")

    # ------------------------------------------------------ section 2.2
    # # load relevant data
    # nov16 = np.load('nov162024.npy')
    # hours = np.arange(0, 24, .5)
    # train = nov16[:len(nov16) // 2]
    # hours_train = hours[:len(nov16) // 2]

    # # load prior parameters and set up basis functions
    # mu, cov = load_prior()
    # pbf = polynomial_basis_functions(7)

    # noise_vars = np.linspace(.05, 2, 100)
    # evs = np.zeros(noise_vars.shape)
    # for i, n in enumerate(noise_vars):
    #     # calculate the evidence
    #     mdl = BayesianLinearRegression(mu, cov, n, pbf)
    #     ev = log_evidence(mdl, hours_train, train)
    #     evs[i] = ev

    # # plot log-evidence versus amount of sample noise
    # plt.figure(figsize=(10, 6))
    # plt.plot(noise_vars, evs, linewidth=2)
    # plt.xlabel('Noise Variance', fontsize=12)
    # plt.ylabel('Log-Evidence', fontsize=12)
    # plt.title('Log-Evidence vs Noise Variance', fontsize=14)
    # plt.grid(True, alpha=0.3)
    
    # # find and highlight the noise variance with highest evidence
    # max_idx = np.argmax(evs)
    # max_noise = noise_vars[max_idx]
    # max_ev = evs[max_idx]
    # plt.plot(max_noise, max_ev, 'r*', markersize=20, label=f'Max at σ²={max_noise:.3f}')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # print(f"Maximum log-evidence = {max_ev:.2f} at noise variance σ² = {max_noise:.3f}")


if __name__ == '__main__':
    main()



