# 文献

各アルゴリズムが依拠する一次文献。実装との差は [[deviations]]。

## 乱数

- Bernstein, D. J. (2008). *ChaCha, a variant of Salsa20*. SASC.
- Steele, G. L., Lea, D., & Flood, C. H. (2014). Fast splittable pseudorandom number generators. *OOPSLA*.
- Box, G. E. P., & Muller, M. E. (1958). A note on the generation of random normal deviates. *Annals of Mathematical Statistics* 29.
- Marsaglia, G., & Tsang, W. W. (2000). A simple method for generating gamma variables. *ACM TOMS* 26.
- Michael, J. R., Schucany, W. R., & Haas, R. W. (1976). Generating random variates using transformations with multiple roots. *The American Statistician* 30.
- Knuth, D. E. *The Art of Computer Programming*, Vol. 2 (Poisson 逆変換).
- Hörmann, W. (1993). The transformed rejection method for generating Poisson random variables. *Insurance: Mathematics and Economics* 12.
- Chambers, J. M., Mallows, C. L., & Stuck, B. W. (1976). A method for simulating stable random variables. *JASA* 71.
- Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer. (Beta = 二つの Gamma の比)

## 線形代数・最適化

- Higham, N. J. (2008). *Functions of Matrices*. SIAM. (scaling and squaring)
- Golub, G. H., & Van Loan, C. F. *Matrix Computations*. (Cholesky, 部分ピボット LU)
- Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal* 7.
- Kiefer, J. (1953). Sequential minimax search for a maximum. *Proceedings of the AMS* 4. (黄金分割)

## スキームとモデル

- Maruyama, G. (1955). Continuous Markov processes and stochastic equations. *Rendiconti del Circolo Matematico di Palermo*.
- Milstein, G. N. (1974). Approximate integration of stochastic differential equations. *Theory of Probability & Its Applications* 19.
- Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *JPE* 81.
- Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the theory of the Brownian motion. *Physical Review* 36.
- Vasicek, O. (1977). An equilibrium characterization of the term structure. *Journal of Financial Economics* 5.
- Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). A theory of the term structure of interest rates. *Econometrica* 53.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *RFS* 6.
- Hull, J., & White, A. (1990). Pricing interest-rate-derivative securities. *RFS* 3.
- Black, F., & Karasinski, P. (1991). Bond and option pricing when short rates are lognormal. *Financial Analysts Journal*.
- Chan, K. C., Karolyi, G. A., Longstaff, F. A., & Sanders, A. B. (1992). An empirical comparison of alternative models of the short-term interest rate. *Journal of Finance* 47. (CKLS)
- Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing smile risk. *Wilmott*. (SABR)
- Stein, E. M., & Stein, J. C. (1991). Stock price distributions with stochastic volatility. *RFS* 4.
- Bates, D. S. (1996). Jumps and stochastic volatility. *RFS* 9.
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *JFE* 3.
- Kou, S. G. (2002). A jump-diffusion model for option pricing. *Management Science* 48.
- Brockwell, P. J. (2001). Lévy-driven CARMA processes. *Annals of the Institute of Statistical Mathematics* 53.
- Klüppelberg, C., Lindner, A., & Maller, R. (2004). A continuous-time GARCH process… *Journal of Applied Probability* 41. (COGARCH)
- Davies, R. B., & Harte, D. S. (1987). Tests for Hurst effect. *Biometrika* 74.
- Wood, A. T. A., & Chan, G. (1994). Simulation of stationary Gaussian processes in $[0,1]^d$. *Journal of Computational and Graphical Statistics* 3.
- Mandelbrot, B. B., & Van Ness, J. W. (1968). Fractional Brownian motions, fractional noises and applications. *SIAM Review* 10.

## 推定・高頻度

- Yoshida, N. (1992). Estimation for diffusion processes from discrete observations. *Journal of Multivariate Analysis* 41.
- Kessler, M. (1997). Estimation of an ergodic diffusion from discrete observations. *Scandinavian Journal of Statistics* 24.
- Iacus, S. M., & Yoshida, N. — YUIMA (`qmle`, `lse`, `adaBayes`, `CPoint`).
- Hayashi, T., & Yoshida, N. (2005). On covariance estimation of non-synchronously observed diffusion processes. *Bernoulli* 11.
- Hayashi, T., & Yoshida, N. (2011). Nonsynchronous covariance for high-frequency financial data. (HY 漸近分散 §8.2)
- Hoffmann, M., Rosenbaum, M., & Yoshida, N. (2013). Estimation of the lead-lag parameter from non-synchronous observations. *Bernoulli* 19.
- Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and bipower variation… *Journal of Financial Econometrics* 2.
- Jacod, J., Li, Y., Mykland, P. A., Podolskij, M., & Vetter, M. (2009). Microstructure noise, realized variance, and preaveraging. *Stochastic Processes and their Applications* 119.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008). Designing realized kernels… *Econometrica* 76.
- Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005). A tale of two time scales. *JASA* 100.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread. *Journal of Finance* 39.
- Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica* 53.
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk* 3.

## 点過程

- Ogata, Y. (1981). On Lewis' simulation method for point processes. *IEEE Transactions on Information Theory* 27.
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58.
- Ozaki, T. (1979). Maximum likelihood estimation of Hawkes' self-exciting point processes. *Annals of the Institute of Statistical Mathematics* 31.
- Isham, V., & Westcott, M. (1979). A self-correcting point process. *Stochastic Processes and their Applications* 8.
- Engle, R. F., & Russell, J. R. (1998). Autoregressive conditional duration. *Econometrica* 66.
- Mercuri, L., Perchiazzo, A., & Rroji, E. — CARMA-Hawkes（強度が CARMA 状態）。

## フィルタ

- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *ASME JBE* 82.
- Kalman, R. E., & Bucy, R. S. (1961). New results in linear filtering and prediction theory. *ASME JBE* 83.
- Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum likelihood estimates of linear dynamic systems. *AIAA Journal* 3.
- Sage, A. P., & Husa, G. W. (1969). Adaptive filtering with unknown prior statistics. *JACC*.
- Bell, B. M., & Cathey, F. W. (1993). The iterated Kalman filter update as a Gauss–Newton method. *IEEE TAC* 38.
- Julier, S. J., & Uhlmann, J. K. (1997). A new extension of the Kalman filter to nonlinear systems. *AeroSense*.
- van der Merwe, R. (2004). *Sigma-Point Kalman Filters for Probabilistic Inference* (PhD).
- Arasaratnam, I., & Haykin, S. (2009). Cubature Kalman filters. *IEEE TAC* 54.
- Evensen, G. (2003). The ensemble Kalman filter. *Ocean Dynamics* 53.
- Alspach, D. L., & Sorenson, H. W. (1972). Nonlinear Bayesian estimation using Gaussian sum approximations. *IEEE TAC* 17.
- Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F* 140.
- Pitt, M. K., & Shephard, N. (1999). Filtering via simulation: Auxiliary particle filters. *JASA* 94.
- Musso, C., Oudjane, N., & Le Gland, F. (2001). Improving regularised particle filters. In *Sequential Monte Carlo Methods in Practice*.
- van der Merwe, R., Doucet, A., de Freitas, N., & Wan, E. (2000). The unscented particle filter. *NIPS*.
- Godsill, S. J., Doucet, A., & West, M. (2004). Monte Carlo smoothing for nonlinear time series. *JASA* 99.
- Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian nonlinear state space models. *JCGS* 5.
- Doucet, A., de Freitas, N., & Gordon, N. (2001). *Sequential Monte Carlo Methods in Practice*. Springer.

## 制御・Malliavin

- Merton, R. C. (1971). Optimum consumption and portfolio rules… *JET* 3.
- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal* 35.
- Fleming, W. H., & Soner, H. M. *Controlled Markov Processes and Viscosity Solutions*.
- Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L., & Touzi, N. (1999). Applications of Malliavin calculus to Monte Carlo methods in finance. *Finance and Stochastics* 3.

## ノンパラメトリックベイズ

- Ferguson, T. S. (1973). A Bayesian analysis of some nonparametric problems. *Annals of Statistics* 1.
- Blackwell, D., & MacQueen, J. B. (1973). Ferguson distributions via Pólya urn schemes. *Annals of Statistics* 1.
- Sethuraman, J. (1994). A constructive definition of Dirichlet priors. *Statistica Sinica* 4.
- Aldous, D. J. (1985). Exchangeability and related topics. *École d'Été de Probabilités de Saint-Flour XIII*.
- Antoniak, C. E. (1974). Mixtures of Dirichlet processes with applications… *Annals of Statistics* 2.
- Escobar, M. D., & West, M. (1995). Bayesian density estimation and inference using mixtures. *JASA* 90.
- Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. *JCGS* 9.
- Perman, M., Pitman, J., & Yor, M. (1992). Size-biased sampling of Poisson point processes and excursions. *Probability Theory and Related Fields* 92.
- Pitman, J., & Yor, M. (1997). The two-parameter Poisson–Dirichlet distribution derived from a stable subordinator. *Annals of Probability* 25.
- Ishwaran, H., & James, L. F. (2001). Gibbs sampling methods for stick-breaking priors. *JASA* 96.
- Hjort, N. L. (1990). Nonparametric Bayes estimators based on beta processes in models for life history data. *Annals of Statistics* 18.
- Thibaux, R., & Jordan, M. I. (2007). Hierarchical Beta processes and the Indian buffet process. *AISTATS*.
- Griffiths, T. L., & Ghahramani, Z. (2005). Infinite latent feature models and the Indian buffet process. *NIPS* 18 / Gatsby TR 2005-001.
- Griffiths, T. L., & Ghahramani, Z. (2011). The Indian Buffet Process: An Introduction and Review. *JMLR* 12.
- Teh, Y. W., Görür, D., & Ghahramani, Z. (2007). Stick-breaking construction for the Indian buffet process. *AISTATS*.
- Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet processes. *JASA* 101.

## 金融・エネルギー・粒子推定

- Shreve, S. E. *Stochastic Calculus for Finance* I / II. Springer.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: a simplified approach. *JFE* 7.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation. *RFS* 14.
- Andersen, L., & Broadie, M. (2004). Primal-dual simulation algorithm for pricing multidimensional American options. *Management Science* 50.
- Dixit, A. K., & Pindyck, R. S. (1994). *Investment under Uncertainty*. Princeton.
- McDonald, R., & Siegel, D. (1986). The value of waiting to invest. *QJE* 101.
- Schwartz, E., & Smith, J. E. (2000). Short-term variations and long-term dynamics in commodity prices. *Management Science* 46.
- Lucia, J. J., & Schwartz, E. S. (2002). Electricity prices and power derivatives. *Review of Derivatives Research* 5.
- Cartea, Á., & Figueroa, M. G. (2005). Pricing in electricity markets. *Applied Mathematical Finance* 12.
- Gibson, R., & Schwartz, E. S. (1990). Stochastic convenience yield and the pricing of oil contingent claims. *Journal of Finance* 45.
- Schwartz, E. S. (1997). The stochastic behavior of commodity prices. *Journal of Finance* 52.
- Margrabe, W. (1978). The value of an option to exchange one asset for another. *Journal of Finance* 33.
- Cont, R., & Voltchkova, E. (2005). A finite difference scheme for option pricing in jump diffusion and exponential Lévy models. *SIAM J. Numer. Anal.* 43.
- Jaillet, P., Ronn, E. I., & Tompaidis, S. (2004). Valuation of commodity-based swing options. *Management Science* 50.
- Haugh, M. B., & Kogan, L. (2004). Pricing American options: a duality approach. *Operations Research* 52.
- Kamrad, B., & Ritchken, P. (1991). Multinomial approximating models for options with $k$ state variables. *Management Science* 37.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *JCF* 11.
- Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series smoothing and forecasting using the EM algorithm. *J. Time Series Analysis* 3.
- Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov chain Monte Carlo methods. *JRSS B* 72.
- Chopin, N., Jacob, P. E., & Papaspiliopoulos, O. (2013). SMC². *JRSS B* 75.
- Lee, S. S., & Mykland, P. A. (2008). Jumps in financial markets. *RFS* 21.
- Ogata, Y. (1988). Statistical models for earthquake occurrences and residual analysis. *JASA* 83.
