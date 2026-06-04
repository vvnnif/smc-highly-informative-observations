Implementation of algorithms described in Svensson et al. 2017 "Learning of state-space models with highly informative
observations: a tempered Sequential Monte Carlo solution" for the seminar course "Sequential Monte Carlo Methods" in the MSc Mathematics at the VU Amsterdam.

File structure:

```bash
├── notebooks
│   ├── prototype_OLD.ipynb         # Contains original implementation of SMC sampler, MH, etc, for linear model
│   ├── smc_full_OLD.ipynb          # Contains original implementaion of SMC sampler + some old experiments
│   ├── results                     # Lots of big files and outdated experiment records
│   └── test_and_debug.ipynb        # Debugging here
│   ├── benchmark_experiments.ipynb # Experiments were performed here
├── pdf
│   ├── smc_report.pdf # The report
│   └── smc_slides.pdf # The presentation
├── plots 
├── README.md
└── src
    ├── benchmark_ssm # Benchmark toy-model
    │   ├── hi_smc.py # Implements SMC sampler
    │   ├── mh.py     # Implements Markov Chain Monte Carlo stuff
    │   ├── pf.py     # Implements particle filters
    │   └── ssm.py    # Implements state-space model definition
    ├── linear_ssm # Linear-Gaussian SSM used primarily as a prototype
    │   ├── hi_smc.py # same idea ..
    │   ├── mh.py
    │   ├── pf.py
    │   └── ssm.py
    ├── nonlinear_ssm # Non-linear toy model from the paper
    │   ├── hi_smc.py
    │   ├── mh.py
    │   ├── pf.py
    │   └── ssm.py
    └── util.py``` # Some general utilities (logsumexp, resampling, ...)