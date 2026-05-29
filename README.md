Implementation of algorithms described in Svensson et al. 2017 "Learning of state-space models with highly informative
observations: a tempered Sequential Monte Carlo solution" for the seminar "Sequential Monte Carlo Methods" in the MSc Mathematics at the VU Amsterdam

File structure:

smc-highly-informative-observations
├── notebooks
│   ├── experiments.ipynb
│   ├── prototype.ipynb
│   ├── results
│   │   ├── particle_filter_morphing_adaptive.gif
│   │   ├── particle_filter_morphing_alternative.gif
│   │   ├── particle_filter_morphing_nice.gif
│   │   ├── particle_filter_morphing_thisone.gif
│   │   ├── smc_records_2.csv
│   │   ├── smc_records.csv
│   │   └── smc_records_thisone.csv
│   └── smc_full.ipynb
├── pdf
│   └── smc_slides.pdf
├── README.md
└── src
    ├── hi_smc.py
    ├── pf.py
    ├── pmmh.py
    ├── ssm.py
    └── util.py