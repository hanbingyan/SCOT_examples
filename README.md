# Distributionally robust portfolios with SCOT
 
* Notebooks `Sharpe_Ratios.ipynb` and `Worst Return Estimation.ipynb` generate
Figures and Tables in the paper.

* `return_ot.py` and `return_scot.py` are for the worst-case returns estimation. 

* `mv_ot.py` and `mv_scot.py` are for distributionally robust mean-variance
portfolios with SCOT. 
* Besides, if we set `causal=False` in Line 167 of `mv_scot.py`, we obtain
the results for the OT method with structural information incoporated by LSTM networks.
The jupyter notebooks can be used to calculate the Sharpe ratios. These ratios are
still lower than the SCOT method on average.
 