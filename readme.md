# On Memory-augmented Gated Recurrent Unit Network

This repository contains the pre-release version of the data and Python code used in the paper "On Memory-augmented Gated Recurrent Unit Network." For detailed information about our model and implementation, please refer to the paper.

## Requirements
```
python==3.11.5
numpy==1.23.5
pandas==2.0.3
scipy==1.10.1
statsmodels==0.14.1
torch==2.0.1
baostock==0.8.9
seaborn==0.12.2
matplotlib==3.7.2
```

Readers can use the following command to install the required packages:
```bash
pip install -r requirements.txt
```

<!-- ##  Modules

<details closed><summary>data</summary>

| File                                                                                          | Summary                         |
| ---                                                                                           | ---                             |
| [get_data.ipynb](https://github.com/MerlynYang/MGRU/blob/master/utils/get_data.ipynb)                 | <code>data acquisition</code> |
| [data_preprocess.ipynb](https://github.com/MerlynYang/MGRU/blob/main/data/data_preprocess.ipynb) | <code>data cleaning and preprocess</code> |
</details>

<details closed><summary>utils</summary>

| File                                                                                          | Summary                         |
| ---                                                                                           | ---                             |
| [plot_omega_k.ipynb](https://github.com/MerlynYang/MGRU/blob/master/utils/plot_omega_k.ipynb) | <code>plot fractional differencing weights</code> |
| [MCS_test.R](https://github.com/MerlynYang/MGRU/blob/master/utils/MCS_test.R)                 | <code>MCS test</code> |
| [lassovar.R](https://github.com/MerlynYang/MGRU/blob/master/utils/lassovar.R)                 | <code>VAR benchmark</code> |

</details>

<details closed><summary>src</summary>

| File                                                                                            | Summary                         |
| ---                                                                                             | ---                             |
| [cells.py](https://github.com/MerlynYang/MGRU/blob/master/src/cells.py)                         | <code>define the cell structure of each model</code> |
| [models.py](https://github.com/MerlynYang/MGRU/blob/master/src/models.py)                       | <code>define each model</code> |
| [train.py](https://github.com/MerlynYang/MGRU/blob/master/src/train.py)                         | <code>train the model</code> |
| [cal_metrics.ipynb](https://github.com/MerlynYang/MGRU/blob/master/src/cal_metrics.ipynb)       | <code>statistical evaluation of the forecasts</code> |
| [portfolio.ipynb](https://github.com/MerlynYang/MGRU/blob/master/src/portfolio.ipynb)           | <code>economic evaluation of the forecasts</code> |
| [arfima_smi.py](https://github.com/MerlynYang/MGRU/blob/master/src/arfima_smi.py)               | <code>generate ARFIMA process</code> |
| [simulation_ARFIMA.py](https://github.com/MerlynYang/MGRU/blob/master/src/simulation_ARFIMA.py) | <code>simulation on ARFIMA process</code> |
| [simulation_ARMA.py](https://github.com/MerlynYang/MGRU/blob/master/src/simulation_ARMA.py)     | <code>simulation on ARMA process</code> |
| [utils.py](https://github.com/MerlynYang/MGRU/blob/master/src/utils.py)                         | <code>some auxiliary codes</code> |

</details>

--- -->

##  Usage
### data
To download and process the raw data (5-minute high-frequency intraday data), follow these steps:
* Run `get_data.ipynb` to download the data.
* Run `data_preprocess.ipynb` to clean and preprocess the data. This notebook also includes some basic descriptive analysis.

*Note: Processed data is available in the `data/realized_volatility/` folder. This includes the `log_rv_table.csv` (processed logarithmic realized volatility) and the `rv_table.csv` (raw realized volatility calculated directly from high-frequency data). Stock returns (raw data) for each stock during the sample period are stored in the `data` folder and named `return_table.csv`.*

### training
To reproduce the results from the paper, start by training the model using the `src/train.py` script. The generated forecasts will be saved in the `results/forecasts` directory.

For example, to generate a one-step ahead forecast ($h=1$) using the **MGRUF** model, use the following command:

```python
python src/train.py --forecast_horizon 1 --rnn_type MGRUF
```
You can modify the `forecast_horizon` and `rnn_type` parameters to produce forecasts for different models and forecast horizons.

*Note: This training process can be time-consuming. To save the time of readers, the forecasts used in our empirical analysis are saved in the `results/forecasts` folder. Each file is named using the format `{model_name}_h_{forecast_horizon}`.*

### evaluation
* Run `cal_metrics.ipynb` to assess the statistical performance of the forecasts.
* Run `portfolio.ipynb` to evaluate the economic performance of the forecasts.

### simulation
To demonstrate the finite performance of MGRU, we conduct simulations using both long memory (ARFIMA process) and short memory (ARMA process) datasets. For detailed information, please refer to the appendix of the paper.

* To evaluate the estimated long memory parameter $d$ across different data-generating processes (DGPs) and sample sizes ($T$), use the following command. For `DGP` parameter, `arfima` represents an ARFIMA(1, d, 1) process, while `fi` stands for ARFIMA(0, d, 0):
```python
python src/simulation_ARFIMA.py --d 0.1 --T 500 --DGP arfima
```
* To compare the performance of MGRUF with GRU on short memory DGPs, use the following command. In this context, `arma` represents an ARMA(1, 1) process, and `ar` stands for an AR(1) process:
```python
python src/simulation_ARMA.py --DGP arma --T 2000 --model MGRUF
```

### sentiment analysis
For data and code related to sentiment analysis, please refer to this repository : [Review classification](https://github.com/huawei-noah/noah-research/tree/master/mRNN-mLSTM).

## Contact
If you have any questions about this repository or the implementation details of the paper, please feel free to submit a new issue or contact the author:
```
Author: Maolin Yang
Email: merlynyang546@gmail.com
```