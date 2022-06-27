# Forecasting and Analyzing Financial Markets with Deep Learning Methods

*Yuriy Sosnin, Vladimir Pyrlik (supervisor)*

Bachelor's thesis for degree in Economics at HSE University, Saint-Petersburg (2022).

## Abstract

Deep Learning is a powerful Machine Learning tool, especially suitable for complex, nonlinear environments, like Natural Language Processing, Computer Vision, or Time Series forecasting. In recent years, there was a rise in applications of Deep learning to Financial Time Series. However, this stream of literature is inherently technical. Machine Learning specialists view financial data as one of time series domains, mostly uninterested in theoretical considerations or meaningful interpretation of results. We aim to shorten this gap.

We deploy LSTM, CNN and CNN-LSTM architectures in the task of forecasting daily return of Russian Exchange Index (IMOEX), and propose the use of 6 groups of macroeconomic variables and index constitutes to try enhance models' performance. We are unable to outperform zero naïve predictor; however, we find that all of 6 groups (Underlying Stocks, Sectoral Indexes, Bonds, Commodities, Exchange rates and Foreign market indexes) improve models' forecast quality.
In addition to this, we employ novel approach to Machine Learning interpretation, using SHAP values as measures of feature importance. We detect the effect of gold on Russian stock market during our considered period of late 2019, as well as several other interesting patterns.
