# 🚗 Tesla Stock Intelligence Dashboard

Deep learning–inspired forecasting models for Tesla stock prices using pure Python and scikit-learn 

## 📖 Description

This project explores historical Tesla stock market data and builds machine learning models to forecast future stock prices. The workflow covers data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment through an interactive Streamlit dashboard.

Unlike many stock prediction projects that depend on TensorFlow/Keras, this implementation is designed to run on Python 3.14 using only scikit-learn, making it lightweight, portable, and easy to reproduce.

### Key Features

* Historical Tesla stock analysis (2015–2024)
* Time-series feature engineering using sliding windows
* SimpleRNN-inspired forecasting model
* LSTM-inspired forecasting model
* Performance evaluation using RMSE and R² metrics
* Automated visualization generation
* Interactive Streamlit dashboard
* Python 3.14 compatible (no TensorFlow required)

---

## 📁 Project Structure

| File                           | Description                                     |
| ------------------------------ | ----------------------------------------------- |
| `TSLA.csv`                     | Historical Tesla stock data (2015–2024)         |
| `Tesla_Stock_Prediction.ipynb` | End-to-end analysis and model training notebook |
| `app.py`                       | Interactive Streamlit dashboard                 |
| `requirements.txt`             | Project dependencies                            |
| `results_summary.csv`          | Model evaluation metrics                        |
| `*.png`                        | Generated charts and visualizations             |

---

## 📊 Model Performance

| Model              | Forecast Horizon | RMSE       | R² Score  |
| ------------------ | ---------------- | ---------- | --------- |
| **LSTM-Inspired**  | **1 Day**        | **$16.30** | **0.902** |
| SimpleRNN-Inspired | 1 Day            | $19.95     | 0.853     |
| LSTM-Inspired      | 5 Days           | $23.32     | 0.800     |
| LSTM-Inspired      | 10 Days          | $36.72     | 0.509     |

### Best Result

🏆 The LSTM-inspired model achieved the strongest performance for 1-day forecasting with:

* RMSE: **$16.30**
* R² Score: **0.902**

---

## ⚙️ Why No TensorFlow?

TensorFlow currently does not support Python 3.14. To maintain compatibility with the latest Python release, this project uses `sklearn.neural_network.MLPRegressor` to emulate recurrent-style architectures.

Benefits include:

* Python 3.14 compatibility
* Adam optimizer support
* Early stopping functionality
* Lightweight dependencies
* Faster setup and execution

| Architecture       | Implementation                                   |
| ------------------ | ------------------------------------------------ |
| SimpleRNN-Inspired | `MLPRegressor(hidden_layer_sizes=(64, 32))`      |
| LSTM-Inspired      | `MLPRegressor(hidden_layer_sizes=(128, 64, 32))` |

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/tesla-stock-price-prediction.git
cd tesla-stock-price-prediction

pip install -r requirements.txt
```

---

## ▶️ Run the Dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the dashboard.

---

## 📈 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Jupyter Notebook

---

## 🔮 Future Improvements

* Real-time stock data integration
* Support for multiple stock symbols
* Hyperparameter optimization
* Transformer-based forecasting models
* Cloud deployment (Streamlit Cloud / Render)

---

## 📜 Disclaimer

This project is intended for educational and research purposes only. Stock market predictions are inherently uncertain and should not be considered financial advice.

