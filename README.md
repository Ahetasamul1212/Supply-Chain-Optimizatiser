# 🚀 **Supply Chain Optimization using Machine Learning**

## 📌 **Project Overview**

This project aims to **optimize a country's supply chain** by predicting shipment delays using **Machine Learning**. By leveraging **Random Forest Regression**, it provides insights into potential delays and helps improve logistics efficiency. The model is trained on historical shipment data and deployed with **Streamlit** for real-time predictions.

## 🔍 **Features**

✅ **Predicts shipment delays** to enhance logistics planning.\
✅ **Automated data preprocessing** (encoding, scaling, and feature selection).\
✅ **Trained using Random Forest Regressor** for accurate predictions.\
✅ **Interactive UI with Streamlit** for real-time predictions.\
✅ **Performance evaluation** using RMSE.

---

## 🏗️ **Project Structure**

```
Supply-Chain-Optimizer/
│── app.py                  # Streamlit web app for real-time predictions
│── main.ipynb              # Jupyter Notebook for training and evaluation
│── requirements.txt        # Dependencies needed to run the project
│── Supply chain logistics problem.csv  # Dataset (add your dataset here)
└── README.md               # Project documentation
```

---

## ⚡ **Installation & Setup**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Supply-Chain-Optimizer.git
cd Supply-Chain-Optimizer
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

This will launch a **web-based UI** where you can input shipment details and get predicted delay estimates.

---

## 📊 **How the Model Works**

1. **Data Preprocessing**
   - Drops irrelevant columns (e.g., Order ID, Customer details).
   - Encodes categorical features (e.g., Origin Port, Carrier).
   - Normalizes numerical features using **StandardScaler**.
2. **Model Training**
   - Uses **Random Forest Regressor** to predict shipment delays.
   - Splits data into training and testing sets.
   - Evaluates model performance using **Root Mean Squared Error (RMSE)**.
3. **Web App (Streamlit)**
   - Users input shipment details.
   - The model predicts the expected **Ship Late Day Count**.
   - Displays real-time results for better logistics planning.

---

## 📌 **Usage Example**

Once the Streamlit app is running, you can enter:

- **Encoded values** for Origin Port, Carrier, Service Level, Plant Code, Destination Port.
- **Numerical inputs** like Unit Quantity, Weight, and TPT (Transit Processing Time).

Upon clicking **Predict**, the model will provide an estimated **shipment delay in days**.

---

## 🔬 **Future Improvements**

🔹 Experiment with **XGBoost** or **Neural Networks** for improved accuracy.\
🔹 Integrate **real-time tracking** and external APIs for live predictions.\
🔹 Develop a **dashboard** with detailed analytics for logistics insights.

---

## 🤝 **Contributing**

Feel free to fork this repo, make improvements, and submit a pull request! Your contributions will help improve supply chain optimization globally.

---

## 🏆 **Acknowledgments**

Special thanks to **Machine Learning & AI researchers** for inspiring this project. 🚀

