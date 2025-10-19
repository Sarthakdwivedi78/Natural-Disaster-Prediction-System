# Natural Disaster Prediction & Analysis Dashboard 🌍🌪️🌊

This project is an interactive web application built with Streamlit for analyzing, visualizing, and predicting natural disasters. It provides a user-friendly dashboard to explore historical disaster data and leverages machine learning to forecast future events.



## 🚀 Features

* **Interactive Dashboard:** A clean, multi-page web interface built entirely in Streamlit.
* **Geographic Visualization:** An interactive Plotly map (`scatter_map`) displaying the geographic distribution of disaster events by latitude and longitude.
* **Exploratory Data Analysis (EDA):** In-depth analysis and visualizations (using Seaborn and Matplotlib) of disaster trends, frequencies, and magnitudes.
* **Machine Learning Prediction:** Utilizes models like **XGBoost** and **scikit-learn** to predict disaster types or other key metrics based on historical data.
* **Data Handling:** Efficiently processes and analyzes datasets using Pandas.

## 🛠️ Tech Stack

* **Core Language:** Python
* **Web Framework:** Streamlit
* **Data Science & ML:** Pandas, scikit-learn, imbalanced-learn, XGBoost
* **Data Visualization:** Plotly, Matplotlib, Seaborn

## 📦 Installation & Setup

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sarthakdwivedi78/natural-disaster-prediction-system.git](https://github.com/sarthakdwivedi78/natural-disaster-prediction-system.git)
    cd natural-disaster-prediction-system
    ```

2.  **Create and activate a virtual environment:**
    * Using `venv`:
        ```bash
        # Create the environment
        python -m venv venv
        
        # Activate on macOS/Linux
        source venv/bin/activate
        
        # Activate on Windows
        .\venv\Scripts\activate
        ```
    * Using `conda`:
        ```bash
        conda create -n disaster-env python=3.11
        conda activate disaster-env
        ```

3.  **Install the required dependencies:**
    The `requirements.txt` file contains all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ How to Run

Once you have installed the dependencies, you can run the Streamlit app with a single command:

```bash<img width="1909" height="976" alt="Screenshot 2025-10-19 210626" src="https://github.com/user-attachments/assets/58bfc73d-0619-4f8f-99bd-6c85c3ead1ce" />

streamlit run app.py





<img width="1909" height="976" alt="Screenshot 2025-10-19 210626" src="https://github.com/user-attachments/assets/cad49c15-d1ba-471d-b6db-e07cd59a39fc" />


