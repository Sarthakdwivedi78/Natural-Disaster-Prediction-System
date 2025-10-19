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

streamlit run app.py<img width="1225" height="721" alt="Screenshot 2025-10-19 212311" src="https://github.com/user-attachments/assets/035621ca-8415-4e93-83dc-bb43b7cf49e3" />
<img width="1762" height="878" alt="Screenshot 2025-10-19 211642" src="https://github.com/user-attachments/assets/7dbcd347-ac2e-410c-b10b-dd568262bde1" />
<img width="1766" height="972" alt="Screenshot 2025-10-19 211609" src="https://github.com/user-attachments/assets/36464a89-7e10-43c1-bda9-852da8e7e4f8" />
<img width="1915" height="906" alt="Screenshot 2025-10-19 211539" src="https://github.com/user-attachments/assets/b0a8d287-b8e2-4536-951b-f00679c8f138" />
<img width="1882" height="977" alt="Screenshot 2025-10-19 211521" src="https://github.com/user-attachments/assets/0f2e6fe9-7175-481f-a08e-f6ae96ab069d" />
<img width="1899" height="1001" alt="Screenshot 2025-10-19 211457" src="https://github.com/user-attachments/assets/44fa4e7c-e6fa-42a5-b4a6-18611b570943" />
<img width="1770" height="907" alt="Screenshot 2025-10-19 211209" src="https://github.com/user-attachments/assets/17a1120a-20b6-4add-9cda-d147e864537a" />
<img width="1865" height="953" alt="Screenshot 2025-10-19 210727" src="https://github.com/user-attachments/assets/b4965e47-a2f6-4fa0-83f9-d36063eed565" />
<img width="1905" height="974" alt="Screenshot 2025-10-19 210702" src="https://github.com/user-attachments/assets/55200e2a-c26b-45b1-8256-9882cd9dfe6c" />
<img width="1911" height="993" alt="Screenshot 2025-10-19 210648" src="https://github.com/user-attachments/assets/c7510368-28a2-4b0a-9f8d-838080048fef" />
<img width="1909" height="976" alt="Screenshot 2025-10-19 210626" src="https://github.com/user-attachments/assets/88c5b46a-f7d7-4704-be06-25489fb000aa" />

