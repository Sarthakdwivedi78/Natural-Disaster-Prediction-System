# Natural Disaster Prediction & Analysis Dashboard 🌍🌪️🌊

<img width="1225" height="721" alt="Screenshot 2025-10-19 212311" src="https://github.com/user-attachments/assets/22a0ee4c-33cb-4f99-a1da-047e9d84c9c3" />

This project is an interactive web application built with Streamlit for analyzing, visualizing, and predicting natural disasters. It provides a user-friendly dashboard to explore historical disaster data and leverages machine learning to forecast future events.


<img width="1000" height="976" alt="Screenshot 2025-10-19 210626" src="https://github.com/user-attachments/assets/7ae5bdd0-2e29-417c-98af-af7b2a237245" />



<img width="1000" height="993" alt="Screenshot 2025-10-19 210648" src="https://github.com/user-attachments/assets/991edd38-7f1e-4e0c-a7b9-58329df9b807" />



<img width="1865" height="953" alt="Screenshot 2025-10-19 210727" src="https://github.com/user-attachments/assets/3bd7df1e-f5e5-4ff7-b952-020b86c3f551" />



<img width="1905" height="974" alt="Screenshot 2025-10-19 210702" src="https://github.com/user-attachments/assets/6e3f44c6-2545-4d81-90a5-d3b7f0d19fb7" />







## 🚀 Features

* **Interactive Dashboard:** A clean, multi-page web interface built entirely in Streamlit.
* **Geographic Visualization:** An interactive Plotly map (`scatter_map`) displaying the geographic distribution of disaster events by latitude and longitude.
* **Exploratory Data Analysis (EDA):** In-depth analysis and visualizations (using Seaborn and Matplotlib) of disaster trends, frequencies, and magnitudes.
* **Machine Learning Prediction:** Utilizes models like **XGBoost** and **scikit-learn** to predict disaster types or other key metrics based on historical data.
* **Data Handling:** Efficiently processes and analyzes datasets using Pandas.



| <img src="https://github.com/user-attachments/assets/1a5076fe-5b2e-475a-ae00-003c88f35571" alt="Screenshot 1" width="450"> | <img src="https://github.com/user-attachments/assets/0e04336b-5481-44ff-a799-086dfe5a2f7e" alt="Screenshot 2" width="450"> |



| <img src="https://github.com/user-attachments/assets/fa1175a4-ff2b-4852-87a9-a316357acff7" alt="Screenshot 3" width="450"> | <img src="https://github.com/user-attachments/assets/44acb4c8-89cd-460a-ae80-0f1a27346424" alt="Screenshot 4" width="450"> |


## 🛠️ Tech Stack

* **Core Language:** Python
* **Web Framework:** Streamlit
* **Data Science & ML:** Pandas, scikit-learn, imbalanced-learn, XGBoost
* **Data Visualization:** Plotly, Matplotlib, Seaborn


<img width="1762" height="878" alt="Screenshot 2025-10-19 211642" src="https://github.com/user-attachments/assets/acde0db5-dd56-4e68-be12-4bb5300d74a9" />
<img width="1766" height="972" alt="Screenshot 2025-10-19 211609" src="https://github.com/user-attachments/assets/8947d9be-3edf-4581-85b8-901d0be11065" />


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

```bash
streamlit run app.py
