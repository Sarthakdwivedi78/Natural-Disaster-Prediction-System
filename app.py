import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             roc_auc_score, roc_curve, classification_report, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score, 
                             explained_variance_score, mean_absolute_percentage_error)
from imblearn.over_sampling import SMOTE
import joblib
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb

# This file should exist in the same folder
from code7 import XGBOOST_AVAILABLE

st.set_page_config(page_title="Natural Disaster Prediction", page_icon="🌍", layout="wide")
st.markdown("""
<style>
    /* ... (your main-header style) ... */
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
    }
    .metric-card {
        /* CHANGE THIS LINE for red background */
        background: #FF6B6B; /* A red color, same as the start of your main-header gradient */
        /* You might want to change the text color back to white for contrast on a red background */
        color: white; /* <--- Add this line for white text on red background */

        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        /* The border-left is already red (#FF6B6B), so it will blend in or provide a subtle outline */
        border-left: 4px solid #FF6B6B;
    }
    /* ... (your prediction-card, risk-high, risk-medium, risk-low styles) ... */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%);
        color: white;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #2ed573 0%, #7bed9f 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class DisasterPredictor:
    def __init__(self):
        # Classification models for disaster type prediction
        self.classification_models = {}
        self.best_classification_model = None
        self.best_classification_model_name = None
        
        # Regression models for magnitude prediction
        self.regression_models = {}
        self.best_regression_model = None
        self.best_regression_model_name = None
        
        self.scaler = StandardScaler()
        self.regression_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['latitude', 'longitude']
        self.classification_results = {}
        self.regression_results = {}
        self.quick_mode = False
        self.test_data = None
        
        # Store individual datasets for visualization
        self.individual_datasets = {}
    
    def validate_and_clean_data(self, df):
        """Validate and clean the dataset"""
        df_clean = df.copy()
        
        # Ensure latitude is within valid range (-90 to 90)
        df_clean = df_clean[(df_clean['latitude'] >= -90) & (df_clean['latitude'] <= 90)]
        
        # Ensure longitude is within valid range (-180 to 180)
        df_clean = df_clean[(df_clean['longitude'] >= -180) & (df_clean['longitude'] <= 180)]
        
        # Ensure magnitude is positive
        df_clean = df_clean[df_clean['mag'] > 0]
        
        # Remove any remaining NaN values
        df_clean = df_clean.dropna()
        
        return df_clean
    
    def remove_outliers_iqr(self, df, columns):
        """Remove outliers using IQR method"""
        df_clean = df.copy()
        for column in columns:
            if column in df_clean.columns:
                # Convert to numeric, replacing non-numeric values with NaN
                df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
                # Drop rows with NaN values in this column
                df_clean = df_clean.dropna(subset=[column])
                
                if len(df_clean) > 0:  # Only proceed if we have data left
                    Q1 = df_clean[column].quantile(0.25)
                    Q3 = df_clean[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        return df_clean
    
    def get_risk_level(self, magnitude, disaster_type):
        """Convert magnitude to risk level based on disaster type"""
        if disaster_type == 'earthquake':
            if magnitude >= 7.0:
                return "High"
            elif magnitude >= 5.0:
                return "Medium"
            else:
                return "Low"
        elif disaster_type == 'flood':
            if magnitude >= 4.0:
                return "High"
            elif magnitude >= 2.5:
                return "Medium"
            else:
                return "Low"
        else:  # cyclone
            if magnitude >= 4.0:
                return "High"
            elif magnitude >= 2.5:
                return "Medium"
            else:
                return "Low"
    
    def load_and_preprocess_data(self):
        try:
            # Load datasets
            earthquake_df = pd.read_csv('earthquakeUSCS.csv')
            flood_df = pd.read_csv('flood_risk_dataset_india.csv')
            cyclone_df = pd.read_csv('pacific.csv')
            
            # Process earthquake data
            earthquake_processed = earthquake_df[['latitude', 'longitude', 'mag']].copy()
            earthquake_processed['disaster_type'] = 'earthquake'
            earthquake_processed['latitude'] = pd.to_numeric(earthquake_processed['latitude'], errors='coerce')
            earthquake_processed['longitude'] = pd.to_numeric(earthquake_processed['longitude'], errors='coerce')
            earthquake_processed['mag'] = pd.to_numeric(earthquake_processed['mag'], errors='coerce')
            earthquake_processed = earthquake_processed.dropna()
            
            # Store for visualization
            self.individual_datasets['earthquake'] = earthquake_processed.copy()
            
            # Process flood data
            flood_processed = flood_df[['Latitude', 'Longitude']].copy()
            flood_processed.columns = ['latitude', 'longitude']
            flood_processed['disaster_type'] = 'flood'
            flood_processed['latitude'] = pd.to_numeric(flood_processed['latitude'], errors='coerce')
            flood_processed['longitude'] = pd.to_numeric(flood_processed['longitude'], errors='coerce')
            flood_processed['mag'] = np.random.uniform(1, 5, len(flood_processed))  # Synthetic magnitude
            flood_processed = flood_processed.dropna()
            
            # Store for visualization
            self.individual_datasets['flood'] = flood_processed.copy()
            
            # Process cyclone data (Pacific dataset)
            cyclone_processed = cyclone_df[['Latitude', 'Longitude', 'Maximum Wind']].copy()
            cyclone_processed.columns = ['latitude', 'longitude', 'mag']
            cyclone_processed['disaster_type'] = 'cyclone'
            cyclone_processed['latitude'] = pd.to_numeric(cyclone_processed['latitude'], errors='coerce')
            cyclone_processed['longitude'] = pd.to_numeric(cyclone_processed['longitude'], errors='coerce')
            cyclone_processed['mag'] = pd.to_numeric(cyclone_processed['mag'], errors='coerce')
            cyclone_processed = cyclone_processed.dropna()
            
            # Store for visualization
            self.individual_datasets['cyclone'] = cyclone_processed.copy()
            
            # Combine all datasets
            combined_df = pd.concat([earthquake_processed, flood_processed, cyclone_processed], ignore_index=True)
            
            # Validate and clean data
            combined_df = self.validate_and_clean_data(combined_df)
            
            # Remove outliers using IQR
            combined_df = self.remove_outliers_iqr(combined_df, ['latitude', 'longitude', 'mag'])
            
            # Final check for valid data
            if len(combined_df) == 0:
                st.warning("No valid data found after preprocessing. Creating synthetic data.")
                return self.create_synthetic_data()
            
            return combined_df
            
        except FileNotFoundError as e:
            st.error(f"Dataset files not found: {e}. Creating synthetic data for demonstration.")
            return self.create_synthetic_data()
        except Exception as e:
            st.error(f"Error processing data: {e}. Creating synthetic data for demonstration.")
            return self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic disaster data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data
        data = []
        
        # Earthquake data (Ring of Fire regions)
        for _ in range(n_samples//3):
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)
            mag = np.random.uniform(1, 9)
            data.append([lat, lon, mag, 'earthquake'])
        
        # Flood data (River basins, coastal areas)
        for _ in range(n_samples//3):
            lat = np.random.uniform(-50, 50)
            lon = np.random.uniform(-150, 150)
            mag = np.random.uniform(1, 5)
            data.append([lat, lon, mag, 'flood'])
        
        # Cyclone data (Tropical regions)
        for _ in range(n_samples//3):
            lat = np.random.uniform(-30, 30)
            lon = np.random.uniform(-180, 180)
            mag = np.random.uniform(1, 5)
            data.append([lat, lon, mag, 'cyclone'])
        
        df = pd.DataFrame(data, columns=['latitude', 'longitude', 'mag', 'disaster_type'])
        
        # Store individual datasets for visualization
        self.individual_datasets['earthquake'] = df[df['disaster_type'] == 'earthquake'].copy()
        self.individual_datasets['flood'] = df[df['disaster_type'] == 'flood'].copy()
        self.individual_datasets['cyclone'] = df[df['disaster_type'] == 'cyclone'].copy()
        
        # Remove outliers using IQR
        df = self.remove_outliers_iqr(df, ['latitude', 'longitude', 'mag'])
        
        return df
    
    def calculate_auc_roc(self, y_test, y_pred_proba, model_name):
        """Calculate AUC ROC with proper error handling"""
        try:
            n_classes = y_pred_proba.shape[1]
            
            # Check if all classes are present in test set
            unique_classes = np.unique(y_test)
            if len(unique_classes) < 2:
                st.warning(f"Only one class present in test set for {model_name}")
                return 0.0
            
            if n_classes == 2:
                # Binary classification
                return roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                # Multi-class classification
                return roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                
        except ValueError as e:
            if "Only one class present" in str(e):
                st.warning(f"Only one class present in test set for {model_name}")
                return 0.0
            else:
                st.warning(f"Could not calculate AUC ROC for {model_name}: {e}")
                return 0.0
        except Exception as e:
            st.warning(f"Unexpected error calculating AUC ROC for {model_name}: {e}")
            return 0.0
    
    def train_classification_models(self, df):
        """Train classification models for disaster type prediction"""
        # Prepare features and target
        X = df[['latitude', 'longitude']].values
        y = df['disaster_type'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Apply SMOTE for class imbalance
        try:
            from imblearn.over_sampling import SMOTE  # <-- ADD THIS LINE HERE
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            st.write(f"Applied SMOTE: {len(X_train)} → {len(X_train_smote)} samples")
        except Exception as e:
            st.warning(f"SMOTE failed: {e}. Using original training data.")
            X_train_smote, y_train_smote = X_train, y_train
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train_smote)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define classification models
        if self.quick_mode:
            models_params = {
                'Random Forest': {
                    'model': RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100),
                    'params': {
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5]
                    }
                },
                'Logistic Regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
                    'params': {
                        'C': [0.1, 1, 10]
                    }
                },
                'KNN': {
                    'model': KNeighborsClassifier(n_jobs=-1),
                    'params': {
                        'n_neighbors': [3, 5, 7]
                    }
                }
            }
        else:
            models_params = {
                'Random Forest': {
                    'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5]
                    }
                },
                'SVM': {
                    'model': SVC(random_state=42, probability=True),
                    'params': {
                        'C': [0.1, 1, 10],
                        'gamma': ['scale', 'auto'],
                        'kernel': ['rbf']
                    }
                },
                'Logistic Regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
                    'params': {
                        'C': [0.1, 1, 10],
                        'penalty': ['l2'],
                        'solver': ['lbfgs']
                    }
                },
                'KNN': {
                    'model': KNeighborsClassifier(n_jobs=-1),
                    'params': {
                        'n_neighbors': [3, 5, 7],
                        'weights': ['uniform', 'distance']
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingClassifier(random_state=42),
                    'params': {
                        'n_estimators': [50, 100],
                        'learning_rate': [0.1, 0.2],
                        'max_depth': [3, 5]
                    }
                }
            }
        
        # Train classification models
        best_score = 0
        for name, config in models_params.items():
            st.write(f"Training classification model: {name}")
            
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train_smote)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test_scaled)
            
            # Check if model supports predict_proba
            if hasattr(best_model, 'predict_proba'):
                y_pred_proba = best_model.predict_proba(X_test_scaled)
            else:
                # For models without predict_proba, create dummy probabilities
                n_classes = len(np.unique(y_test))
                y_pred_proba = np.zeros((len(y_test), n_classes))
                for i, pred in enumerate(y_pred):
                    y_pred_proba[i, pred] = 1.0
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Calculate AUC ROC with proper error handling
            auc_roc = self.calculate_auc_roc(y_test, y_pred_proba, name)
            
            # Store results
            self.classification_results[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'auc_roc': auc_roc,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            self.classification_models[name] = best_model
            
            if accuracy > best_score:
                best_score = accuracy
                self.best_classification_model = best_model
                self.best_classification_model_name = name
        
        return X_test_scaled, y_test
    
    def train_regression_models(self, df):
        """Train regression models for magnitude prediction"""
        # Prepare features and target
        X = df[['latitude', 'longitude']].values
        y = df['mag'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize features
        X_train_scaled = self.regression_scaler.fit_transform(X_train)
        X_test_scaled = self.regression_scaler.transform(X_test)
        
        # Define regression models
        regression_models_params = {
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            },
            'Random Forest Regressor': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'Ridge Regression': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1, 10, 100]
                }
            },
            'Lasso Regression': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': [0.1, 1, 10, 100]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            regression_models_params['XGBoost Regressor'] = {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.1, 0.2]
                }
            }
        
        # Adjust parameters for quick mode
        if self.quick_mode:
            for name, config in regression_models_params.items():
                if name == 'Random Forest Regressor':
                    config['params'] = {
                        'n_estimators': [100],
                        'max_depth': [10, 20]
                    }
                elif name == 'XGBoost Regressor':
                    config['params'] = {
                        'n_estimators': [100],
                        'max_depth': [3, 5]
                    }
        
        # Train regression models
        best_score = -np.inf
        for name, config in regression_models_params.items():
            st.write(f"Training regression model: {name}")
            
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            explained_var = explained_variance_score(y_test, y_pred)
            
            try:
                mape = mean_absolute_percentage_error(y_test, y_pred)
            except:
                mape = 0
            
            # Store results
            self.regression_results[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'explained_variance': explained_var,
                'mape': mape,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            self.regression_models[name] = best_model
            
            if r2 > best_score:
                best_score = r2
                self.best_regression_model = best_model
                self.best_regression_model_name = name
        
        return X_test_scaled, y_test
    
    def train_models(self, df):
        """Train both classification and regression models"""
        st.write("Training Regression Models...")
        X_test_reg, y_test_reg = self.train_regression_models(df) # <-- RUN REGRESSION FIRST

        st.write("Training Classification Models...")
        X_test_class, y_test_class = self.train_classification_models(df) # <-- RUN CLASSIFICATION (with SMOTE) LAST
        
        # Store test data for evaluation
        self.test_data = {
            'classification': {
                'X_test_scaled': X_test_class,
                'y_test': y_test_class
            },
            'regression': {
                'X_test_scaled': X_test_reg,
                'y_test': y_test_reg
            }
        }
        
        return X_test_class, y_test_class, X_test_reg, y_test_reg
    
    def plot_roc_curve(self):
        """Plot ROC curve for the best classification model"""
        if not self.classification_results:
            return None
        
        best_results = self.classification_results[self.best_classification_model_name]
        y_test = best_results['y_test']
        y_pred_proba = best_results['y_pred_proba']
        
        n_classes = len(self.label_encoder.classes_)
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc:.3f})',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'ROC Curve - {self.best_classification_model_name}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=600, height=500
            )
            
        else:
            # Multi-class classification
            fig = go.Figure()
            
            # Plot ROC curve for each class
            for i in range(n_classes):
                class_name = self.label_encoder.inverse_transform([i])[0]
                y_test_binary = (y_test == i).astype(int)
                y_score = y_pred_proba[:, i]
                
                try:
                    fpr, tpr, _ = roc_curve(y_test_binary, y_score)
                    auc = roc_auc_score(y_test_binary, y_score)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{class_name} (AUC = {auc:.3f})',
                        line=dict(width=2)
                    ))
                except:
                    # Skip if cannot calculate ROC for this class
                    continue
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', dash='dash')
            ))
            
            fig.update_layout(
                title=f'ROC Curves - {self.best_classification_model_name}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=800, height=600
            )
        
        return fig
    
    def plot_regression_predictions(self):
        """Plot actual vs predicted values for regression models"""
        if not self.regression_results:
            return None
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=list(self.regression_results.keys()),
            specs=[[{"secondary_y": False} for _ in range(3)],
                   [{"secondary_y": False} for _ in range(2)] + [None]]
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (name, results) in enumerate(self.regression_results.items()):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            y_test = results['y_test']
            y_pred = results['y_pred']
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=y_test, 
                    y=y_pred,
                    mode='markers',
                    name=f'{name} (R² = {results["r2"]:.3f})',
                    marker=dict(color=colors[i % len(colors)], size=4)
                ),
                row=row, col=col
            )
            
            # Perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Actual vs Predicted Magnitude - Regression Models",
            height=800
        )
        
        return fig
    
    def plot_individual_dataset_visualizations(self):
        """Create comprehensive visualizations for individual datasets including Pacific"""
        if not self.individual_datasets:
            return None
        
        visualizations = {}
        
        for dataset_name, dataset in self.individual_datasets.items():
            if len(dataset) == 0:
                continue
            
            # Create subplots for each dataset
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    f'{dataset_name.title()} Geographic Distribution',
                    f'{dataset_name.title()} Magnitude Distribution',
                    f'{dataset_name.title()} Latitude vs Magnitude',
                    f'{dataset_name.title()} Longitude vs Magnitude'
                ],
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # 1. Geographic scatter plot
            fig.add_trace(
                go.Scatter(
                    x=dataset['longitude'],
                    y=dataset['latitude'],
                    mode='markers',
                    marker=dict(
                        size=dataset['mag'] * 2,
                        color=dataset['mag'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Magnitude")
                    ),
                    name=f'{dataset_name.title()} Events',
                    text=dataset['mag'],
                    hovertemplate='<b>%{text}</b><br>Lat: %{y}<br>Lon: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. Magnitude histogram
            fig.add_trace(
                go.Histogram(
                    x=dataset['mag'],
                    nbinsx=30,
                    name=f'{dataset_name.title()} Magnitude',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            
            # 3. Latitude vs Magnitude
            fig.add_trace(
                go.Scatter(
                    x=dataset['latitude'],
                    y=dataset['mag'],
                    mode='markers',
                    marker=dict(color='red', size=4),
                    name=f'{dataset_name.title()} Lat vs Mag'
                ),
                row=2, col=1
            )
            
            # 4. Longitude vs Magnitude
            fig.add_trace(
                go.Scatter(
                    x=dataset['longitude'],
                    y=dataset['mag'],
                    mode='markers',
                    marker=dict(color='green', size=4),
                    name=f'{dataset_name.title()} Lon vs Mag'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f'{dataset_name.title()} Dataset Analysis',
                height=800,
                showlegend=False
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Longitude", row=1, col=1)
            fig.update_yaxes(title_text="Latitude", row=1, col=1)
            fig.update_xaxes(title_text="Magnitude", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_xaxes(title_text="Latitude", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude", row=2, col=1)
            fig.update_xaxes(title_text="Longitude", row=2, col=2)
            fig.update_yaxes(title_text="Magnitude", row=2, col=2)
            
            visualizations[dataset_name] = fig
        
        return visualizations
    
    def predict_disaster(self, latitude, longitude):
        """Predict disaster type and magnitude for given coordinates"""
        if self.best_classification_model is None or self.best_regression_model is None:
            return None, None, None
        
        # Prepare input
        input_data = np.array([[latitude, longitude]])
        
        # Scale input for classification
        input_scaled = self.scaler.transform(input_data)
        
        # Predict disaster type
        disaster_type_encoded = self.best_classification_model.predict(input_scaled)[0]
        disaster_type = self.label_encoder.inverse_transform([disaster_type_encoded])[0]
        
        # Get prediction probabilities
        disaster_prob = self.best_classification_model.predict_proba(input_scaled)[0]
        
        # Scale input for regression
        input_scaled_reg = self.regression_scaler.transform(input_data)
        
        # Predict magnitude
        magnitude = self.best_regression_model.predict(input_scaled_reg)[0]
        
        # Get risk level
        risk_level = self.get_risk_level(magnitude, disaster_type)
        
        return disaster_type, magnitude, risk_level
    
    def save_models(self):
        """Save trained models"""
        model_data = {
            'classification_model': self.best_classification_model,
            'regression_model': self.best_regression_model,
            'scaler': self.scaler,
            'regression_scaler': self.regression_scaler,
            'label_encoder': self.label_encoder,
            'classification_model_name': self.best_classification_model_name,
            'regression_model_name': self.best_regression_model_name
        }
        
        with open('disaster_prediction_models.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        return 'disaster_prediction_models.pkl'
    
    def load_models(self, file_path):
        """Load pre-trained models"""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_classification_model = model_data['classification_model']
            self.best_regression_model = model_data['regression_model']
            self.scaler = model_data['scaler']
            self.regression_scaler = model_data['regression_scaler']
            self.label_encoder = model_data['label_encoder']
            self.best_classification_model_name = model_data['classification_model_name']
            self.best_regression_model_name = model_data['regression_model_name']
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = DisasterPredictor()
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Natural Disaster Prediction System</h1>
        <p>Advanced Machine Learning for Disaster Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Quick mode toggle
    quick_mode = st.sidebar.checkbox("⚡ Quick Mode", value=False, 
                                     help="Enable for faster training with reduced parameters")
    st.session_state.predictor.quick_mode = quick_mode
    
    # Data loading and training section
    st.sidebar.subheader("📊 Data & Training")
    
    # Load data button
    if st.sidebar.button("📁 Load Data"):
        with st.spinner("Loading and preprocessing data..."):
            df = st.session_state.predictor.load_and_preprocess_data()
            st.session_state.data = df
            st.session_state.data_loaded = True
            st.success(f"Data loaded successfully! Shape: {df.shape}")
    
    # Train models button
    if st.sidebar.button("🚀 Train Models", disabled=not st.session_state.data_loaded):
        if st.session_state.data_loaded:
            with st.spinner("Training machine learning models..."):
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Update progress
                progress_text.text("Training classification models...")
                progress_bar.progress(25)
                
                # Train models
                X_test_class, y_test_class, X_test_reg, y_test_reg = st.session_state.predictor.train_models(st.session_state.data)
                
                progress_bar.progress(100)
                progress_text.text("Training completed!")
                
                st.session_state.models_trained = True
                st.success("Models trained successfully!")
        else:
            st.error("Please load data first!")
    
    # Model management
    st.sidebar.subheader("💾 Model Management")
    
    # Save models
    if st.sidebar.button("💾 Save Models", disabled=not st.session_state.models_trained):
        if st.session_state.models_trained:
            file_path = st.session_state.predictor.save_models()
            st.success(f"Models saved to {file_path}")
    
    # Load models
    uploaded_file = st.sidebar.file_uploader("📂 Load Pre-trained Models", type=['pkl'])
    if uploaded_file is not None:
        if st.session_state.predictor.load_models(uploaded_file):
            st.session_state.models_trained = True
            st.success("Models loaded successfully!")
    
    # Main content area
    if st.session_state.data_loaded:
        # Dataset overview
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Records</h3>
                <h2>{:,}</h2>
            </div>
            """.format(len(st.session_state.data)), unsafe_allow_html=True)
        
        with col2:
            disaster_counts = st.session_state.data['disaster_type'].value_counts()
            st.markdown("""
            <div class="metric-card">
                <h3>Disaster Types</h3>
                <h2>{}</h2>
            </div>
            """.format(len(disaster_counts)), unsafe_allow_html=True)
        
        with col3:
            avg_magnitude = st.session_state.data['mag'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Magnitude</h3>
                <h2>{:.2f}</h2>
            </div>
            """.format(avg_magnitude), unsafe_allow_html=True)
        
        with col4:
            max_magnitude = st.session_state.data['mag'].max()
            st.markdown("""
            <div class="metric-card">
                <h3>Max Magnitude</h3>
                <h2>{:.2f}</h2>
            </div>
            """.format(max_magnitude), unsafe_allow_html=True)
        
        # Disaster type distribution
        st.subheader("🔍 Disaster Type Distribution")
        disaster_counts = st.session_state.data['disaster_type'].value_counts()
        
        fig = px.pie(
            values=disaster_counts.values,
            names=disaster_counts.index,
            title="Distribution of Disaster Types"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual dataset visualizations
        st.subheader("📈 Individual Dataset Analysis")
        visualizations = st.session_state.predictor.plot_individual_dataset_visualizations()
        
        if visualizations:
            dataset_choice = st.selectbox(
                "Select Dataset to Visualize:",
                list(visualizations.keys())
            )
            
            if dataset_choice in visualizations:
                st.plotly_chart(visualizations[dataset_choice], use_container_width=True)
        
        # Geographic distribution
        st.subheader("🗺️ Geographic Distribution")
        fig = px.scatter_map(
            st.session_state.data,
            lat='latitude',
            lon='longitude',
            color='disaster_type',
            size='mag',
            hover_data=['mag'],
            title="Global Distribution of Natural Disasters",
            zoom=4
        )
        fig.update_layout(map_style="open-street-map")
      
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance section
    if st.session_state.models_trained:
        st.header("📈 Model Performance")
        
        # Classification results
        st.subheader("🎯 Classification Results")
        
        if st.session_state.predictor.classification_results:
            # Create comparison table
            classification_df = pd.DataFrame({
                'Model': list(st.session_state.predictor.classification_results.keys()),
                'Accuracy': [results['accuracy'] for results in st.session_state.predictor.classification_results.values()],
                'F1 Score': [results['f1_score'] for results in st.session_state.predictor.classification_results.values()],
                'Precision': [results['precision'] for results in st.session_state.predictor.classification_results.values()],
                'Recall': [results['recall'] for results in st.session_state.predictor.classification_results.values()],
                'AUC ROC': [results['auc_roc'] for results in st.session_state.predictor.classification_results.values()]
            })
            
            st.dataframe(classification_df.round(4))
            
            # Best model highlight
            st.success(f"🏆 Best Classification Model: {st.session_state.predictor.best_classification_model_name}")
    
        # Regression results
        st.subheader("📊 Regression Results")
        
        if st.session_state.predictor.regression_results:
            # Create comparison table
            regression_df = pd.DataFrame({
                'Model': list(st.session_state.predictor.regression_results.keys()),
                'R² Score': [results['r2'] for results in st.session_state.predictor.regression_results.values()],
                'RMSE': [results['rmse'] for results in st.session_state.predictor.regression_results.values()],
                'MAE': [results['mae'] for results in st.session_state.predictor.regression_results.values()],
                'Explained Variance': [results['explained_variance'] for results in st.session_state.predictor.regression_results.values()]
            })
            
            st.dataframe(regression_df.round(4))
            
            # Best model highlight
            st.success(f"🏆 Best Regression Model: {st.session_state.predictor.best_regression_model_name}")
    
        # Visualizations
        st.subheader("📊 Model Visualizations")
        
        # ROC Curve
        roc_fig = st.session_state.predictor.plot_roc_curve()
        if roc_fig:
            st.plotly_chart(roc_fig, use_container_width=True)
        
        # Regression predictions
        reg_fig = st.session_state.predictor.plot_regression_predictions()
        if reg_fig:
            st.plotly_chart(reg_fig, use_container_width=True)
    
    # Prediction section
    if st.session_state.models_trained:
        st.header("🔮 Disaster Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            latitude = st.number_input("🌐 Latitude", min_value=-90.0, max_value=90.0, 
                                      value=0.0, step=0.1)
        
        with col2:
            longitude = st.number_input("🌐 Longitude", min_value=-180.0, max_value=180.0, 
                                        value=0.0, step=0.1)
        
        if st.button("🔍 Predict Disaster Risk"):
            disaster_type, magnitude, risk_level = st.session_state.predictor.predict_disaster(latitude, longitude)
            
            if disaster_type and magnitude and risk_level:
                # Display prediction results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Disaster Type</h3>
                        <p>{disaster_type.title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Magnitude</h3>
                        <p>{magnitude:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    risk_class = f"risk-{risk_level.lower()}"
                    st.markdown(f"""
                    <div class="prediction-card {risk_class}">
                        <h3>Risk Level</h3>
                        <p>{risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional information
                st.subheader("📋 Prediction Details")
                
                st.info(f"""
                **Location**: {latitude}°N, {longitude}°E
                **Predicted Disaster**: {disaster_type.title()}
                **Estimated Magnitude**: {magnitude:.2f}
                **Risk Level**: {risk_level}
                **Classification Model**: {st.session_state.predictor.best_classification_model_name}
                **Regression Model**: {st.session_state.predictor.best_regression_model_name}
                """)
                
                # Risk level interpretation
                if risk_level == "High":
                    st.error("⚠️ High risk detected! Immediate attention and preparation recommended.")
                elif risk_level == "Medium":
                    st.warning("⚡ Medium risk detected. Monitor conditions and prepare accordingly.")
                else:
                    st.success("✅ Low risk detected. Normal precautions should be sufficient.")
            else:
                st.error("Prediction failed. Please ensure models are trained properly.")
    
    # Instructions for getting started
    if not st.session_state.data_loaded:
        st.header("🚀 Getting Started")
        st.info("""
        1. **Load Data**: Click 'Load Data' in the sidebar to load and preprocess the disaster datasets
        2. **Train Models**: Once data is loaded, click 'Train Models' to train machine learning models
        3. **Make Predictions**: After training, use the prediction section to assess disaster risk for any location
        4. **Save/Load Models**: Save trained models for future use or load pre-trained models
        """)
        
        st.subheader("📚 About the System")
        st.markdown("""
        This system uses advanced machine learning techniques to predict natural disasters based on geographic coordinates.
        
        **Features:**
        - **Multi-class Classification**: Predicts disaster type (earthquake, flood, cyclone)
        - **Regression Analysis**: Estimates disaster magnitude
        - **Risk Assessment**: Provides risk level classification
        - **Model Comparison**: Tests multiple algorithms and selects the best performers
        - **Interactive Visualizations**: Comprehensive data analysis and model performance charts
        - **Geographic Mapping**: Visual representation of disaster distributions
        
        **Supported Disasters:**
        - 🌍 **Earthquakes**: Based on USGS earthquake data
        - 🌊 **Floods**: Based on India flood risk dataset
        - 🌪️ **Cyclones**: Based on Pacific typhoon data
        """)

if __name__ == "__main__":
    main()
