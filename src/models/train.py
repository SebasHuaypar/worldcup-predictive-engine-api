import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.features.builder import build_features_dataset

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

def train_and_save_models():
    # Train a single unified XGBoost model to predict a team's goal count
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Armamos el dataset con todas las características
    df = build_features_dataset()
    if df.empty:
        print("Empty dataset. Cannot train models.")
        return
        
    # Define the columns the model will use (must match builder.py)
    feature_cols = [
        'is_home', 'elo_self', 'elo_opponent', 'elo_diff',
        'elo_trend_self', 'elo_trend_opponent', 'elo_trend_diff',
        'market_value_self', 'market_value_opponent', 'market_value_diff', 'market_value_ratio',
        'conf_coeff_self', 'conf_coeff_opponent', 'conf_coeff_diff',
        'poss_style_self', 'poss_style_opponent', 'xg_style_self', 'xg_style_opponent',
        'rolling_gf_self_l5', 'rolling_ga_self_l5', 'rolling_gf_opp_l5', 'rolling_ga_opp_l5',
        'rolling_gf_self_l10', 'rolling_ga_self_l10', 'rolling_gf_opp_l10', 'rolling_ga_opp_l10'
    ]
    
    X = df[feature_cols]
    y = df['goals_scored']
    y_clf = df['match_outcome']
    
    # 2. Split the data: train with everything up to 2021, test on 2022+ matches
    df['date'] = pd.to_datetime(df['date'])
    train_idx = df['date'] < '2022-01-01'
    test_idx = df['date'] >= '2022-01-01'
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_train_clf, y_test_clf = y_clf[train_idx], y_clf[test_idx]
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Evaluation set size: {len(X_test)} samples")
    
    # -------------------------------------------------------------
    # A. GOAL REGRESSOR TRAINING
    # -------------------------------------------------------------
    # Hyperparameter tuning using RandomizedSearchCV
    print("\nRunning hyperparameter optimization for goals regressor with 3-fold cross-validation...")
    from sklearn.model_selection import RandomizedSearchCV
    
    base_model = XGBRegressor(
        objective='count:poisson',
        eval_metric='poisson-nloglik',
        random_state=42
    )
    
    param_distributions = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.02, 0.05, 0.1],
        'n_estimators': [50, 100, 150],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='neg_mean_poisson_deviance',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print(f"Best goals regressor hyperparameters found: {best_params}")
    
    best_eval_model = search.best_estimator_
    
    # Evaluate how well it did
    pred = best_eval_model.predict(X_test)
    pred = np.clip(pred, 0, None)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    
    print(f"\n--- Unified Model Evaluation Results (Matches since 2022) ---")
    print(f"Goals - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Train the final model with ALL the data and save it
    print("\nTraining final unified goals model on complete dataset using best hyperparameters...")
    final_model = XGBRegressor(
        objective='count:poisson',
        eval_metric='poisson-nloglik',
        random_state=42,
        **best_params
    )
    
    final_model.fit(X, y)
    
    # Save the model safely so we don't corrupt files
    model_path = os.path.join(MODELS_DIR, "model_goals.pkl")
    temp_path = model_path + ".tmp"
    
    joblib.dump(final_model, temp_path)
    os.replace(temp_path, model_path)
    print(f"Unified goals model successfully saved to: {model_path}")
    
    # -------------------------------------------------------------
    # B. OUTCOME CLASSIFIER TRAINING
    # -------------------------------------------------------------
    print("\nRunning hyperparameter optimization for outcome classifier with 3-fold cross-validation...")
    from xgboost import XGBClassifier
    
    base_classifier = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )
    
    param_distributions_clf = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.02, 0.05, 0.1],
        'n_estimators': [50, 100, 150],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    search_clf = RandomizedSearchCV(
        estimator=base_classifier,
        param_distributions=param_distributions_clf,
        n_iter=10,
        scoring='neg_log_loss',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    search_clf.fit(X_train, y_train_clf)
    best_params_clf = search_clf.best_params_
    print(f"Best classifier hyperparameters found: {best_params_clf}")
    
    best_eval_clf = search_clf.best_estimator_
    
    # Evaluate classifier
    pred_clf_probs = best_eval_clf.predict_proba(X_test)
    from sklearn.metrics import log_loss, accuracy_score
    test_log_loss = log_loss(y_test_clf, pred_clf_probs)
    pred_clf_classes = np.argmax(pred_clf_probs, axis=1)
    test_accuracy = accuracy_score(y_test_clf, pred_clf_classes)
    
    print(f"\n--- Classifier Evaluation Results (Matches since 2022) ---")
    print(f"Outcome - Log Loss: {test_log_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    
    # Train final classifier and save
    print("\nTraining final outcome classifier on complete dataset using best hyperparameters...")
    final_clf = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        **best_params_clf
    )
    final_clf.fit(X, y_clf)
    
    clf_path = os.path.join(MODELS_DIR, "model_outcome_classifier.pkl")
    temp_clf_path = clf_path + ".tmp"
    joblib.dump(final_clf, temp_clf_path)
    os.replace(temp_clf_path, clf_path)
    print(f"Classifier model successfully saved to: {clf_path}")
    
    # Delete old models if they exist to avoid confusion
    for old_model in ["model_home_goals.pkl", "model_away_goals.pkl"]:
        old_path = os.path.join(MODELS_DIR, old_model)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except Exception as e:
                print(f"Error removing old model {old_model}: {e}")
                
    print("\nAll models trained and saved successfully.")

if __name__ == "__main__":
    train_and_save_models()
