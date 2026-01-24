"""
Tree-based machine learning for bank account fraud detection.

Uses SMOTENC inside cross-validation to avoid data leakage.
Tree-based models (RandomForest, XGBoost): no feature scaling.
"""

import pickle
from collections import Counter
from pathlib import Path
import os

import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.constants import TARGET_COL, DATA_DIR, RESULTS_DIR
from src.load_data import get_data, preprocess_global
from src.evaluation import MLMetricsEvaluator, ValueAwareEvaluator


def _get_encoded_feature_indices(X: pd.DataFrame) -> list[int]:
    """
    Indices of encoded (categorical-like) features: 2 <= nunique < 10.
    Matches: X_train.columns.get_loc(str(feature)) for feature in ... if 2 <= nunique < 10
    """
    return [
        X.columns.get_loc(str(f))
        for f in X.columns
        if 2 <= X[f].nunique() < 10
    ]


def _get_data_path() -> Path:
    """Resolve data path: prefer data/2/Base.csv (Kaggle layout), else data/Base.csv."""
    p = DATA_DIR / "2" / "Base.csv"
    if p.exists():
        return p
    return DATA_DIR / "Base.csv"


def prepare_data(
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    Load data, preprocess, and split into train/test (75/25 split).
    
    Note: No resampling is applied here. SMOTENC is applied inside cross-validation
    to avoid data leakage.

    Args:
        test_size: Proportion of data to use for testing (default 0.25, i.e. 75/25 split)
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test, encoded_features (column indices)
    """
    # Load data
    path = _get_data_path()
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = get_data(DATA_DIR)
    
    # Preprocess globally
    df = preprocess_global(df)
    
    # Drop month column if present (not needed for simple split)
    if 'month' in df.columns:
        df = df.drop(columns=['month'])
    
    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set samples per class: {Counter(y_train)}")
    print(f"Test set samples per class: {Counter(y_test)}")

    encoded_features = _get_encoded_feature_indices(X_train)

    return X_train, X_test, y_train, y_test, encoded_features


def train_classifier(classifier, param_dist, X_train, y_train, encoded_features, search_type='random'):
    """
    Train a classifier using SMOTENC inside cross-validation to avoid data leakage.
    
    Parameters:
    -----------
    classifier : sklearn estimator
        The estimator to train.
    param_dist : dict
        The distribution of parameters to search over. For pipeline, use format:
        {'classifier__param_name': [values]} (e.g., {'randomforestclassifier__n_estimators': [100, 200]})
    X_train : pandas DataFrame
        The training features.
    y_train : pandas Series
        The training target.
    encoded_features : list
        A list of encoded categorical feature indices (for SMOTENC).
    search_type : str, optional
        The type of hyperparameter search to perform. Must be either 'random' or 'grid'. 
        Defaults to 'random'.
    
    Returns:
    --------
    A fitted RandomizedSearchCV or GridSearchCV object with the best model.
    """
    # Apply SMOTENC to deal with imbalanced classes
    smote_nc = SMOTENC(categorical_features=encoded_features, sampling_strategy='minority', random_state=42)
    
    # Create pipeline with SMOTENC and the classifier
    pipeline = make_pipeline(smote_nc, classifier)
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform hyperparameter search
    if search_type == 'random':
        search_cv = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=15,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            random_state=42
        )
    elif search_type == 'grid':
        search_cv = GridSearchCV(
            estimator=pipeline,
            param_grid=param_dist,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv
        )
    else:
        raise ValueError('search_type must be either "random" or "grid"')
    
    # Fit the model and return the trained classifier
    search_cv.fit(X_train, y_train)
    
    return search_cv


def train_and_save_classifiers(
    test_size: float = 0.25,
    random_state: int = 42,
    models_dir: str = "models",
    search_type: str = 'random',
):
    """
    Train tree-based classifiers using SMOTENC inside cross-validation and save them as pickle files.
    
    ⚠️  WARNING: This function performs hyperparameter tuning with cross-validation and can take
    SEVERAL HOURS to complete. Models are already saved in the models/ directory.
    DO NOT rerun this unless you need to retrain with different parameters.
    
    To use existing models, call evaluate_on_test_set() instead, which automatically loads
    saved models from the models/ directory.
    
    Args:
        test_size: Proportion of data to use for testing (default 0.25, i.e. 75/25 split)
        random_state: Random seed for reproducibility (default 42)
        models_dir: Directory to save models (default "models")
        search_type: Type of hyperparameter search ('random' or 'grid', default 'random')
    
    Returns:
        tuple: (fitted_models_dict, X_test, y_test) - Dictionary of fitted CV search objects and test set
    """
    print("=" * 80)
    print("⚠️  WARNING: HYPERPARAMETER TUNING - THIS WILL TAKE SEVERAL HOURS ⚠️")
    print("=" * 80)
    print("This function performs RandomizedSearchCV/GridSearchCV with cross-validation.")
    print("Expected runtime: 2-6+ hours depending on your hardware.")
    print("\nModels are already saved in the models/ directory.")
    print("To use existing models, call evaluate_on_test_set() instead.")
    print("=" * 80)
    
    # Check if models already exist
    model_filenames = {
        "RandomForest": "rf_model.pkl",
        "XGBoost": "xgb_model.pkl",
    }
    existing_models = []
    for name, filename in model_filenames.items():
        path = Path(models_dir) / filename
        if path.exists():
            existing_models.append(name)
    
    if existing_models:
        print(f"\n⚠️  WARNING: Models already exist for: {', '.join(existing_models)}")
        print("These will be OVERWRITTEN if you continue.")
        print("Press Ctrl+C within 10 seconds to cancel...")
        import time
        time.sleep(10)
        print("\nContinuing with training...\n")
    
    print("=" * 60)
    print("Training Tree-Based Classifiers with Cross-Validation")
    print("=" * 60)
    
    # Prepare data with train/test split (no resampling - SMOTENC applied in CV)
    X_train, X_test, y_train, y_test, encoded_features = prepare_data(
        test_size=test_size,
        random_state=random_state,
    )
    print(f"Encoded feature indices (2 <= nunique < 10): {encoded_features}")

    # Define models and their parameter grids
    models_config = {
        "RandomForest": {
            "classifier": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_dist": {
                'randomforestclassifier__n_estimators': [20, 40, 60, 80, 100],
                'randomforestclassifier__criterion': ['gini', 'entropy'],
                'randomforestclassifier__max_depth': [2, 4, 6, 8, 10],
                'randomforestclassifier__max_features': ['sqrt', 'log2']
            }
        },
        "XGBoost": {
            "classifier": XGBClassifier(random_state=42, n_jobs=-1),
            "param_dist": {
                'xgbclassifier__n_estimators': [20, 40, 60, 80, 100],
                'xgbclassifier__max_depth': [2, 4, 6, 8, 10],
                'xgbclassifier__learning_rate': [0.05, 0.1, 0.15, 0.20],
                'xgbclassifier__min_child_weight': [1, 2, 3, 4],
                'xgbclassifier__subsample': [0.6, 0.8, 1.0],
                'xgbclassifier__colsample_bytree': [0.6, 0.8, 1.0]
            }
        }
    }

    fitted = {}
    ml_evaluator = MLMetricsEvaluator()
    
    for name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training {name} with {search_type} search...")
        print(f"{'='*60}")
        
        search_cv = train_classifier(
            classifier=config["classifier"],
            param_dist=config["param_dist"],
            X_train=X_train,
            y_train=y_train,
            encoded_features=encoded_features,
            search_type=search_type,
        )
        
        fitted[name] = search_cv
        
        print(f"\nBest CV score (ROC-AUC): {search_cv.best_score_:.4f}")
        print(f"Best parameters: {search_cv.best_params_}")
        
        # Print CV results
        print(f"\nCV Results for {name}:")
        ml_evaluator.print_cv_results(search_cv)

    # Save each trained model as pickle (save only the classifier, not the pipeline)
    os.makedirs(models_dir, exist_ok=True)
    model_filenames = {
        "RandomForest": "rf_model.pkl",
        "XGBoost": "xgb_model.pkl",
    }
    for name, search_cv in fitted.items():
        # Extract the classifier from the pipeline (last step is the classifier, first is SMOTENC)
        best_pipeline = search_cv.best_estimator_
        # Get the last step which is the classifier
        step_names = list(best_pipeline.named_steps.keys())
        classifier_step_name = step_names[-1]  # Last step is the classifier
        best_classifier = best_pipeline.named_steps[classifier_step_name]
        
        path = Path(models_dir) / model_filenames[name]
        with open(path, "wb") as f:
            pickle.dump(best_classifier, f)
        print(f"\nSaved {name} best model to {path}")

    return fitted, X_test, y_test


def train_logistic_regression(
    test_size: float = 0.25,
    random_state: int = 42,
    models_dir: str = "models",
):
    """
    Train logistic regression on the same train/test split as tree-based models.
    
    Uses SMOTENC for resampling and saves the model to models/lr_model.pkl.
    
    Args:
        test_size: Proportion of data to use for testing (default 0.25, i.e. 75/25 split)
        random_state: Random seed for reproducibility (default 42)
        models_dir: Directory to save model (default "models")
    
    Returns:
        tuple: (fitted_model, X_test, y_test) - Trained logistic regression model and test set
    """
    print("=" * 60)
    print("Training Logistic Regression")
    print("=" * 60)
    
    # Prepare data with same train/test split
    X_train, X_test, y_train, y_test, encoded_features = prepare_data(
        test_size=test_size,
        random_state=random_state,
    )
    
    # Apply SMOTENC for resampling
    smote_nc = SMOTENC(categorical_features=encoded_features, sampling_strategy='minority', random_state=random_state)
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)
    
    # Train logistic regression
    model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=random_state, max_iter=1000)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Save model
    os.makedirs(models_dir, exist_ok=True)
    path = Path(models_dir) / "lr_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved Logistic Regression model to {path}")
    
    return model, X_test, y_test


def evaluate_on_test_set(
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    models_dir: str = "models",
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    Evaluate models on the test set.
    
    This function automatically loads existing trained models from the models/ directory.
    No training is performed - models are loaded from pickle files.
    
    Runs both MLMetricsEvaluator (test_classifier, plot_roc_curves) and
    ValueAwareEvaluator (cost-based metrics) on the test set.
    
    Args:
        X_test: Test feature data. If None, will prepare test set using prepare_data().
        y_test: Test target labels. If None, will prepare test set using prepare_data().
        models_dir: Directory containing saved model pickle files (default "models")
        test_size: Proportion of data to use for testing (default 0.25). Only used if X_test/y_test are None.
        random_state: Random seed for reproducibility. Only used if X_test/y_test are None.
    
    Raises:
        ValueError: If no models are found in the models_dir directory.
    """
    print("=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    print("Loading existing models from models/ directory...")
    
    # Prepare test set if not provided
    if X_test is None or y_test is None:
        print("Preparing test set...")
        _, X_test, _, y_test, _ = prepare_data(
            test_size=test_size,
            random_state=random_state,
        )
        print(f"Test set prepared: {len(X_test)} samples")
    
    # Load models
    model_filenames = {
        "LogisticRegression": "lr_model.pkl",
        "RandomForest": "rf_model.pkl",
        "XGBoost": "xgb_model.pkl",
    }
    
    fitted = {}
    for name, filename in model_filenames.items():
        path = Path(models_dir) / filename
        if path.exists():
            with open(path, "rb") as f:
                fitted[name] = pickle.load(f)
            print(f"Loaded {name} from {path}")
        else:
            print(f"Warning: {name} model not found at {path}")
    
    if not fitted:
        raise ValueError("No models found to evaluate.")
    
    # ML Metrics Evaluation
    print("\n" + "=" * 60)
    print("ML Metrics Evaluation")
    print("=" * 60)
    
    ml_evaluator = MLMetricsEvaluator()
    value_evaluator = ValueAwareEvaluator() if "proposed_credit_limit" in X_test.columns else None
    
    fpr_list = []
    tpr_list = []
    label_list = []
    va_points = []  # Value-aware threshold points for ROC plot
    static_05_points = []  # Static 0.5 threshold points for ROC plot
    va_thresholds = []  # Value-aware threshold values for annotation
    optimal_thresholds = {}  # Store optimal thresholds for each model
    
    for name, model in fitted.items():
        print(f"\n--- Evaluating {name} ---")
        result = ml_evaluator.test_classifier(model, X_test, y_test, value_evaluator=value_evaluator)
        fpr, tpr, _, _, _, va_threshold, va_fpr_tpr, static_05_fpr_tpr = result
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        label_list.append(name)
        va_points.append(va_fpr_tpr)
        static_05_points.append(static_05_fpr_tpr)
        va_thresholds.append(va_threshold)
        optimal_thresholds[name] = va_threshold  # Store the optimal threshold
    
    # Plot ROC curves with value-aware threshold points and static 0.5 points
    print("\n--- ROC Curves Comparison ---")
    ml_evaluator.plot_roc_curves(fpr_list, tpr_list, label_list, va_points=va_points, 
                                 static_05_points=static_05_points, va_thresholds=va_thresholds)
    
    # Value-Aware Evaluation (if proposed_credit_limit is available)
    if "proposed_credit_limit" in X_test.columns:
        print("\n" + "=" * 60)
        print("Value-Aware Evaluation")
        print("=" * 60)
        
        value_evaluator = ValueAwareEvaluator()
        
        for name, model in fitted.items():
            print(f"\n--- {name} ---")
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Static threshold (0.5)
            results_static = value_evaluator.evaluate(
                y_true=y_test,
                y_pred_prob=y_prob,
                X_features=X_test,
                threshold_method="static",
                static_threshold=0.5,
            )
            print(f"Static (0.5) Threshold:")
            print(f"  Total Bank Loss: ${results_static['Total_Bank_Loss_($)']:,.2f}")
            print(f"  Fraud Loss: ${results_static['Fraud_Loss_($)']:,.2f}")
            print(f"  False Alarm Cost: ${results_static['False_Alarm_Cost_($)']:,.2f}")
            print(f"  Fraud Caught: ${results_static['Fraud_Caught_($)']:,.2f}")
            print(f"  Recall: {results_static['recall']:.4f}")
            print(f"  Accuracy: {results_static['accuracy']:.4f}")
            print(f"  F1: {results_static['f1']:.4f}")
            
            # Optimal static threshold (minimizes total bank loss)
            if name in optimal_thresholds:
                opt_thresh = optimal_thresholds[name]
                results_opt = value_evaluator.evaluate(
                    y_true=y_test,
                    y_pred_prob=y_prob,
                    X_features=X_test,
                    threshold_method="static",
                    static_threshold=opt_thresh,
                )
                print(f"\nOptimal Static Threshold @ {opt_thresh:.4f} (Minimizes Total Loss):")
                print(f"  Total Bank Loss: ${results_opt['Total_Bank_Loss_($)']:,.2f}")
                print(f"  Fraud Loss: ${results_opt['Fraud_Loss_($)']:,.2f}")
                print(f"  False Alarm Cost: ${results_opt['False_Alarm_Cost_($)']:,.2f}")
                print(f"  Fraud Caught: ${results_opt['Fraud_Caught_($)']:,.2f}")
                print(f"  Recall: {results_opt['recall']:.4f}")
                print(f"  Accuracy: {results_opt['accuracy']:.4f}")
                print(f"  F1: {results_opt['f1']:.4f}")
            
            # Per-sample dynamic threshold (original value-aware method - for comparison)
            results_dynamic = value_evaluator.evaluate(
                y_true=y_test,
                y_pred_prob=y_prob,
                X_features=X_test,
                threshold_method="dynamic",
            )
            print(f"\nPer-Sample Dynamic Threshold (Original Value-Aware Method):")
            print(f"  Total Bank Loss: ${results_dynamic['Total_Bank_Loss_($)']:,.2f}")
            print(f"  Fraud Loss: ${results_dynamic['Fraud_Loss_($)']:,.2f}")
            print(f"  False Alarm Cost: ${results_dynamic['False_Alarm_Cost_($)']:,.2f}")
            print(f"  Fraud Caught: ${results_dynamic['Fraud_Caught_($)']:,.2f}")
            print(f"  Recall: {results_dynamic['recall']:.4f}")
            print(f"  Accuracy: {results_dynamic['accuracy']:.4f}")
            print(f"  F1: {results_dynamic['f1']:.4f}")
            
            # Threshold sweep visualization
            print(f"\n--- Threshold Sweep Analysis for {name} ---")
            sweep_path = RESULTS_DIR / f"{name}_threshold_sweep.png"
            sweep_df, optimal_thresh = value_evaluator.plot_threshold_sweep(
                y_true=y_test,
                y_pred_prob=y_prob,
                X_features=X_test,
                n_thresholds=200,
                save_path=sweep_path,
            )
    else:
        print("\nWarning: 'proposed_credit_limit' not found in test set. Skipping Value-Aware evaluation.")
