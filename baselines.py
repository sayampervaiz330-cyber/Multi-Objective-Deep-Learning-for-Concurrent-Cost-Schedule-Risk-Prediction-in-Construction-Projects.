"""
Single-task baselines: one independently-trained model per target.
This is the "silo" approach your thesis is arguing against — the multi-task
net should beat these, or at minimum match them with fewer parameters.
"""
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier


def train_single_task_baselines(X_train, y_train, seed=0):
    models = {}

    models["cost_rf"] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    models["cost_rf"].fit(X_train, y_train["cost_growth_ratio"])

    models["sched_rf"] = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed,
                                                 class_weight="balanced", n_jobs=-1)
    models["sched_rf"].fit(X_train, y_train["schedule_overrun_label"])

    models["risk_rf"] = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed,
                                                class_weight="balanced", n_jobs=-1)
    models["risk_rf"].fit(X_train, y_train["risk_label"])

    # single-task MLP baselines (same rough capacity as one head of the MTL net)
    models["cost_mlp"] = MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=150, random_state=seed)
    models["cost_mlp"].fit(X_train, y_train["cost_growth_ratio"])

    models["sched_mlp"] = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=150, random_state=seed)
    models["sched_mlp"].fit(X_train, y_train["schedule_overrun_label"])

    models["risk_mlp"] = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=150, random_state=seed)
    models["risk_mlp"].fit(X_train, y_train["risk_label"])

    return models
