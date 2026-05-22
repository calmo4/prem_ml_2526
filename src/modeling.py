#Takes raw match CSVs and creates:
# rolling features
# aggregated season tables
# processed datasets

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_linear_model(
    train_df,
    test_df,
    features,
    target="Points"
):
    """
    Train and evaluate linear regression model.
    """

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "predictions": predictions,
        "mae": mae,
        "r2": r2
    }


def train_random_forest(
    train_df,
    test_df,
    features,
    target="Points",
    n_estimators=300,
    random_state=42
):
    """
    Train and evaluate Random Forest model.
    """

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "predictions": predictions,
        "mae": mae,
        "r2": r2
    }