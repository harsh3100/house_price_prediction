from data.load_data import load_boston_data
from utils.helpers import split_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.visualize import plot_heatmap

def main():
    df, boston = load_boston_data()
    plot_heatmap(df)

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)

    print(f"\nMean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

if __name__ == "__main__":
    main()
