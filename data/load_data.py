from sklearn.datasets import load_boston
import pandas as pd

def load_boston_data():
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    return df, boston
