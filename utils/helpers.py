from sklearn.model_selection import train_test_split

def split_data(df):
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    return train_test_split(X, y, test_size=0.2, random_state=42)
