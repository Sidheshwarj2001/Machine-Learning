import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#
# def DrawGraph(x,y):
#     plt.plot(x,y)
#     plt.show()

def HeadBrainPredictor():
    # load data
    df = pd.read_csv("HeadBrain.csv")

    print("The size of the data",df.shape)

    x= df["Head Size(cm^3)"].values
    y = df["Brain Weight(grams)"].values

    # DrawGraph(x,y)

    x= x.reshape((-1,1))

    n = len(x)

    reg = LinearRegression()

    reg = reg.fit(x,y)

    y_pred = reg.predict(x)

    r2 = reg.score(x,y)

    print(r2*100)

def main():
    print("______________Welcome to our HeadBrain Case study___________")
    print("Supervised Machine Learning")
    print("Linear regression on Head and Brain size data set")
    HeadBrainPredictor()

if __name__ == "__main__":
    main()