import matplotlib.pyplot as plt
import pandas as pd


def draw_charts(history):
    pd.DataFrame(history.history).plot(figsize=(10, 5))
    plt.show()
