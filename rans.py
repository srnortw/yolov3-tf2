import numpy as np
import pandas as pd
#import streamlit as st
import matplotlib.pyplot as plt
# if "x" not in st.session_state:
#     st.session_state.x = np.random.rand(7,1) #pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])
# @st.cache_data
# def x():
#     x=np.random.rand(7,1)
#     return x
# x = x()
y=np.random.rand(7,1)
x=list(range(0,7))

# Create a Matplotlib figure
fig, ax = plt.subplots()

ax.scatter(x,y)

