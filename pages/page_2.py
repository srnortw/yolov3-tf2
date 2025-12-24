import streamlit as st
import rans
import numpy as np

st.markdown("# Page 2 ❄️")




left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:

left_column.button('Press me!')


with left_column:
    rans.y
    st.write('hey my')
# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    # Display the plot in Streamlit
    st.pyplot(rans.fig)

st.sidebar.markdown("# Page 2 ❄️")