import streamlit as st
import rans
import numpy as np
st.markdown("# Page 3 x")
rans.y
st.sidebar.markdown("# Page 3 x")

@st.cache_data
def c():
    v=np.random.rand(2,2)
    return v

v=c()
v

if "f" not in st.session_state:
    st.session_state.f=np.random.rand(2,2)

st.session_state.f

q=np.random.rand(2,2)
q