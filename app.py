import streamlit as st

pg = st.navigation([st.Page("page/Single Mode.py"), st.Page("page/Bulk Mode.py")])
st.set_page_config(page_title="Home", page_icon=":material/settings:")
pg.run()