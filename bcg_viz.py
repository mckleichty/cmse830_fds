# app.py

import streamlit as st

main_page = st.Page("main_page.py", title="Abell 2597", icon=":material/home:")
#spec_page = st.Page("spectrum.py", title="User's Turn", icon=":material/earthquake:")
#set up navigation
#pg = st.navigation([main_page, spec_page])
pg = st.navigation([main_page])

pg.run() #run the selected page
