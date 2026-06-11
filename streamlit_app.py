import streamlit as st

activity = st.Page("dashboards/activity.py", title="Activity", icon=":material/bar_chart:")
nlp = st.Page("dashboards/nlp.py", title="NLP Analysis", icon=":material/analytics:")

pg = st.navigation([nlp, activity])
pg.run()
