import streamlit as st

page_config = {
    "page_title": "Search Topics on Youtube.",
    "layout": "centered",
}
st.set_page_config(**page_config)
st.markdown("<h1 style='text-align: center;'> Ask Youtube Anything </h1>", unsafe_allow_html=True)

# Text input for search query
query = st.text_input('Enter a topic to search on YouTube:', '')

# Horizontal checkboxes for topic selection
col1, col2, col3, col4 = st.columns(4)
with col1:
    fpl = st.checkbox('FPL')
with col2:
    golf = st.checkbox('Golf')
with col3:
    xrp = st.checkbox('XRP')
with col4:
    other = st.checkbox('Other')

# Search button
if st.button('Search'):
    if query:
        if fpl:
            st.write(f'Searching for: {query} in FPL database')
            # Add your database querying logic here for FPL
        if golf:
            st.write(f'Searching for: {query} in Golf database')
            # Add your database querying logic here for Golf
        if xrp:
            st.write(f'Searching for: {query} in XRP database')
            # Add your database querying logic here for XRP
        if other:
            st.write(f'Searching for: {query} in Other database')
            # Add your database querying logic here for Other
    else:
        st.error('Please enter a search term.')