import streamlit as st
from process_excel import process_excel
import base64
import io
import pandas as pd

def get_download_link(df):
    # Generate a link to download the processed data as an Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="processed_data.xlsx">Download Excel File</a>'
    return href

def main():
    st.title("AI Assisted Audit")

    # File Upload
    uploaded_file = st.file_uploader("Choose an Excel file to run checks on", type=["xlsx"])

    if uploaded_file is not None:
        # Display uploaded file details
        st.write("Uploaded file details:")
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        # Process the data
        processed_data = process_excel(uploaded_file)

        # Display the processed data
        st.write("Processed Data:")
        st.write(processed_data)

        # Download processed data as Excel file
        st.markdown(get_download_link(processed_data), unsafe_allow_html=True)

main()

