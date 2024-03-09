# streamlit_app.py

import streamlit as st
from process_data import process_excel
import base64

def main():
    st.title("Excel Processing App")

    # File Upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Display uploaded file details
        st.write("Uploaded file details:")
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        # Process the data based on user choice
        process_choice = st.selectbox("Choose a process:", ["Process 1", "Process 2", "Process 3", "Process 4", "Process 5"])

        processes = {
            "Process 1": process_data_1,
            "Process 2": process_data_2,
            "Process 3": process_data_3,
            "Process 4": process_data_4,
            "Process 5": process_data_5,
        }

        # Apply the selected process
        processed_data = processes[process_choice](process_excel(uploaded_file))

        # Display the processed data
        st.write("Processed Data:")
        st.write(processed_data)

        # Download processed data as Excel file
        st.markdown(get_download_link(processed_data), unsafe_allow_html=True)

def get_download_link(df):
    # Generate a link to download the processed data as an Excel file
    csv = df.to_excel(index=False, encoding='utf-8', header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64 encoding
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="processed_data.xlsx">Download Excel File</a>'
    return href

if __name__ == "__main__":
    main()

