import os
import pandas as pd
import textract
import streamlit as st
from pathlib import Path

# Function to extract text from a document using textract
def extract_text(file_path):
    try:
        # Extract text using textract
        text = textract.process(file_path).decode("utf-8")  # Decode to convert bytes to string
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

# Function to gather file data and extract text
def collect_files_and_extract_text(folder_path):
    data = []  # List to hold file data for CSV

    # Walk through the folder to find files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            folder_name = Path(file_path).parent.name  # Last folder name in path (for 'Category' column)

            # Extract text from the document using textract
            document_text = extract_text(file_path)

            # Add row data to the list (file name, path, category, document text)
            data.append({
                "File Path": file_path,
                "Cadidate Resume": file,
                "Category": folder_name,  # Assuming the category is the last folder in the path
                "Document Data Extracted": document_text  # Adding extracted text as a new column
            })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Remove rows where 'Category' is 'Resume_Docx' (if needed)
    df = df[df['Category'] != 'Resume_Docx']

    # Save the DataFrame to CSV (or return for further use)
    df.to_csv('document_data.csv', index=False)
    return df

# Streamlit UI
st.title("Extract and Display Document Text")

# Folder path input
folder_path = st.text_input("Enter the folder path where documents are located:")

# Button to process and generate CSV
if st.button("Process Documents and Generate CSV"):
    if folder_path:
        # Check if folder exists
        if os.path.exists(folder_path):
            # Collect files and extract text
            df = collect_files_and_extract_text(folder_path)
            
            # Show the dataframe in Streamlit
            st.subheader("Extracted Data")
            st.dataframe(df)

            st.success("CSV file generated successfully!")
        else:
            st.error("The provided folder path does not exist.")