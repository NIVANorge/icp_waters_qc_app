import streamlit as st


def app():
    """Main function for the 'check' page."""
    st.markdown(
        """This application implements preliminary quality checking of data submitted to 
           [ICP Waters](https://www.icp-waters.no/). Data must be supplied using the Excel template available 
           [here](https://github.com/NIVANorge/icp_waters_qc_app/blob/main/data/icpw_input_template_chem_v0-4.xlsx) 
           (please do not modify any column headings).
    """
    )
    st.markdown(
        """The aim of the app is to make it easier to quickly check data templates for obvious errors. Before 
           sending data templates to NIVA, please first upload them via the app and review any issues identified. 
           Cleaned templates can then be sent to NIVA for upload to the database.
    """
    )
    st.markdown(
        """**Note:** This app is a prototype. Please send questions or comments to 
           [James Sample](mailto:james.sample@niva.no).
    """
    )
    with st.expander("Getting started"):
        st.markdown(
            """To begin, click the `Check data` option in the left sidebar, then click the `Browse files` button 
               to upload a completed data template. Dragging and dropping a file should also work.
        """
        )
        st.markdown(
            """The app will perform basic data validation and results will appear in the main window. Data
               `ERRORS` (red message boxes) indicate problems parsing the data and these must be fixed before other
               tests can be completed. `WARNINGS` (yellow/orange message boxes) highlight potential issues that
               should be checked, but which do not prevent the file from being processed further. 
        """
        )
        st.markdown(
            """Please scroll through the output in the main window and fix as many of the issues as possible 
               before sending the cleaned file to NIVA for further processing.
        """
        )
        st.info(
            """**Tip:** If you wish to try out the app, you can download Excel files containing "fake errors" 
               to illustrate how the tests work. Click the Excel files 
               [here](https://github.com/NIVANorge/icp_waters_qc_app/tree/main/data/test_data) and use the
               `Download` button to save each file to your local machine (alternatively, if you are familiar 
               with GitHub, you can simply clone the entire repository).
        """
        )

    return None
