import streamlit as st
from streamlit_option_menu import option_menu

import subpages.check
import subpages.home

PAGES = {
    "Home": subpages.home,
    "Check data": subpages.check,
}


def main():
    """Main function of the app."""
    st.title("ICP Waters QC app")
    st.sidebar.image(r"./images/niva-logo.png", use_column_width=True)
    with st.sidebar:
        selection = option_menu(
            None,
            options=["Home", "Check data"],
            icons=["house", "check-square"],
            default_index=0,
        )
    page = PAGES[selection]
    page.app()


if __name__ == "__main__":
    main()
