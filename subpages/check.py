from io import BytesIO

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import FastMarkerCluster, MarkerCluster
from st_aggrid import AgGrid
from streamlit_folium import folium_static

IDX_COLS = [
    "Code",
    "Name",
    "Date",
]
MINIMUM_COLS = [
    "pH",
    "Ca_mg/L",
    "Mg_mg/L",
    "Na_mg/L",
    "K_mg/L",
    "Alk_µeq/L",
    "SO4_mg/L",
    "NO3-N_µgN/L",
    "Cl_mg/L",
]
DESIRABLE_COLS = [
    "Cond25_mS/m at 25C",
    "NH4-N_µgN/L",
    "TOTP_µgP/L",
    "DOC_mgC/L",
    "TOC_mgC/L",
    "PERM_mgO/L",
    "LAl_µg/L",
]
OPTIONAL_COLS = [
    "RAl_µg/L",
    "ILAl_µg/L",
    "TAl_µg/L",
    "TOTN_µgN/L",
    "ORTP_µgP/L",
    "OKS_mgO/L",
    "SiO2_mgSiO2/L",
    "F_µg/L",
    "Fe_Total_µg/L",
    "Mn_Total_µg/L",
    "Cd_Total_µg/L",
    "Zn_Total_µg/L",
    "Cu_Total_µg/L",
    "Ni_Total_µg/L",
    "Pb_Total_µg/L",
    "Cr_Total_µg/L",
    "As_Total_µg/L",
    "Hg_Total_ng/L",
    "Fe_Filt_µg/L",
    "Mn_Filt_µg/L",
    "Cd_Filt_µg/L",
    "Zn_Filt_µg/L",
    "Cu_Filt_µg/L",
    "Ni_Filt_µg/L",
    "Pb_Filt_µg/L",
    "Cr_Filt_µg/L",
    "As_Filt_µg/L",
    "Hg_Filt_ng/L",
    "COLOUR_mgPt/L",
    "TURB_FTU",
    "TEMP_C",
    "RUNOFF_m3/s",
]


def app():
    """Main function for the 'check' page."""
    stn_df = pd.read_excel(r"./data/icpw_all_stations.xlsx", sheet_name="stations")
    # countries = [None] + sorted(stn_df["country"].unique())
    # country = st.sidebar.selectbox("Select country:", countries)
    data_file = st.sidebar.file_uploader("Upload template")

    # if country:
    #     stn_df = stn_df.query("country == @country")

    if data_file:
        with st.spinner("Reading data..."):
            df = read_data_template(data_file)
            # st.header("Raw data")
            # st.markdown("The raw data from Excel are shown in the table below.")
            st.markdown(f"**File name:** `{data_file.name}`")
            # AgGrid(df, height=400)
            st.markdown(
                "Please **scroll down** to see results from the quality checks."
            )

        with st.spinner("Checking data..."):
            check_columns(df)
            check_parameters(df)
            check_numeric(df)
            check_greater_than_zero(df)
            check_stations(df, stn_df)
            check_duplicates(df)
            st.header("Checking water chemistry")
            check_no3_totn(df)
            check_po4_totp(df)
            check_ral_ilal_lal(df)
            check_lal_ph(df)
            st.subheader("Ion balance")
            df = convert_to_numeric(df)
            df = convert_to_microequivalents(df)
            df = calculate_oh_and_h(df)
            df = calculate_a_minus(df)
            df = calculate_cations_and_anions(df)
            df = check_ion_balance(df, thresh_pct=10)
            st.subheader("Conductivity")
            df = calculate_ion_strength(df)
            df = calculate_gamma(df)
            df = calculate_theoretical_conductivity(df)
            df = check_conductivity(df, thresh_pct=10)
            check_outliers(df)

    return None


@st.cache
def prepare_df_for_download(df):
    """Convert dataframe to bytes for download.

    Args
        df: Fataframe to be converted

    Returns
        Bytes.
    """
    output = BytesIO()
    with pd.ExcelWriter(output) as writer:
        readme_sheet = writer.book.create_sheet(title="readme")
        msg = "Data exported from the ICP Waters quality checking app."
        readme_sheet.cell(column=1, row=1, value=msg)
        df.to_excel(writer, index=False, sheet_name="data")

    data = output.getvalue()

    return data


def read_data_template(file_path):
    """Read the ICPW template. An example of the template is here:

           ../data/icpw_input_template_chem_v0-3.xlsx

    Args
        file_path:  Raw str. Path to Excel template
        sheet_name: Str. Name of sheet to read

    Returns
        Dataframe.
    """
    df = pd.read_excel(file_path, sheet_name="Data", skiprows=1, header=[0, 1])
    df = merge_multi_header(df)

    return df


def merge_multi_header(df):
    """Merge the parameter and unit rows of the template into a single header.

    Args
        df: Raw dataframe read from the template

    Returns
        Dataframe with single, tidied header.
    """
    df.columns = [f"{i[0]}_{i[1]}" for i in df.columns]
    df.columns = [i.replace("_-", "") for i in df.columns]

    return df


def check_columns(df):
    """Check column names in the parsed template are valid.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.header("Checking columns")
    n_errors = 0
    for col in df.columns:
        if col not in (IDX_COLS + MINIMUM_COLS + DESIRABLE_COLS + OPTIONAL_COLS):
            n_errors += 1
            st.markdown(f" * Column **{col}** is not recognised")
    if n_errors > 0:
        st.error(
            "ERROR: Some column names are not recognised. Please use the template available "
            "[here](https://github.com/NIVANorge/icp_waters_qc_app/blob/main/data/icpw_input_template_chem_v0-4.xlsx)."
        )
        st.stop()
    else:
        st.success("OK!")


def check_parameters(df):
    """Check tha mandatory parameters are present.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.header("Checking parameters")
    st.subheader("Required parameters")
    n_errors = 0
    for col in IDX_COLS + MINIMUM_COLS:
        if (col not in df.columns) or (df[col].isna().all()):
            n_errors += 1
            st.markdown(
                f" * Column **{col}** is required (it is part of the `minimum` parameter group)"
            )
    if n_errors > 0:
        st.error(
            "ERROR: Some required parameters are not included. Please see the `Info` worksheet of the "
            "template for details."
        )
        # st.stop()
    else:
        st.success("OK!")

    st.subheader("Other parameters")
    n_errors = 0
    for col in DESIRABLE_COLS:
        if (col not in df.columns) or (df[col].isna().all()):
            if (col == "PERM_mgO/L") and (
                ~df["TOC_mgC/L"].isna().all() or ~df["DOC_mgC/L"].isna().all()
            ):
                # PERM is not required if DOC or TOC are provided
                pass
            else:
                n_errors += 1
                st.markdown(f" * Data for **{col}** not provided")
    if n_errors > 0:
        st.warning(
            "WARNING: Please consider adding data for additional parameters, if possible. See the `Info` "
            "worksheet of the template for details."
        )
    else:
        st.success("OK!")

    return None


def check_numeric(df):
    """Check that relevant columns in 'df' contain numeric data. LOD values beginning with '<' are
       permitted.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.header("Checking for non-numeric data")
    non_num_cols = [
        "Code",
        "Name",
        "Date",
    ]
    num_cols = [col for col in df.columns if col not in non_num_cols]
    n_errors = 0
    for col in num_cols:
        # num_series = pd.to_numeric(
        #     df[col].fillna(-9999).astype(str).str.strip("<").str.replace(",", "."),
        #     errors="coerce",
        # )
        num_series = pd.to_numeric(
            df[col].fillna(-9999).astype(str).str.strip("<"),
            errors="coerce",
        )
        non_num_vals = df[pd.isna(num_series)][col].values
        if len(non_num_vals) > 0:
            n_errors += 1
            st.markdown(
                f" * Column **{col}** contains non-numeric values: `{non_num_vals}`"
            )

    if n_errors > 0:
        st.error(
            "ERROR: The template contains non-numeric data (see above). Please fix these issues and try again."
        )
        st.stop()
    else:
        st.success("OK!")

    return None


def check_greater_than_zero(df):
    """Check that relevant columns in 'df' contain values greater than zero.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.header("Checking for negative and zero values")
    non_num_cols = [
        "Code",
        "Name",
        "Date",
    ]
    allow_neg_cols = ["Alk_µeq/L", "TEMP_C"]
    allow_zero_cols = ["LAl_µg/L"]
    gt_zero_cols = [
        col
        for col in df.columns
        if col not in (non_num_cols + allow_neg_cols + allow_zero_cols)
    ]
    n_errors = 0
    for col in gt_zero_cols:
        # num_series = pd.to_numeric(
        #     df[col].fillna(-9999).astype(str).str.strip("<").str.replace(",", ".")
        # )
        num_series = pd.to_numeric(
            df[col].fillna(-9999).astype(str).str.strip("<"),
            errors="coerce",
        )
        num_series[num_series == -9999] = np.nan
        if num_series.min() <= 0:
            n_errors += 1
            st.markdown(
                f" * Column **{col}** contains values less than or equal to zero."
            )

    for col in allow_zero_cols:
        # num_series = pd.to_numeric(
        #     df[col].fillna(-9999).astype(str).str.strip("<").str.replace(",", ".")
        # )
        num_series = pd.to_numeric(
            df[col].fillna(-9999).astype(str).str.strip("<"),
            errors="coerce",
        )
        num_series[num_series == -9999] = np.nan
        if num_series.min() < 0:
            n_errors += 1
            st.markdown(f" * Column **{col}** contains values less than zero.")

    if n_errors == 0:
        st.success("OK!")
    else:
        st.error(
            "ERROR: Some reported values are less than or equal to zero. Values should all be "
            "positive, except for column `LAl_µg/L` (which permit values of zero), "
            "and columns `Alk_µeq/L` and `TEMP_C` (which can be negative)."
        )
        # st.stop()

    return None


def check_stations(df, stn_df):
    """Check stations in 'df' against reference data in 'stn_df'.

    Args
        df:     Dataframe of sumbitted water chemistry data
        stn_df: Dataframe of reference station details

    Returns
        None. Problems identified are printed to output.
    """
    st.header("Checking stations")
    n_errors = 0
    if not set(df["Code"]).issubset(set(stn_df["station_code"])):
        n_errors += 1
        st.markdown(
            "The following station codes are not in the definitive station list."
        )
        st.code(set(df["Code"]) - set(stn_df["station_code"]))

    # Check station ID have consistent names
    msg = ""
    site_ids = df["Code"].unique()
    for site_id in site_ids:
        true_name = stn_df.query("station_code == @site_id")["station_name"].values
        names = df.query("Code == @site_id")["Name"].unique()

        if len(names) > 1:
            msg += f"\n * **{site_id}** ({true_name[0]}) has names: `{names}`"

    if msg != "":
        n_errors += 1
        st.markdown(
            "The following station codes have inconsistent names within this template:"
        )
        st.markdown(msg)

    # Check station names have consistent IDs
    msg = ""
    site_names = df["Name"].unique()
    for site_name in site_names:
        true_id = stn_df.query("station_name == @site_name")["station_code"].values
        ids = df.query("Name == @site_name")["Code"].unique()

        if len(ids) > 1:
            msg += f"\n * **{site_name}** has codes: `{ids}`"
    if msg != "":
        n_errors += 1
        st.markdown(
            "The following station names have multiple codes within this template:"
        )
        st.markdown(msg)

    st.markdown("The template contains data for the following stations:")
    template_stns = df["Code"].unique()
    country_stn_df = stn_df.query("station_code in @template_stns")
    stn_map = quickmap(
        country_stn_df,
        cluster=True,
        popup="station_code",
        aerial_imagery=True,
        kartverket=False,
    )
    folium_static(stn_map, width=700)

    if n_errors == 0:
        st.success("OK!")
    else:
        st.error(
            "ERROR: The template contains unknown or duplicated station names and/or IDs. "
            "The full list of ICP Waters stations is shown in the table below. Please filter "
            "using the `country` column to identify the current codes and names used for sites "
            "in your country. Please [contact us](mailto:kari.austnes@niva.no) if details are "
            "missing or incorrect, or if you would like to add a new station to the database."
        )
        AgGrid(stn_df, height=400)
        # st.stop()

    return None


def check_duplicates(df):
    """Check for multiple samples at the same location and time.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.header("Checking duplicates")
    key_cols = [
        "Code",
        "Date",
    ]
    dup_df = df[
        df.duplicated(
            key_cols,
            keep=False,
        )
    ].sort_values(key_cols)

    n_dups = len(dup_df)
    if n_dups > 0:
        st.markdown(
            f"There are **{n_dups}** duplicated samples (samples with the same location and sample date).\n"
        )
        AgGrid(dup_df, height=200)
        st.error(
            "ERROR: The template contains multiple samples for the same location and date.\n\nPlease note "
            "that only **surface samples** (depth ≤ 0.5 m) are relevant to ICP Waters - see the `Info` "
            "worksheet of the template for details."
        )
        # st.stop()
    else:
        st.success("OK!")


def check_no3_totn(df):
    """Highlights all rows where nitrate > TOTN. Emphasises rows where
    NO3 > TOTN and TOC > 5 based on advice from Øyvind G (see e-mail received
    04.05.2022 at 23.23 for details).

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.subheader("NO3 and TOTN")
    mask_df = df[
        [
            "Code",
            "Date",
            "NO3-N_µgN/L",
            "TOTN_µgN/L",
            "TOC_mgC/L",
        ]
    ].copy()

    for col in ["NO3-N_µgN/L", "TOTN_µgN/L", "TOC_mgC/L"]:
        mask_df[col].fillna(0, inplace=True)
        mask_df[col] = pd.to_numeric(mask_df[col].astype(str).str.strip("<"))
        mask_df[mask_df == 0] = np.nan
    mask = mask_df["NO3-N_µgN/L"] > mask_df["TOTN_µgN/L"]
    mask_df = mask_df[mask]
    mask_df_toc = mask_df[mask_df["TOC_mgC/L"] > 5]

    if len(mask_df) > 0:
        st.markdown(
            f"The following {len(mask_df)} samples have nitrate greater than total nitrogen:"
        )
        AgGrid(mask_df, height=200)

    if len(mask_df_toc) > 0:
        st.markdown(
            f"Of these, the following {len(mask_df_toc)} samples have nitrate greater than total "
            "nitrogen **and** TOC > 5 mgC/L. This is unlikely to be within instrument error."
        )
        AgGrid(mask_df_toc, height=200)

    if len(mask_df) > 0:
        st.warning(f"WARNING: Possible issues with NO3 and TOTN.")
    else:
        st.success("OK!")

    return None


def check_po4_totp(df):
    """Highlights all rows where PO4 > TOTP.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.subheader("ORTP and TOTP")
    mask_df = df[
        [
            "Code",
            "Date",
            "ORTP_µgP/L",
            "TOTP_µgP/L",
        ]
    ].copy()

    for col in ["ORTP_µgP/L", "TOTP_µgP/L"]:
        mask_df[col].fillna(0, inplace=True)
        mask_df[col] = pd.to_numeric(mask_df[col].astype(str).str.strip("<"))
        mask_df[mask_df == 0] = np.nan
    mask = mask_df["ORTP_µgP/L"] > mask_df["TOTP_µgP/L"]
    mask_df = mask_df[mask]

    if len(mask_df) > 0:
        st.markdown(
            f"The following {len(mask_df)} samples have orthophosphate greater than total phosphrous:"
        )
        AgGrid(mask_df, height=200)

    if len(mask_df) > 0:
        st.warning(f"WARNING: Possible issues with ORTP and TOTP.")
    else:
        st.success("OK!")

    return None


def check_ral_ilal_lal(df):
    """Check RAl - ILAl = LAl.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.subheader("Al fractions")
    mask_df = df[
        [
            "Code",
            "Date",
            "RAl_µg/L",
            "ILAl_µg/L",
            "LAl_µg/L",
        ]
    ].copy()
    mask_df.dropna(subset="LAl_µg/L", inplace=True)

    for col in ["RAl_µg/L", "ILAl_µg/L", "LAl_µg/L"]:
        mask_df[col].fillna(0, inplace=True)
        mask_df[col] = pd.to_numeric(mask_df[col].astype(str).str.strip("<"))
    mask_df["LAl_Expected_µg/L"] = (mask_df["RAl_µg/L"] - mask_df["ILAl_µg/L"]).round(1)
    mask_df[mask_df["LAl_Expected_µg/L"] < 0] = 0
    mask_df["LAl_µg/L"] = mask_df["LAl_µg/L"].round(1)
    mask = mask_df["LAl_Expected_µg/L"] != mask_df["LAl_µg/L"]
    mask_df = mask_df[mask]

    if len(mask_df) > 0:
        st.markdown(
            f"The following {len(mask_df)} samples have LAl not equal to (RAl - ILAl):"
        )
        AgGrid(mask_df, height=200)
        st.warning(f"WARNING: Possible issues with the calculation of LAl.")
    else:
        st.success("OK!")

    return None


def check_lal_ph(df):
    """Highlight rows where pH > 6.4 and LAl > 20 ug/l. See e-mail from
    Øyvind G received 04.05.2022 at 23.23 for background.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.subheader("LAl and pH")
    mask_df = df[
        [
            "Code",
            "Date",
            "pH",
            "LAl_µg/L",
        ]
    ].copy()
    mask_df = mask_df[(mask_df["pH"] > 6.4) & (mask_df["LAl_µg/L"] > 20)]
    if len(mask_df) > 0:
        st.markdown(
            f"The following {len(mask_df)} samples have LAl > 20 µg/l and pH > 6.4, which is considered unlikely:"
        )
        AgGrid(mask_df, height=200)
        st.warning(f"WARNING: Possible issues with LAl and/or pH.")
    else:
        st.success("OK!")

    return None


def convert_to_numeric(df):
    """Convert a chemistry columns in a dataframe to numeric values. LOD values
    beginning with '<' are replaced with the LOD itself.

    Args
        df:  Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. Chemistry cols in 'df' are converted to numeric.
    """
    non_num_cols = [
        "Code",
        "Name",
        "Date",
    ]
    num_cols = [col for col in df.columns if col not in non_num_cols]
    for col in num_cols:
        df[col] = pd.to_numeric(
            df[col].fillna(-9999).astype(str).str.strip("<"),
            errors="coerce",
        )
        df[col].replace(-9999, np.nan, inplace=True)

        # Remove columns that are all NaN
        if df[col].isna().all():
            del df[col]

    return df


def convert_to_microequivalents(df):
    """Basic conversion from mass/l to microequivalents/l.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. New columns are added with valuyes in µeq/L.
    """
    chem_prop_df = pd.read_csv(r"./data/chemical_properties.csv")

    for idx, row in chem_prop_df.iterrows():
        par_unit = row["par"]
        valency = row["valency"]
        molar_mass = row["molar_mass"]

        if par_unit in df.columns:
            # Separate par and unit
            parts = par_unit.split("_")
            par = "_".join(parts[:-1])
            unit = parts[-1]

            # Determine unit factor
            if unit[0] == "m":
                factor = 1000
            elif unit[0] == "µ":
                factor = 1
            else:
                raise ValueError("Unit factor could not be identified.")

            df[f"{par}_µeq/L"] = df[par_unit] * valency * factor / molar_mass

    return df


def calculate_oh_and_h(df):
    """Calculate H+ and OH-.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. Two new columns are added.
    """
    if "pH" in df.columns:
        df["OH_µeq/L"] = 1e6 * 10 ** -(14 - df["pH"])
        df["H_µeq/L"] = 1e6 * 10 ** -(df["pH"])
    else:
        st.warning(f"WARNING: Parameter 'pH' not provided. Stopping checks.")

    return df


def calculate_a_minus(df):
    """Calculate A-.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. Column 'A-_µeq/L' is added.
    """
    req_pars = ["TOC_mgC/L", "pH"]
    df_pars = list(df.columns)
    if all(par in df_pars for par in req_pars):
        df["A-_µeq/L"] = df["TOC_mgC/L"] * (1.77 * df["pH"] - 3)
    else:
        missing_pars = [col for col in req_pars if col not in df.columns]
        st.warning(
            "WARNING: Cannot calculate **A-**. Stopping checks.  "
            f"\n\nMissing parameters:  \n\n    {missing_pars}"
        )

    return df


def calculate_cations_and_anions(df):
    """Calculates total cations and anions.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. Columns 'Cations_µeq/L' and 'Anions_µeq/L' are added.
    """
    req_pars = [
        "Ca_µeq/L",
        "Mg_µeq/L",
        "Na_µeq/L",
        "K_µeq/L",
        "NH4-N_µeq/L",
        "H_µeq/L",
        "LAl_µeq/L",
        "Cl_µeq/L",
        "SO4_µeq/L",
        "NO3-N_µeq/L",
        "Alk_µeq/L",
        "A-_µeq/L",
    ]
    df_pars = list(df.columns)
    if all(par in df_pars for par in req_pars):
        df["Cations_µeq/L"] = (
            df["Ca_µeq/L"]
            + df["Mg_µeq/L"]
            + df["Na_µeq/L"]
            + df["K_µeq/L"]
            + df["NH4-N_µeq/L"]
            + df["H_µeq/L"]
            + df["LAl_µeq/L"]
        )
        df["Anions_µeq/L"] = (
            df["Cl_µeq/L"]
            + df["SO4_µeq/L"]
            + df["NO3-N_µeq/L"]
            + df["Alk_µeq/L"]
            + df["A-_µeq/L"]
        )
    else:
        missing_pars = [col for col in req_pars if col not in df.columns]
        st.warning(
            "WARNING: Cannot calculate **total cations and anions**. Stopping checks.  "
            f"\n\nMissing parameters:  \n\n    {missing_pars}"
        )

    return df


def check_ion_balance(df, thresh_pct=10):
    """Check the ion balance and highlight rows where the difference is greater than
    'thresh_pct'.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. Column 'Zdiff_pct' is added.
    """
    assert 0 < thresh_pct < 100, "'thresh_pct' must be between 0 and 100."
    req_pars = [
        "Cations_µeq/L",
        "Anions_µeq/L",
    ]
    df_pars = list(df.columns)
    if all(par in df_pars for par in req_pars):
        df["Zdiff_pct"] = (
            100 * (df["Cations_µeq/L"] - df["Anions_µeq/L"]) / df["Cations_µeq/L"]
        )
        mask_df = df.query("(Zdiff_pct > @thresh_pct) or (Zdiff_pct < -@thresh_pct)")
        if len(mask_df) > 0:
            # Get relevant cols to display
            cols = (
                ["Code", "Name", "Date"]
                + [col for col in df.columns if col.split("_")[-1] == "µeq/L"]
                + ["Zdiff_pct"]
            )
            st.markdown(
                f"The following {len(mask_df)} samples have a large difference between cations and anions:"
            )
            AgGrid(mask_df[cols].round(2), height=200)
            with st.spinner("Preparing data for download..."):
                st.markdown(
                    "The ion balance dataset can be **downloaded to Excel** for further checking."
                )
                excel_bytes = prepare_df_for_download(mask_df)
                st.download_button(
                    label="Download data",
                    data=excel_bytes,
                    file_name="ion_balance_data.xlsx",
                )
            st.warning(f"WARNING: Possible issues with the ion balance.")
        else:
            st.success("OK!")
    else:
        st.warning(f"WARNING: Ion balance could not be calculated.")

    return df


def calculate_ion_strength(df):
    """Calculate ion strength.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. Column 'ion_strength' is added.
    """
    req_pars = [
        "Na_µeq/L",
        "K_µeq/L",
        "NH4-N_µeq/L",
        "Cl_µeq/L",
        "NO3-N_µeq/L",
        "Alk_µeq/L",
        "OH_µeq/L",
        "H_µeq/L",
        "Ca_µeq/L",
        "Mg_µeq/L",
        "SO4_µeq/L",
        "LAl_µeq/L",
    ]
    df_pars = list(df.columns)
    if all(par in df_pars for par in req_pars):
        df["ion_strength"] = (
            df["Na_µeq/L"]
            + df["K_µeq/L"]
            + df["NH4-N_µeq/L"]
            + df["Cl_µeq/L"]
            + df["NO3-N_µeq/L"]
            + df["Alk_µeq/L"]
            + df["OH_µeq/L"]
            + df["H_µeq/L"]
            + (
                2
                * (df["Ca_µeq/L"] + df["Mg_µeq/L"] + df["SO4_µeq/L"] + df["LAl_µeq/L"])
            )
        ) / 2e6
    else:
        missing_pars = [col for col in req_pars if col not in df.columns]
        st.warning(
            "WARNING: Cannot calculate **ion strength**. Stopping checks.  "
            f"\n\nMissing parameters:  \n\n    {missing_pars}"
        )

    return df


def calculate_gamma(df):
    """Calculate gamma with z=1 and z=2.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. Columns 'gamma_z=1' and 'gamma_z=2' are added.
    """
    if "ion_strength" in df.columns:
        df["gamma_z=1"] = 10 ** -(
            (
                0.5 * ((df["ion_strength"] ** 0.5) / (1 + df["ion_strength"] ** 0.5))
                - (0.3 * df["ion_strength"])
            )
        )
        df["gamma_z=2"] = 10 ** -(
            (
                0.5
                * 4
                * ((df["ion_strength"] ** 0.5) / (1 + df["ion_strength"] ** 0.5))
                - (0.3 * df["ion_strength"])
            )
        )
    else:
        st.warning(f"WARNING: Parameter 'ion_strength' not provided. Stopping checks.")

    return df


def calculate_theoretical_conductivity(df):
    """Calculate theoretical conductivity..

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. Column 'CondTheory_mS/m at 25C' is added.
    """
    req_pars = [
        "gamma_z=1",
        "gamma_z=2",
        "Ca_µeq/L",
        "Mg_µeq/L",
        "Na_µeq/L",
        "K_µeq/L",
        "NH4-N_µeq/L",
        "Cl_µeq/L",
        "SO4_µeq/L",
        "NO3-N_µeq/L",
        "Alk_µeq/L",
        "H_µeq/L",
        "LAl_µeq/L",
        "OH_µeq/L",
        "A-_µeq/L",
        "Cond25_mS/m at 25C",
    ]
    df_pars = list(df.columns)
    if all(par in df_pars for par in req_pars):
        df["CondTheory_mS/m at 25C"] = (
            0.0595 * df["Ca_µeq/L"] * df["gamma_z=2"]
            + 0.053 * df["Mg_µeq/L"] * df["gamma_z=2"]
            + 0.0501 * df["Na_µeq/L"] * df["gamma_z=1"]
            + 0.0735 * df["K_µeq/L"] * df["gamma_z=1"]
            + 0.0735 * df["NH4-N_µeq/L"] * df["gamma_z=1"]
            + 0.0763 * df["Cl_µeq/L"] * df["gamma_z=1"]
            + 0.08 * df["SO4_µeq/L"] * df["gamma_z=2"]
            + 0.0714 * df["NO3-N_µeq/L"] * df["gamma_z=1"]
            + 0.0445 * df["Alk_µeq/L"] * df["gamma_z=1"]
            + 0.35 * df["H_µeq/L"] * df["gamma_z=1"]
            + 0.061 * df["LAl_µeq/L"] * df["gamma_z=2"]
            + 0.1983 * df["OH_µeq/L"] * df["gamma_z=1"]
            + 0.05 * df["A-_µeq/L"] * df["gamma_z=1"]
        ) / 10
    else:
        missing_pars = [col for col in req_pars if col not in df.columns]
        st.warning(
            "WARNING: Cannot calculate **theoretical conductivity**. Stopping checks.  "
            f"\n\nMissing parameters:  \n\n    {missing_pars}"
        )

    return df


def check_conductivity(df, thresh_pct=10):
    """Compare measured and theoretical conductivity and highlight rows where the difference
    is greater than 'thresh_pct'.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        DataFrame. Column 'Cond_Diff_pct' is added.
    """
    assert 0 < thresh_pct < 100, "'thresh_pct' must be between 0 and 100."
    req_pars = [
        "Cond25_mS/m at 25C",
        "CondTheory_mS/m at 25C",
    ]
    df_pars = list(df.columns)
    if all(par in df_pars for par in req_pars):
        df["CondDiff_pct"] = (
            100
            * (df["CondTheory_mS/m at 25C"] - df["Cond25_mS/m at 25C"])
            / df["CondTheory_mS/m at 25C"]
        )
        mask_df = df.query(
            "(CondDiff_pct > @thresh_pct) or (CondDiff_pct < -@thresh_pct)"
        )
        if len(mask_df) > 0:
            # Get relevant cols to display
            cols = (
                ["Code", "Name", "Date"]
                + [col for col in df.columns if col.split("_")[-1] == "µeq/L"]
                + [
                    "Zdiff_pct",
                    "ion_strength",
                    "gamma_z=1",
                    "gamma_z=2",
                    "CondTheory_mS/m at 25C",
                    "Cond25_mS/m at 25C",
                    "CondDiff_pct",
                ]
            )
            st.markdown(
                f"The following {len(mask_df)} samples have a large difference between measured and theoretical conductivity:"
            )
            AgGrid(mask_df[cols].round(2), height=200)
            with st.spinner("Preparing data for download..."):
                st.markdown(
                    "The theoretical conductivity dataset can be **downloaded to Excel** for further checking."
                )
                excel_bytes = prepare_df_for_download(mask_df)
                st.download_button(
                    label="Download data",
                    data=excel_bytes,
                    file_name="conductivity_data.xlsx",
                )
            st.warning(f"WARNING: Possible issues with theoretical conductivity.")
        else:
            st.success("OK!")
    else:
        st.warning(f"WARNING: Theoretical conductivity could not be calculated.")

    return df


def check_outliers(df, iqr_factor=3):
    """Identify outlier as points more than

            IQR * iqr_factor

    above or below the limits of the IQR.

    Args
        df: Dataframe of sumbitted water chemistry data

    Returns
        None. Problems identified are printed to output.
    """
    st.subheader("Outliers")
    st.warning(f"WARNING: Check not yet implemented.")


def quickmap(
    df,
    lon_col="longitude",
    lat_col="latitude",
    popup=None,
    cluster=False,
    tiles="Stamen Terrain",
    aerial_imagery=False,
    kartverket=False,
    layer_name="Stations",
):
    """Make an interactive map from a point dataset. Can be used with any dataframe
    containing lat/lon co-ordinates (in WGS84 decimal degrees), but primarily
    designed to be used directly with the functions in nivapy.da.

    Args
        df:            Dataframe. Must include columns for lat and lon in WGS84 decimal degrees
        lon_col:       Str. Column with longitudes
        lat_col:       Str. Column with latitudes
        popup:         Str or None. Default None. Column containing text for popup labels
        cluster:       Bool. Whether to implement marker clustering
        tiles:         Str. Basemap to use. See folium.Map for full details. Choices:
                            - 'OpenStreetMap'
                            - 'Mapbox Bright' (Limited levels of zoom for free tiles)
                            - 'Mapbox Control Room' (Limited levels of zoom for free tiles)
                            - 'Stamen' (Terrain, Toner, and Watercolor)
                            - 'Cloudmade' (Must pass API key)
                            - 'Mapbox' (Must pass API key)
                            - 'CartoDB' (positron and dark_matter)
                            - Custom tileset by passing a Leaflet-style URL to the tiles
                              parameter e.g. http://{s}.yourtiles.com/{z}/{x}/{y}.png
        aerial_imagery: Bool. Whether to include Google satellite serial imagery as an
                        additional layer
        kartverket:     Bool. Whether to include Kartverket's topographic map as an additonal
                        layer
        layer_name:     Str. Name of layer to create in "Table of Contents"

    Returns
        Folium map
    """
    # Drop NaN
    df2 = df.dropna(subset=[lon_col, lat_col])

    # Get data
    if popup:
        df2 = df2[[lat_col, lon_col, popup]]
        df2[popup] = df2[popup].astype(str)
    else:
        df2 = df2[[lat_col, lon_col]]

    # Setup map
    avg_lon = df2[lon_col].mean()
    avg_lat = df2[lat_col].mean()
    map1 = folium.Map(location=[avg_lat, avg_lon], zoom_start=4, tiles=tiles)

    # Add aerial imagery if desired
    if aerial_imagery:
        folium.raster_layers.TileLayer(
            tiles="http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="google",
            name="Google satellite",
            max_zoom=20,
            subdomains=["mt0", "mt1", "mt2", "mt3"],
            overlay=False,
            control=True,
        ).add_to(map1)

    if kartverket:
        folium.raster_layers.TileLayer(
            tiles="https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=topo4&zoom={z}&x={x}&y={y}",
            attr="karkverket",
            name="Kartverket topographic",
            overlay=False,
            control=True,
        ).add_to(map1)

    # Add feature group to map
    grp = folium.FeatureGroup(name=layer_name)

    # Draw points
    if cluster and popup:
        locs = list(zip(df2[lat_col].values, df2[lon_col].values))
        popups = list(df2[popup].values)

        # Marker cluster with labels
        marker_cluster = MarkerCluster(locations=locs, popups=popups)
        grp.add_child(marker_cluster)
        grp.add_to(map1)

    elif cluster and not popup:
        locs = list(zip(df2[lat_col].values, df2[lon_col].values))
        marker_cluster = FastMarkerCluster(data=locs)
        grp.add_child(marker_cluster)
        grp.add_to(map1)

    elif not cluster and popup:  # Plot separate circle markers, with popup
        for idx, row in df2.iterrows():
            marker = folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5,
                weight=1,
                color="black",
                popup=folium.Popup(row[popup], parse_html=False),
                fill_color="red",
                fill_opacity=1,
            )
            grp.add_child(marker)
        grp.add_to(map1)

    else:  # Plot separate circle markers, no popup
        for idx, row in df2.iterrows():
            marker = folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5,
                weight=1,
                color="black",
                fill_color="red",
                fill_opacity=1,
            )
            grp.add_child(marker)
        grp.add_to(map1)

    # Add layer control
    folium.LayerControl().add_to(map1)

    # Zoom to data
    xmin, xmax = df2[lon_col].min(), df2[lon_col].max()
    ymin, ymax = df2[lat_col].min(), df2[lat_col].max()
    map1.fit_bounds([[ymin, xmin], [ymax, xmax]])

    return map1
