import os
import io



import numpy as np
import pydeck as pdk
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from lightgbm import LGBMClassifier
from graphviz import Digraph
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import zipfile

# Set page title and favicon
st.set_page_config(page_title="Rain Prediction for Australia", page_icon="🌧️")



df = pd.read_csv('weatherAUS.csv')
df_api = pd.read_csv('meteo_api.csv')
# Path to the zip file
zip_file_path = 'merged_df.zip'

# Extract the CSV file from the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Assuming the CSV file is the first file in the zip
    csv_filename = zip_ref.namelist()[0]
    with zip_ref.open(csv_filename) as file:
        # Read the CSV file into a DataFrame
        merged_df = pd.read_csv(file)

# Identifying numerical and categorical variables
def grab_col_names(dataframe, cat_th=3, car_th=50):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    # 1.3
    # Kategorik ve numerik değişken analizi

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)



# Radio button for selecting section
selected_section = st.sidebar.radio("Select Section", ["Data Overview",
                                                       "EDA",
                                                       "Web Scraping",
                                                       "Feature Engineering",
                                                       "Data Preparation",
                                                       "Modeling",
                                                       "Test"])

#######################################################################################################################
# Display selected section
if selected_section == "Data Overview":
    st.sidebar.title('Data Overview')
    # Sidebar option to select the section
    section = st.sidebar.selectbox("Select Section", ["Dataset", "Unique Values", "Data Types", "Summary Statistics",
                                                      "Observations and Variables", "Missing Values"])

    # Display the selected section
    if section == "Dataset":
        import streamlit as st

        st.title('Overview')
        st.markdown("""
        The Weather Australia dataset is a collection of historical weather observations recorded at various weather stations across Australia between 2007 and 2017. It contains a comprehensive set of meteorological variables, including temperature, rainfall, humidity, wind speed, atmospheric pressure, and more. The dataset is widely used by researchers, meteorologists, and data scientists for weather analysis, forecasting, and climate studies.
        """)

        st.title('Features')

        st.markdown("""
        | Feature           | Description                                                            |
        |-------------------|------------------------------------------------------------------------|
        | **Date**          | Date of the weather observation.                                       |
        | **Location**      | Name or code of the weather station.                                   |
        | **MinTemp**       | Minimum temperature recorded (in degrees Celsius).                     |
        | **MaxTemp**       | Maximum temperature recorded (in degrees Celsius).                     |
        | **Rainfall**      | Amount of rainfall recorded (in millimeters).                          |
        | **Evaporation**   | Water evaporation (in millimeters).                                    |
        | **Sunshine**      | Hours of bright sunshine recorded.                                     |
        | **WindGustSpeed** | Maximum wind gust speed (in kilometers per hour).                      |
        | **WindSpeed9am**  | Wind speed at 9 am (in kilometers per hour).                          |
        | **WindSpeed3pm**  | Wind speed at 3 pm (in kilometers per hour).                          |
        | **Humidity9am**   | Relative humidity at 9 am (in percentage).                            |
        | **Humidity3pm**   | Relative humidity at 3 pm (in percentage).                            |
        | **Pressure9am**   | Atmospheric pressure at 9 am (in hPa).                                 |
        | **Pressure3pm**   | Atmospheric pressure at 3 pm (in hPa).                                 |
        | **Cloud9am**      | Cloud cover at 9 am (in octas).                                        |
        | **Cloud3pm**      | Cloud cover at 3 pm (in octas).                                        |
        | **Temp9am**       | Temperature at 9 am (in degrees Celsius).                             |
        | **Temp3pm**       | Temperature at 3 pm (in degrees Celsius).                             |
        | **RainToday**     | Binary variable indicating if it rained today (1 for "Yes", 0 for "No").|
        | **RainTomorrow**  | Binary target variable indicating if it will rain tomorrow (1 for "Yes", 0 for "No").|
        """)

        st.title('Usage')
        st.markdown("""
        Feel free to use this dataset for research, analysis, or machine learning projects. Ensure to cite the source appropriately.
        """)

        st.title('Data Sources')
        st.markdown("""
        - Australian Bureau of Meteorology (BOM) - Daily Weather Observations  
        - Australian Bureau of Meteorology (BOM) - Climate Data Online  
        - Open Meteo Archive API  
        """)

        st.title('Models Used')
        st.markdown("""
        - **Logistic Regression:** A simple linear model used for binary classification tasks.  
        - **Random Forest Classifier:** An ensemble learning method based on decision trees, known for its robustness and accuracy.  
        - **LightGBM:** A gradient boosting framework that uses tree-based learning algorithms and is optimized for speed and efficiency.
        """)

        st.image("ivan-tsaregorodtsev-gvEYSYj0tOA-unsplash.jpg",
                 caption="Opera house amidst morning fog by Ivan Tsaregorodtsev",
                 use_column_width="always",
                 width=500)

        st.dataframe(df.head(), use_container_width=True)
    elif section == "Unique Values":
        st.write("## Number of Unique Values")
        st.dataframe(df.nunique(), use_container_width=True)

    elif section == "Data Types":
        st.write("## Data Types")
        st.dataframe(df.dtypes, use_container_width=True)

    elif section == "Summary Statistics":
        st.write("## Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    elif section == "Observations and Variables":
        st.write("## Number of Observations and Variables")
        st.write(f"Number of observations: {df.shape[0]}")
        st.write(f"Number of variables: {df.shape[1]}")
    elif section == "Missing Values":
        st.write("## Missing Value Query")
        st.write("### Any Column Has Missing Values?")
        st.dataframe(df.isnull().any(), use_container_width=True)

        st.write("### Count of Missing Values in Each Column")
        st.dataframe(df.isna().sum(), use_container_width=True)

# Get dummies for categorical variables
df = pd.get_dummies(df, columns=["RainToday", "RainTomorrow"], drop_first=True)
df = df.rename(columns={"RainToday_Yes": "RainToday", "RainTomorrow_Yes": "RainTomorrow"})
df["RainToday"] = df["RainToday"].astype(int)
df["RainTomorrow"] = df["RainTomorrow"].astype(int)


#######################################################################################################################
# Display selected section
if selected_section == "EDA":
    st.sidebar.title('Exploratory Data Analysis')
    # Sidebar option to select the section
    section = st.sidebar.selectbox("Select Section", ["Numerical Variables Summary", "Categorical Variables Summary",
                                                      "Outliers Summary", "Correlation Matrix"])
    # Check if the directory exists, if not, create it
    if not os.path.exists("plots"):
        os.makedirs("plots")
    if section == "Numerical Variables Summary":
        st.write("## Numerical Variables Summary")
        for col in num_cols:
            plt_file_path = os.path.join("plots", f"{col}_histogram.png")
            if os.path.exists(plt_file_path):
                st.image(plt_file_path)
                st.write(df.groupby("RainTomorrow").agg({col: "mean"}))
            else:
                plt.figure(figsize=(10, 8))
                sns.histplot(x=col, data=df, kde=True)
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                if col in ["Evaporation", "Rainfall"]:
                    plt.yscale("log")
                plt.ylabel('Frequency')
                plt.savefig(plt_file_path)
                st.image(plt_file_path)



    elif section == "Categorical Variables Summary":
        st.write("## Categorical Variables Summary")
        for col in cat_cols:
            plt_file_path = os.path.join("plots", f"{col}_barplot.png")
            if os.path.exists(plt_file_path):
                st.image(plt_file_path)
                st.write(df.groupby(col).agg({"RainTomorrow": "mean"}))
            else:
                plt.figure(figsize=(12, 10))
                sns.countplot(x=col, data=df, palette='viridis')
                plt.title(f'Bar Plot of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=90)
                plt.savefig(plt_file_path)
                st.image(plt_file_path)


    elif section == "Outliers Summary":
        st.write("## Outliers Summary")
        for col in num_cols:
            plt_file_path = os.path.join("plots", f"{col}_boxplot.png")
            if os.path.exists(plt_file_path):
                st.image(plt_file_path)

            else:
                plt.figure(figsize=(10, 8))
                sns.boxplot(x=col, data=df)
                plt.title(f'Box Plot of {col}')
                plt.xlabel(col)
                plt.ylabel('Value')
                plt.savefig(plt_file_path)
                st.image(plt_file_path)

    elif section == "Correlation Matrix":
        st.write("## Correlation Matrix")
        plt_file_path = os.path.join("plots", "correlation_matrix.png")
        corr_df = pd.concat([df[num_cols], df["RainToday"], df["RainTomorrow"]], axis=1)
        if os.path.exists(plt_file_path):
            st.image(plt_file_path)
        else:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.savefig(plt_file_path)
            st.image(plt_file_path)
def create_flowchart(title, nodes, edges, direction='TB'):
    st.write(f"## {title}")
    dot = Digraph(graph_attr={'size': '10,10', 'rankdir': direction})

    # Add nodes
    for node_name, node_label, node_shape, node_color in nodes:
        dot.node(node_name, node_label, shape=node_shape, style='filled', fillcolor=node_color, fontsize='16')

    # Add edges
    for edge_source, edge_target, edge_label, edge_color, edge_fontcolor in edges:
        dot.edge(edge_source, edge_target, label=edge_label, color=edge_color, fontcolor=edge_fontcolor)

    # Render the graph
    st.graphviz_chart(dot.source)
#######################################################################################################################
if selected_section == "Web Scraping":

    nodes = [('Start', 'Start', 'ellipse', 'red'),
             ('Loop', 'Loop', 'box', 'lightblue'),
             ('Scrapper', 'Scrapper', 'box', 'lightblue'),
             ('Sleep', 'Sleep', 'box', 'lightblue'),
             ('Concatenate', 'Concatenate', 'box', 'lightblue'),
             ('Create DataFrame', 'Create DataFrame', 'box', 'lightblue'),
             ('Rename Columns', 'Rename Columns', 'box', 'lightblue'),
             ('Drop Column', 'Drop Column', 'box', 'lightblue'),
             ('Save to CSV', 'Save to CSV', 'box', 'lightblue'),
             ('End', 'End', 'ellipse', 'green')]

    edges = [('Start', 'Loop', '', '', ''),
             ('Loop', 'Scrapper', '', '', ''),
             ('Scrapper', 'Sleep', '', '', ''),
             ('Sleep', 'Loop', '49 times', '', ''),
             ('Loop', 'Concatenate', '', '', ''),
             ('Concatenate', 'Create DataFrame', '', '', ''),
             ('Create DataFrame', 'Rename Columns', '', '', ''),
             ('Rename Columns', 'Drop Column', '', '', ''),
             ('Drop Column', 'Save to CSV', '', '', ''),
             ('Save to CSV', 'End', '', '', '')]

    create_flowchart("Web Scraping", nodes, edges)

    st.write(df_api.head())


######################################################################################################################


if selected_section == "Feature Engineering":

    cat_cols, num_cols, cat_but_car = grab_col_names(merged_df)
    # Sidebar title
    st.sidebar.title('Feature Engineering')

    # Sidebar option to select the section
    section = st.sidebar.selectbox("Select Section",
                                   ["Merge DataFrames", "Convert Angle to Directional Name",
                                    "Create Meteorological Seasons Variable", "Create Astronomical Season Variable",
                                    "Create Latitude, Longitude and Altitude Variables", "Create Climate Zone Variable",
                                    "One-Hot Encoding"])


    # Merge DataFrames
    if section == "Merge DataFrames":
        nodes = [('Start', 'Start', 'ellipse', 'red'),
                 ('df', 'df', 'parallelogram', 'lightblue'),
                 ('df_api', 'df_api', 'parallelogram', 'lightblue'),
                 ('merged_df', 'merged_df', 'parallelogram', 'lightblue'),
                 ('End', 'End', 'ellipse', 'green')]
        edges = [('Start', 'df', '', '', ''),
                 ('Start', 'df_api', '', '', ''),
                 ('df', 'merged_df', 'Date, Location', 'black', 'black'),
                 ('df_api', 'merged_df', 'Date, Location', 'black', 'white'),
                 ('merged_df', 'End', '', '', '')]
        create_flowchart('Merge DataFrames', nodes, edges)

    # Convert Angle to Directional Name
    elif section == "Convert Angle to Directional Name":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('end', 'End', 'ellipse', 'green'),
                 ('define', 'Define directional sectors and names', 'box', 'lightblue'),
                 ('calculate', 'Calculate index of nearest sector', 'box', 'lightblue'),
                 ('return', 'Return directional name', 'box', 'lightblue')]
        edges = [('start', 'define', '', '', ''),
                 ('define', 'calculate', '', '', ''),
                 ('calculate', 'return', '', '', ''),
                 ('return', 'end', '', '', '')]
        create_flowchart('Convert Angle to Directional Name', nodes, edges)

# Create Meteorological Seasons Variable
    elif section == "Create Meteorological Seasons Variable":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('convert_date', "Convert 'Date' column to datetime", 'box', 'lightblue'),
                 ('define_seasons', 'Define seasons based on month', 'box', 'lightblue'),
                 ('assign_seasons', 'Assign seasons', 'box', 'lightblue'),
                 ('end', 'End', 'ellipse', 'green'),
                 ('spring', 'Spring', 'diamond', 'purple'),
                 ('winter', 'Winter', 'diamond', 'blue'),
                 ('autumn', 'Autumn', 'diamond', 'orange'),
                 ('summer', 'Summer', 'diamond', 'brown')]
        edges = [('start', 'convert_date', '', '', ''),
                 ('convert_date', 'define_seasons', '', '', ''),
                 ('define_seasons', 'spring', 'September to November (9-11)', 'purple', 'purple'),
                 ('define_seasons', 'winter', 'June to August (6-8)', 'blue', 'blue'),
                 ('define_seasons', 'autumn', 'March to May (3-5)', 'orange', 'orange'),
                 ('define_seasons', 'summer', 'December to February (12-2)', 'brown', 'brown'),
                 ('spring', 'assign_seasons', 'Yes', 'purple', 'purple'),
                 ('winter', 'assign_seasons', 'Yes', 'blue', 'blue'),
                 ('autumn', 'assign_seasons', 'Yes', 'orange', 'orange'),
                 ('summer', 'assign_seasons', 'Yes', 'brown', 'brown'),
                 ('spring', 'define_seasons', 'No', 'purple', 'purple'),
                 ('winter', 'define_seasons', 'No', 'blue', 'blue'),
                 ('autumn', 'define_seasons', 'No', 'orange', 'orange'),
                 ('summer', 'define_seasons', 'No', 'brown', 'brown'),
                 ('assign_seasons', 'end', '', '', '')]
        create_flowchart('Create Meteorological Seasons Variable', nodes, edges)

    # Create Astronomical Season Variable
    elif section == "Create Astronomical Season Variable":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('define_seasons', 'Define seasons based on equinox ', 'box', 'lightblue'),
                 ('end', 'End', 'ellipse', 'green'),
                 ('spring', 'Spring (September 23 - December 21)', 'diamond', 'purple'),
                 ('summer', 'Summer (December 21 - March 21)', 'diamond', 'blue'),
                 ('fall', 'Fall (March 21 - June 21)', 'diamond', 'orange'),
                 ('winter', 'Winter (June 21 - September 23)', 'diamond', 'brown')]
        edges = [('start', 'define_seasons', '', '', ''),
                 ('define_seasons', 'spring',
                  'Is the month September and the day >= 23,\n or is the month October, November, or December?',
                  'purple', 'purple'),
                 ('define_seasons', 'summer',
                  'Is the month December and the day >= 21,\n or is the month January, February, or March?', 'blue',
                  'blue'),
                 ('define_seasons', 'fall',
                  'Is the month March and the day >= 21,\n or is the month April, May, or June?', 'orange', 'orange'),
                 ('define_seasons', 'winter', 'Otherwise (i.e., other months and days)', 'brown', 'brown'),
                 ('spring', 'summer', 'No', '', ''),
                 ('summer', 'fall', 'No', '', ''),
                 ('fall', 'winter', 'No', '', ''),
                 ('spring', 'end', 'Yes', 'purple', 'purple'),
                 ('summer', 'end', 'Yes', 'blue', 'blue'),
                 ('fall', 'end', 'Yes', 'orange', 'orange'),
                 ('winter', 'end', 'Yes', 'brown', 'brown')]
        create_flowchart('Create Astronomical Season Variable', nodes, edges)

    # Create Latitude, Longitude and Altitude Variables
    elif section == "Create Latitude, Longitude and Altitude Variables":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('assign_lat', 'Assign latitude values', 'box', 'lightblue'),
                 ('assign_long', 'Assign longitude values', 'box', 'lightblue'),
                 ('define_endpoint', 'Define API endpoint', 'box', 'lightblue'),
                 ('fetch_altitudes', 'Fetch altitudes for each location', 'box', 'lightblue'),
                 ('assign_alt', 'Assign altitude values', 'box', 'lightblue'),
                 ('end', 'End', 'ellipse', 'green')]
        edges = [('start', 'assign_lat', '', '', ''),
                 ('start', 'assign_long', '', '', ''),
                 ('assign_lat', 'define_endpoint', '', '', ''),
                 ('assign_long', 'define_endpoint', '', '', ''),
                 ('define_endpoint', 'fetch_altitudes', '', '', ''),
                 ('fetch_altitudes', 'assign_alt', '', '', ''),
                 ('assign_alt', 'end', '', '', '')]
        create_flowchart('Create Latitude, Longitude and Altitude Variables', nodes, edges)

    # Create Climate Zone Variable
    elif section == "Create Climate Zone Variable":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('assign_lat', 'Assign latitude values', 'box', 'lightblue'),
                 ('define_climate_zone', 'Define get_climate_zone function', 'box', 'lightblue'),
                 ('apply_climate_zone', 'Apply get_climate_zone function to create "Climate_Zone" column', 'box',
                  'lightblue'),
                 ('end', 'End', 'ellipse', 'green'),
                 ('tropical', 'Latitude <= -23.5 \n Tropical', 'diamond', 'red'),
                 ('subtropical', '-23.5 < Latitude < -40 \n Subtropical', 'diamond', 'orange'),
                 ('temperate', '-40 < Latitude <= -60 \n Temperate', 'diamond', 'blue')]
        edges = [('start', 'assign_lat', '', '', ''),
                 ('assign_lat', 'define_climate_zone', '', '', ''),
                 ('define_climate_zone', 'tropical', '', '', ''),
                 ('define_climate_zone', 'subtropical', '', '', ''),
                 ('define_climate_zone', 'temperate', '', '', ''),
                 ('apply_climate_zone', 'end', '', '', ''),
                 ('tropical', 'apply_climate_zone', 'Yes', 'red', 'red'),
                 ('subtropical', 'apply_climate_zone', 'Yes', 'orange', 'orange'),
                 ('temperate', 'apply_climate_zone', 'Yes', 'blue', 'blue'),
                 ('tropical', 'end', 'No', 'red', 'red'),
                 ('subtropical', 'end', 'No', 'orange', 'orange'),
                 ('temperate', 'end', 'No', 'blue', 'blue')]
        create_flowchart('Create Climate Zone Variable', nodes, edges)


        # One-Hot Encoding
    elif section == "One-Hot Encoding":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('one_hot_encoding',
                  'Perform One-Hot Encoding\non "WindDirDom", "Seasons", "Astron_Season",\nand "Climate_Zone" columns',
                  'box', 'lightblue'),
                 ('drop_column', 'Drop "Unnamed: 0" column', 'box', 'lightblue'),
                 ('save_csv', 'Save merged DataFrame to "merged_df.csv"', 'box', 'lightblue'),
                 ('read_csv', 'Read "merged_df.csv" back\ninto a DataFrame', 'box', 'lightblue'),
                 ('end', 'End', 'ellipse', 'green')]
        edges = [('start', 'one_hot_encoding', '', '', ''),
                 ('one_hot_encoding', 'drop_column', '', '', ''),
                 ('drop_column', 'save_csv', '', '', ''),
                 ('save_csv', 'read_csv', '', '', ''),
                 ('read_csv', 'end', '', '', '')]
        create_flowchart('One-Hot Encoding', nodes, edges)

        ######################################################################################################################

if selected_section == "Data Preparation":
    # Sidebar title
    st.sidebar.title('Data Preparation')

    # Sidebar option to select the section
    section = st.sidebar.selectbox("Select Section",
                                   ["Identify Missing Values",
                                    "Fill NaN Values",
                                    "Replace Outliers with Thresholds",
                                    "Scaling"])

# Display different sections based on the selected option

    # Identify Missing Values
    if section == "Identify Missing Values":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('fill_min_temp', 'Fill NaN values in "MinTemp"\nwith "temperature_2m_min" values', 'box', 'lightblue'),
                 ('fill_max_temp', 'Fill NaN values in "MaxTemp"\nwith "temperature_2m_max" values', 'box', 'lightblue'),
                 ('fill_rainfall', 'Fill NaN values in "Rainfall"\nwith "rain_sum" values', 'box', 'lightblue'),
                 ('fill_evaporation', 'Fill NaN values in "Evaporation"\nwith "et0_fao_evapotranspiration" values', 'box',
                  'lightblue'),
                 ('fill_wind_gust_speed', 'Fill NaN values in "WindGustSpeed"\nwith "wind_gusts_10m_max" values', 'box',
                  'lightblue'),
                 ('fill_wind_speed',
                  'Fill NaN values in "WindSpeed9am" and "WindSpeed3pm"\nwith "wind_speed_10m_max" values', 'box',
                  'lightblue'),
                 ('drop_min_temp', 'Drop "temperature_2m_min" column', 'box', 'lightblue'),
                 ('drop_max_temp', 'Drop "temperature_2m_max" column', 'box', 'lightblue'),
                 ('drop_rainfall', 'Drop "precipitation_sum" and "rain_sum" columns', 'box', 'lightblue'),
                 ('drop_evaporation', 'Drop "et0_fao_evapotranspiration" column', 'box', 'lightblue'),
                 ('drop_wind_gust_speed', 'Drop "wind_gusts_10m_max" column', 'box', 'lightblue'),
                 ('drop_wind_speed', 'Drop "wind_speed_10m_max" column', 'box', 'lightblue'),
                 ('end', 'End', 'ellipse', 'green')]
        edges = [('start', 'fill_min_temp', '', '', ''),
                 ('start', 'fill_max_temp', '', '', ''),
                 ('start', 'fill_rainfall', '', '', ''),
                 ('start', 'fill_evaporation', '', '', ''),
                 ('start', 'fill_wind_gust_speed', '', '', ''),
                 ('start', 'fill_wind_speed', '', '', ''),
                 ('fill_min_temp', 'drop_min_temp', '', '', ''),
                 ('fill_max_temp', 'drop_max_temp', '', '', ''),
                 ('fill_rainfall', 'drop_rainfall', '', '', ''),
                 ('fill_evaporation', 'drop_evaporation', '', '', ''),
                 ('fill_wind_gust_speed', 'drop_wind_gust_speed', '', '', ''),
                 ('fill_wind_speed', 'drop_wind_speed', '', '', ''),
                 ('drop_min_temp', 'end', '', '', ''),
                 ('drop_max_temp', 'end', '', '', ''),
                 ('drop_rainfall', 'end', '', '', ''),
                 ('drop_evaporation', 'end', '', '', ''),
                 ('drop_wind_gust_speed', 'end', '', '', ''),
                 ('drop_wind_speed', 'end', '', '', '')]
        create_flowchart('Identify Missing Values', nodes, edges, direction='LR')

    # Fill NaN Values
    elif section == "Fill NaN Values":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('fill_pressure9am', 'Fill NaN values in "Pressure9am"\nwith mean values of "Pressure9am"', 'box',
                  'lightblue'),
                 ('fill_pressure3pm', 'Fill NaN values in "Pressure3pm"\nwith mean values of "Pressure3pm"', 'box',
                  'lightblue'),
                 ('fill_humidity9am', 'Fill NaN values in "Humidity9am"\nwith mean values of "Humidity9am"', 'box',
                  'lightblue'),
                 ('fill_humidity3pm', 'Fill NaN values in "Humidity3pm"\nwith mean values of "Humidity3pm"', 'box',
                  'lightblue'),
                 ('fill_cloud9am', 'Fill NaN values in "Cloud9am"\nwith mean values of "Cloud9am"', 'box', 'lightblue'),
                 ('fill_cloud3pm', 'Fill NaN values in "Cloud3pm"\nwith mean values of "Cloud3pm"', 'box', 'lightblue'),
                 ('fill_temp9am', 'Fill NaN values in "Temp9am"\nwith mean values of "Temp9am"', 'box', 'lightblue'),
                 ('fill_temp3pm', 'Fill NaN values in "Temp3pm"\nwith mean values of "Temp3pm"', 'box', 'lightblue'),
                 ('fill_sunshine', 'Fill NaN values in "Sunshine"\nwith mean values of "Sunshine"', 'box', 'lightblue'),
                 ('end', 'End', 'ellipse', 'green')]
        edges = [('start', 'fill_pressure9am', '', '', ''),
                 ('start', 'fill_pressure3pm', '', '', ''),
                 ('start', 'fill_humidity9am', '', '', ''),
                 ('start', 'fill_humidity3pm', '', '', ''),
                 ('start', 'fill_cloud9am', '', '', ''),
                 ('start', 'fill_cloud3pm', '', '', ''),
                 ('start', 'fill_temp9am', '', '', ''),
                 ('start', 'fill_temp3pm', '', '', ''),
                 ('start', 'fill_sunshine', '', '', ''),
                 ('fill_pressure9am', 'end', '', '', ''),
                 ('fill_pressure3pm', 'end', '', '', ''),
                 ('fill_humidity9am', 'end', '', '', ''),
                 ('fill_humidity3pm', 'end', '', '', ''),
                 ('fill_cloud9am', 'end', '', '', ''),
                 ('fill_cloud3pm', 'end', '', '', ''),
                 ('fill_temp9am', 'end', '', '', ''),
                 ('fill_temp3pm', 'end', '', '', ''),
                 ('fill_sunshine', 'end', '', '', '')]
        create_flowchart('Fill NaN Values', nodes, edges, direction="LR")



    # Replace Outliers with Thresholds
    elif section == "Replace Outliers with Thresholds":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('calculate_thresholds', 'Calculate outlier thresholds\n(q1, q3, IQR)', 'box', 'lightblue'),
                 ('replace_outliers', 'Replace outliers with thresholds', 'box', 'lightblue'),
                 ('check_outliers', 'Check for remaining outliers', 'box', 'lightblue'),
                 ('end', 'End', 'ellipse', 'green')]
        edges = [('start', 'calculate_thresholds', '', '', ''),
                 ('calculate_thresholds', 'replace_outliers', '', '', ''),
                 ('replace_outliers', 'check_outliers', '', '', ''),
                 ('check_outliers', 'end', '', '', '')]
        create_flowchart('Replace Outliers with Thresholds', nodes, edges)


    # Scaling
    elif section == "Scaling":
        nodes = [('start', 'Start', 'ellipse', 'red'),
                 ('separate_columns', 'Separate columns\nbased on scaling method', 'box', 'lightblue'),
                 ('define_transformer', 'Define ColumnTransformer\nfor different scalers', 'box', 'lightblue'),
                 ('apply_standard_scaler', 'Apply Standard Scaler\nto Gaussian columns', 'box', 'lightblue'),
                 ('apply_robust_scaler', 'Apply Robust Scaler\nto non-Gaussian columns', 'box', 'lightblue'),
                 ('end', 'End', 'ellipse', 'green')]
        edges = [('start', 'separate_columns', '', '', ''),
                 ('separate_columns', 'define_transformer', '', '', ''),
                 ('define_transformer', 'apply_standard_scaler', '', '', ''),
                 ('define_transformer', 'apply_robust_scaler', '', '', ''),
                 ('apply_standard_scaler', 'end', '', '', ''),
                 ('apply_robust_scaler', 'end', '', '', '')]
        create_flowchart('Scaling', nodes, edges)

######################################################################################################################
cat_cols, num_cols, cat_but_car = grab_col_names(merged_df)
def calculate_outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculate the lower and upper outlier thresholds based on the interquartile range (IQR).

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - col_name (str): The name of the column for which outliers are being calculated.
    - q1 (float, optional): The first quartile value. Default is 0.25.
    - q3 (float, optional): The third quartile value. Default is 0.75.

    Returns:
    - tuple: A tuple containing the lower and upper outlier thresholds.
    """

    q1_value, q3_value = dataframe[col_name].quantile([q1, q3])
    iqr = q3_value - q1_value
    return float(q1_value - 1.5 * iqr), float(q3_value + 1.5 * iqr)

def replace_outliers_with_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Replace outliers in a column of a DataFrame with the lower and upper thresholds.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the column with outliers.
    variable : str
        The name of the column with outliers.
    q1 : float, optional
        The first quartile value. Default is 0.25.
    q3 : float, optional
        The third quartile value. Default is 0.75.

    Returns
    -------
    None
    """
    low_limit, up_limit = calculate_outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe.loc[dataframe[col_name] < float(low_limit), col_name] = int(low_limit)
    dataframe.loc[dataframe[col_name] > float(up_limit), col_name] = int(up_limit)
def check_outlier(dataframe, col_name):
    low_limit, up_limit = calculate_outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    replace_outliers_with_thresholds(merged_df, col)

def logistic_regression_feature_importance(model, feature_names):
    """
    Calculate and display feature importance for a logistic regression model.

    Args:
    - model (LogisticRegression): The trained logistic regression model.
    - feature_names (list): List of feature names used in the model.

    Returns:
    - dict: A dictionary mapping feature names to their importance scores.
    """
    if not hasattr(model, 'coef_'):
        raise ValueError("Model does not have coefficients attribute. "
                         "Please provide a trained logistic regression model.")

    if len(feature_names) != len(model.coef_[0]):
        raise ValueError("Number of feature names does not match the number of coefficients.")

    importances = abs(model.coef_[0])
    feature_importance = dict(zip(feature_names, importances))

    # Sort features by importance (descending order)
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=False))

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(list(feature_importance.keys()), list(feature_importance.values()))
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance for Logistic Regression Model')

    # Display plot in Streamlit
    st.pyplot(fig)

    return feature_importance

def plot_feature_importance(model, feature_names, model_name):
    """
    Calculate and plot feature importance for a given model.

    Args:
    - model: The trained model (e.g., RandomForestClassifier, GradientBoostingRegressor).
    - feature_names (list): List of feature names used in the model.
    - model_name (str): Name of the model for the plot title.

    Returns:
    - dict: A dictionary mapping feature names to their importance scores.
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute. "
                         "Please provide a model with feature importances.")

    if len(feature_names) != len(model.feature_importances_):
        raise ValueError("Number of feature names does not match the number of feature importances.")

    importances = model.feature_importances_
    feature_importance = dict(zip(feature_names, importances))

    # Sort features by importance (descending order)
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=False))

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.barh(list(feature_importance.keys()), list(feature_importance.values()))
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for {model_name} Model')
    st.pyplot(plt)

    return feature_importance


gaussian_columns = []
non_gaussian_columns = []
for col in merged_df.select_dtypes(include=['float64', 'int64']).columns:
    if not check_outlier(merged_df, col):
        gaussian_columns.append(col)
    else:
        non_gaussian_columns.append(col)

# Define a ColumnTransformer to apply different scalers to different columns
preprocessor = ColumnTransformer([
    ('standard_scaler', StandardScaler(), gaussian_columns),
    ('robust_scaler', RobustScaler(), non_gaussian_columns)
])

# Apply the scaling to the dataset
scaled_features = preprocessor.fit_transform(merged_df.select_dtypes(include=['float64', 'int64']))

# Replace the original columns with the scaled features
merged_df[gaussian_columns + non_gaussian_columns] = scaled_features
if selected_section == "Modeling":
    # Sidebar title
    st.sidebar.title("Modeling")

    # Sidebar option to select the section
    section = st.sidebar.selectbox("Select Section",
                                   ["Logistic Regression", "Random Forest Classifier", "LightGBM Classifier", "Stacked Model"])

    # Create a multiselect widget for selecting X variables
    # Select X variables
    selected_columns = st.multiselect(
        "Select X variables:",
        ["Select All"] + [col for col in merged_df.columns.tolist() if col not in ["RainTomorrow", "Date", "Location"]],
        default=["Select All"]
    )

    # Check if "Select All" is selected
    if "Select All" in selected_columns:
        X = merged_df.drop(["RainTomorrow", "Date", "Location"], axis=1)
    else:
        # Create the X dataframe with the selected columns
        X = merged_df[selected_columns]

    # y remains the same
    y = merged_df["RainTomorrow"]

    # Splitting the dataset into training and testing sets
    test_size = st.slider("Select the test size:", 0.1, 0.5, 0.2, 0.05)
    train_size = 1 - test_size

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size,
                                                        random_state=1)

    if section == "Logistic Regression":

        # Model 1: Logistic Regression
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        y_pred_log_reg = log_reg.predict(X_test)

        # Display the classification report
        st.write("Logistic Regression")
        log_reg_params = log_reg.get_params()
        if st.button("Get Parameters", key="log_reg_button"):
            st.write(log_reg_params)
        report_str = classification_report(y_test, y_pred_log_reg)
        report_df = pd.read_fwf(io.StringIO(report_str), index_col=0)
        st.dataframe(report_df,  use_container_width=True)

        logistic_regression_feature_importance(log_reg, X.columns.tolist())

        if st.button("Hyperparameter Optimization"):
            # Define the hyperparameter grid for Logistic Regression
            param_grid_log_reg = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }

            # Perform RandomizedSearchCV for Logistic Regression
            random_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=param_grid_log_reg, n_iter=3,
                                                cv=5)
            random_log_reg.fit(X_train, y_train)

            # Get the best parameters and best score
            best_params_log_reg = random_log_reg.best_params_
            best_score_log_reg = random_log_reg.best_score_

            # Display the best parameters and best score
            if st.button("Get Parameters", key="log_reg_button_opt"):
                st.write("Best Parameters for Logistic Regression:")
                st.write(best_params_log_reg)

            # Use the best model to make predictions
            best_log_reg = random_log_reg.best_estimator_
            y_pred_log_reg = best_log_reg.predict(X_test)

            # Display the classification report
            st.write("Classification Report for Logistic Regression:")
            report_str_log_reg = classification_report(y_test, y_pred_log_reg)
            report_df_log_reg = pd.read_fwf(io.StringIO(report_str_log_reg), index_col=0)
            st.dataframe(report_df_log_reg, use_container_width=True)

            logistic_regression_feature_importance(best_log_reg, X.columns.tolist())


    elif section == "Random Forest Classifier":
        # Model 2: Random Forest Classifier
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        y_pred_rf = rf_classifier.predict(X_test)

        # Display the classification report
        st.write("Random Forest Classifier")
        rf_classifier_params = rf_classifier.get_params()
        if st.button("Get Parameters", key="rf_button"):
            st.write(rf_classifier_params)

        report_str = classification_report(y_test, y_pred_rf)
        report_df = pd.read_fwf(io.StringIO(report_str), index_col=0)
        st.dataframe(report_df, use_container_width=True)

        plot_feature_importance(rf_classifier, X.columns.tolist(), "Random Forest Classifier")

        if st.button("Hyperparameter Optimization"):
            # Define the hyperparameter grid for Random Forest Classifier
            param_grid_rf = {
                'n_estimators': [100, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }

            # Perform RandomizedSearchCV for Random Forest Classifier
            random_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid_rf, n_iter=4, cv=5)
            random_rf.fit(X_train, y_train)

            # Get the best parameters and best score
            best_params_rf = random_rf.best_params_
            best_score_rf = random_rf.best_score_

            if st.button("Get Parameters", key="rf_button_opt"):
                st.write("Best Parameters for Random Forest Classifier:")
                st.write(best_params_rf)
            # Use the best model to make predictions
            best_rf = random_rf.best_estimator_
            y_pred_rf = best_rf.predict(X_test)

            # Display the classification report
            st.write("Classification Report for Random Forest Classifier:")
            report_str_rf = classification_report(y_test, y_pred_rf)
            report_df_rf = pd.read_fwf(io.StringIO(report_str_rf), index_col=0)
            st.dataframe(report_df_rf, use_container_width=True)
            # Plot feature importance
            plot_feature_importance(best_rf, X.columns.tolist(), "Random Forest Classifier")

    elif section == "LightGBM Classifier":
        # Model 3: LightGBM Classifier
        lgb_classifier = LGBMClassifier()
        lgb_classifier.fit(X_train, y_train)
        y_pred_lgb = lgb_classifier.predict(X_test)
        # Save the model to a file
        """joblib.dump(lgb_classifier, 'lgb_model.pkl')"""
        lgb_params = lgb_classifier.get_params()
        st.write("LightGBM Classifier")
        if st.button("Get Parameters", key="lgb_button"):
            st.write(lgb_params)

        report_str = classification_report(y_test, y_pred_lgb)
        report_df = pd.read_fwf(io.StringIO(report_str), index_col=0)
        st.dataframe(report_df, use_container_width=True)

        plot_feature_importance(lgb_classifier, X.columns.tolist(), "LightGBM Classifier")

        if st.button("Hyperparameter Optimization"):
            # Define the hyperparameter grid for LightGBM Classifier
            param_grid_lgb = {
                'n_estimators': [100, 300],
                'max_depth': [-1, 5, 10],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            }

            # Perform RandomizedSearchCV for LightGBM Classifier
            random_lgb = RandomizedSearchCV(LGBMClassifier(), param_distributions=param_grid_lgb, n_iter=4, cv=5)
            random_lgb.fit(X_train, y_train)

            # Get the best parameters and best score
            best_params_lgb = random_lgb.best_params_
            best_score_lgb = random_lgb.best_score_


            if st.button("Get Parameters", key="lgb_button_opt"):
                st.write("Best Parameters for LightGBM Classifier:")
                st.write(best_params_lgb)

            # Use the best model to make predictions
            best_lgb = random_lgb.best_estimator_
            y_pred_lgb = best_lgb.predict(X_test)

            # Display the classification report
            st.write("Classification Report for LightGBM Classifier:")
            report_str_lgb = classification_report(y_test, y_pred_lgb)
            report_df_lgb = pd.read_fwf(io.StringIO(report_str_lgb), index_col=0)
            st.dataframe(report_df_lgb, use_container_width=True)


            # Plot feature importance
            plot_feature_importance(best_lgb, X.columns.tolist(), "LightGBM Classifier")



    elif section == "Stacked Model":
        # Initialize the base models
        base_models = [
            ('lr', LogisticRegression()),
            ('rf', RandomForestClassifier()),
            ('lgb', LGBMClassifier())
        ]

        # Initialize the stacking classifier
        stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

        # Train the stacking classifier
        stacking_clf.fit(X_train, y_train)

        # Make predictions
        y_pred_stacked = stacking_clf.predict(X_test)

        stacking_clf_params = stacking_clf.get_params()
        st.write("Stacked Model")
        if st.button("Get Parameters", key="stacking_clf_button"):
            st.write(stacking_clf_params)

        report_str = classification_report(y_test, y_pred_stacked)
        report_df = pd.read_fwf(io.StringIO(report_str), index_col=0)
        st.dataframe(report_df, use_container_width=True)

        # Plot feature importances for each base model
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # Get the base models from the stacking classifier
        base_lr = stacking_clf.named_estimators_['lr']
        base_rf = stacking_clf.named_estimators_['rf']
        base_lgb = stacking_clf.named_estimators_['lgb']
        # Logistic Regression
        importance_lr = base_lr.coef_[0]
        axs[0].barh(X_train.columns, importance_lr)
        axs[0].set_title('Logistic Regression Feature Importance')

        # Random Forest
        importance_rf = base_rf.feature_importances_
        axs[1].barh(X_train.columns, importance_rf)
        axs[1].set_title('Random Forest Feature Importance')

        # LightGBM
        importance_lgb = base_lgb.feature_importances_
        axs[2].barh(X_train.columns, importance_lgb)
        axs[2].set_title('LightGBM Feature Importance')

        plt.tight_layout()
        st.pyplot(fig)

        if st.button("Hyperparameter Optimization"):
            # Define the parameter grid for hyperparameter optimization
            param_grid_stacked = {
                'final_estimator__C': [0.1, 1, 10],
                'final_estimator__penalty': ['l1', 'l2'],
                'final_estimator__solver': ['liblinear', 'lbfgs']
            }
            # Perform RandomizedSearchCV for the stacked model
            random_stacked = RandomizedSearchCV(stacking_clf, param_distributions=param_grid_stacked, n_iter=4, cv=5)
            random_stacked.fit(X_train, y_train)

            # Get the best parameters and best score
            best_params_stacked = random_stacked.best_params_
            best_score_stacked = random_stacked.best_score_

            if st.button("Get Parameters", key="stacking_clf_button_opt"):
                st.write("Best Parameters for Stacked Model:")
                st.write(best_params_stacked)

            # Use the best model to make predictions
            best_stacked = random_stacked.best_estimator_
            y_pred_stacked = best_stacked.predict(X_test)

            # Display the classification report
            st.write("Classification Report for Stacked Model:")
            report_str_stacked = classification_report(y_test, y_pred_stacked)
            report_df_stacked = pd.read_fwf(io.StringIO(report_str_stacked), index_col=0)
            st.dataframe(report_df_stacked, use_container_width=True)

            plot_feature_importance(best_stacked, X.columns.tolist(), "Stacked Model")



#######################################################################################################################

if selected_section == "Test":

    # Sidebar
    location_info = {
        'Albury': (-36.0748, 146.924),
        'BadgerysCreek': (-33.8907, 150.7426),
        'Cobar': (-31.4967, 145.8344),
        'CoffsHarbour': (-30.2963, 153.1135),
        'Moree': (-29.4628, 149.8416),
        'Newcastle': (-32.9295, 151.7801),
        'NorahHead': (-33.281544, 151.579147),
        'NorfolkIsland': (-29.0333, 167.95),
        'Penrith': (-33.75, 150.7),
        'Richmond': (-41.3333, 173.1833),
        'Sydney': (-33.8678, 151.2073),
        'SydneyAirport': (-33.9399228, 151.17527640000003),
        'WaggaWagga': (-28.3339, 116.9352),
        'Williamtown': (-32.8064, 151.8436),
        'Wollongong': (-34.424, 150.8935),
        'Canberra': (-35.2835, 149.1281),
        'Tuggeranong': (-35.4568, 149.1099),
        'MountGinini': (-35.5307, 148.7713),
        'Ballarat': (-37.5662, 143.8496),
        'Bendigo': (-36.7582, 144.2802),
        'Sale': (-38.111, 147.068),
        'MelbourneAirport': (-37.667111, 144.833480766796),
        'Melbourne': (-37.814, 144.9633),
        'Mildura': (-34.1855, 142.1625),
        'Nhil': (-36.3333, 141.65),
        'Portland': (-38.354, 141.574),
        'Watsonia': (-37.7167, 145.0833),
        'Dartmoor': (-37.9222, 141.2749),
        'Brisbane': (-27.4679, 153.0281),
        'Cairns': (-16.9237, 145.7661),
        'GoldCoast': (-28.0003, 153.4309),
        'Townsville': (-19.2664, 146.8057),
        'Adelaide': (-34.9287, 138.5986),
        'MountGambier': (-37.8318, 140.7792),
        'Nuriootpa': (-34.4682, 138.9977),
        'Woomera': (-31.1998, 136.8326),
        'Albany': (-35.0269, 117.8837),
        'Witchcliffe': (-34.0333, 115.1),
        'PearceRAAF': (-31.667663996, 116.008999964),
        'PerthAirport': (-31.9321, 115.9564),
        'Perth': (-31.9522, 115.8614),
        'SalmonGums': (-32.9833, 121.6333),
        'Walpole': (-34.976, 116.7302),
        'Hobart': (-42.8794, 147.3294),
        'Launceston': (-41.4388, 147.1347),
        'AliceSprings': (-23.6975, 133.8836),
        'Darwin': (-12.4611, 130.8418),
        'Katherine': (-14.4652, 132.2635),
        'Uluru': (-25.3457, 131.0367)
    }

    # Modelling
    X = merged_df.drop(["RainTomorrow", "Date", "Location"], axis=1)
    y = merged_df["RainTomorrow"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Load the model
    # model = joblib.load('lgb_model.pkl')
    lgb_classifier = LGBMClassifier()
    lgb_classifier.fit(X_train, y_train)

    # Get the indices of X_train
    train_indices = X_train.index

    # Get the indices of merged_df
    merged_indices = merged_df.index

    # Find the intersection of indices
    common_indices = train_indices.intersection(merged_indices)

    # Use the common indices to filter the merged_df
    common_locations_dates_df = merged_df.loc[common_indices]

    # Get unique locations and dates
    unique_locations = common_locations_dates_df['Location'].unique()
    unique_dates = common_locations_dates_df['Date'].unique()

    # Sidebar
    st.sidebar.title('Select Date and Location')
    selected_location = st.sidebar.selectbox('Select Location', unique_locations, key="location_select")

    # Filter X_train based on the selected location
    X_test_selected_loc = X_test[X_test.index.isin(merged_df[merged_df['Location'] == selected_location].index)]

    # Get unique dates for the selected location from the filtered X_train
    unique_dates = merged_df.loc[X_test_selected_loc.index, 'Date'].unique()

    # Find the minimum and maximum dates for the selected location
    min_date = min(unique_dates)
    max_date = max(unique_dates)

    # Set default value to the maximum date
    selected_date = st.sidebar.selectbox('Select Date',
                                         options=unique_dates,
                                         index=len(unique_dates) - 1,
                                         key="date_select")

    # Retrieve latitude and longitude for a selected location
    latitude, longitude = location_info[selected_location]

    # Filter data
    filtered_df = merged_df[(pd.to_datetime(merged_df['Date']) == selected_date)
                            & (merged_df['Location'] == selected_location)]

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=8,
        bearing=0,
        pitch=0,
    )

    # Create a PyDeck scatter plot layer for all locations with text labels
    scatter_layer_all = pdk.Layer(
        'ScatterplotLayer',
        data=[{"latitude": lat, "longitude": long} for key, (lat, long) in location_info.items()],
        get_position='[longitude, latitude]',
        get_radius=8000,
        get_fill_color=[255, 0, 0],
        opacity=0.1,
        pickable=True
    )

    # Create a PyDeck scatter plot layer for the selected location with text label
    scatter_layer_selected = pdk.Layer(
        'ScatterplotLayer',
        data=[{"latitude": latitude, "longitude": longitude}],
        get_position='[longitude, latitude]',
        get_radius=8000,
        get_fill_color=[255, 0, 0],
        opacity=0.1,
        get_text='text',
        get_text_anchor='middle',
        get_text_size=20,
        get_text_color=[0, 0, 0],
        get_text_offset=[0, 10],  # Adjust the vertical offset here
        pickable=True
    )

    r = pdk.Deck(layers=[scatter_layer_all, scatter_layer_selected], initial_view_state=view_state,
                 map_provider="mapbox", map_style=pdk.map_styles.SATELLITE)


    # Display the map
    st.pydeck_chart(r)

    # Filter X_test and y_test based on the selected location and date
    filtered_X_test_index = merged_df[
        (merged_df['Location'] == selected_location) & (merged_df['Date'] == selected_date)].index

    filtered_X_test = X_test.loc[filtered_X_test_index]
    filtered_y_test = y_test.loc[filtered_X_test_index]

    # Predict y_test for the filtered data
    y_test_pred = lgb_classifier.predict(filtered_X_test)

    # Map the prediction and actual result to "No" or "Yes"
    prediction_mapping = {0: 'No', 1: 'Yes'}
    y_test_pred_mapped = [prediction_mapping[pred] for pred in y_test_pred]
    filtered_y_test_mapped = [prediction_mapping[actual] for actual in filtered_y_test]

    # Create a DataFrame with the mapped prediction and actual values
    results_df = pd.DataFrame({'Prediction': y_test_pred_mapped, 'Actual': filtered_y_test_mapped},
                              index=filtered_X_test.index)

    # Display the prediction and actual result in a single DataFrame
    st.write(f"Prediction and Actual result for RainTomorrow at {selected_location} on {selected_date}:")
    st.dataframe(results_df, use_container_width=True)


st.markdown("Asaf Cem Akın")
st.markdown("https://www.linkedin.com/in/asafcemakin/")

