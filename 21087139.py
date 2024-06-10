
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def create_line_plot(
        x_values,
        y_values,
        x_label,
        y_label,
        plot_title,
        series_labels):
    """ Function to create a line plot with multiple series.

    Args:
    x_values (list): The list of values for the x-axis.
    y_values (list of lists): A list containing multiple lists of y-values, each representing a different series.
    x_label (str): The label for the x-axis.
    y_label (str): The label for the y-axis.
    plot_title (str): The title of the plot.
    series_labels (list): A list of labels for the different series in the plot.

    This function creates a line plot where each series is plotted with a square marker and a dash-dot line style. The legend is placed outside the top left corner of the plot.
    """
    plt.style.use('tableau-colorblind10')  # Using a colorblind-friendly style.
    plt.figure(figsize=(7, 5))  # Setting the figure size.

    # Plotting each series in the y_values list.
    for index in range(len(y_values)):
        plt.plot(
            x_values,
            y_values[index],
            label=series_labels[index],
            linestyle='-.')

    # Setting the number of bins for the x-axis.
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))

    # Rotating x-axis labels for better visibility.
    plt.xticks(rotation=90)

    # Setting labels and title.
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)

    # Positioning the legend outside of the plot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Saving the plot as a high-resolution image.
    plt.savefig('Line_plot.jpg', dpi=500)

    # Displaying the plot.
    plt.show()

    return


def create_bar_plot(
        dataframe,
        x_label,
        y_label,
        plot_title,
        output_filename='barplot.jpg'):
    """ Function to create a bar plot from a pandas DataFrame.

    Args:
    dataframe (DataFrame): The pandas DataFrame containing the data to plot.
    x_label (str): The label for the x-axis.
    y_label (str): The label for the y-axis.
    plot_title (str): The title of the plot.
    output_filename (str): Filename to save the plot. Default is 'barplot.jpg'.

    This function creates a bar plot using the DataFrame's plotting capability. The plot is then customized with labels and a title.
    The plot is saved as a high-resolution image and displayed.
    """
    # Plotting the dataframe as a bar chart.
    dataframe.plot(kind='bar', figsize=(10, 6))

    # Setting the labels and title of the plot.
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)

    # Saving the plot as a high-resolution image.
    plt.savefig(output_filename, dpi=500)

    # Displaying the plot.
    plt.show()

    return


def create_correlation_heatmap(data_frame, plot_title, color_map='viridis'):
    """Generate a heatmap for the correlation matrix of a DataFrame.

    Args:
    data_frame (DataFrame): The pandas DataFrame to analyze.
    plot_title (str): The title for the heatmap.
    color_map (str): The colormap for the heatmap, defaults to 'viridis'.

    This function calculates the correlation matrix of the DataFrame and displays it as a heatmap.
    Each cell of the heatmap contains the correlation coefficient, formatted to two decimal places.
    The heatmap is then saved as a high-resolution image and displayed.
    """
    # Calculate the correlation matrix
    correlation_matrix = data_frame.corr()

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.pcolormesh(correlation_matrix, cmap=color_map)

    # Add a color bar
    plt.colorbar(heatmap)

    # Annotate the heatmap with correlation coefficients
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(
                j + 0.5,
                i + 0.5,
                f'{correlation_matrix.iloc[i, j]:.2f}',
                ha='center',
                va='center',
                color='white')

    # Set the ticks and labels
    ax.set_xticks([i + 0.5 for i in range(len(correlation_matrix.columns))])
    ax.set_yticks([i + 0.5 for i in range(len(correlation_matrix.columns))])
    ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    ax.set_yticklabels(correlation_matrix.columns)

    # Set the title and save/show the plot
    plt.title(plot_title)
    plt.savefig('Correlation_Heatmap.jpg', dpi=500)
    plt.show()

    return


def refine_data(df, countries, start_year, up_to_year):
    """Refines a DataFrame by transposing, filtering, and reformatting for specified years and countries.

    Args:
        df (DataFrame): The original DataFrame.
        countries (list): List of countries to retain in the DataFrame.
        start_year (int): The starting year for data inclusion.
        up_to_year (int): The last year for data inclusion.

    Returns:
        DataFrame: A refined DataFrame with selected countries and year range.
    """
    # Data cleaning and transposition
    df = df.drop(['Country Code', 'Indicator Name',
                 'Indicator Code', 'Unnamed: 67'], axis=1).T
    df.columns = df.iloc[0]
    df = df.drop(
        ['Country Name']).reset_index().rename(
        columns={
            'index': 'Years'})
    df['Years'] = df['Years'].astype(int)

    # Data slicing by year and countries
    df = df[(df['Years'] >= start_year) & (df['Years'] <= up_to_year)]
    selected_data = df[countries]
    selected_data = selected_data.fillna(selected_data.mean(axis=0))

    return selected_data


def get_data_for_specific_country(
        data_frame_list,
        country_name,
        names,
        start_year,
        end_year):
    """Aggregates specific country data from multiple DataFrame sources.

    Args:
        data_frame_list (list of DataFrame): The list of DataFrames to process.
        country_name (list): The list containing a single country name.
        names (list): The list of new names corresponding to each DataFrame.
        start_year (int): The starting year for data inclusion.
        end_year (int): The last year for data inclusion.

    Returns:
        DataFrame: A DataFrame containing merged and deduplicated data.
    """
    country_data = []
    for i, data in enumerate(data_frame_list):
        refined_data = refine_data(data, country_name, start_year, end_year)
        refined_data.rename(columns={country_name[0]: names[i]}, inplace=True)
        country_data.append(refined_data)

    country_data = pd.concat(country_data, axis=1)
    country_data = country_data.loc[:, ~country_data.columns.duplicated()].drop(
        'Years', axis=1)

    return country_data


def data_for_bar(df, years):
    """Filters a DataFrame for specific years and sets 'Years' as the index.

    Args:
        df (DataFrame): The DataFrame to filter.
        years (list): List of years to include.

    Returns:
        DataFrame: A DataFrame indexed by 'Years' and filtered by the specified years.
    """
    df = df[df['Years'].isin(years)]
    df.set_index('Years', inplace=True)
    return df


def get_data_description(
        data_frame_list,
        country_names,
        names,
        start_year,
        end_year):
    """Compiles specific country data across multiple DataFrames for a given period and renames the columns.

    Args:
        data_frame_list (list of DataFrame): The list of DataFrames to process.
        country_names (list): List containing the names of the countries to filter the data for.
        names (list): List of new column names for the data corresponding to each DataFrame.
        start_year (int): The starting year for data inclusion.
        end_year (int): The last year for data inclusion.

    Returns:
        DataFrame: A DataFrame indexed by 'Years' with merged, deduplicated, and renamed data for the specified country.
    """
    country_data = []
    # Process each DataFrame in the list
    for i, data in enumerate(data_frame_list):
        refined_data = refine_data(data, country_names, start_year, end_year)
        refined_data.rename(columns={country_names[0]: names[i]}, inplace=True)
        country_data.append(refined_data)

    # Concatenate all DataFrames horizontally, drop duplicated columns if any
    country_data = pd.concat(country_data, axis=1)
    country_data = country_data.loc[:, ~country_data.columns.duplicated()]
    country_data.set_index('Years', inplace=True)

    return country_data


def extract_column_lists(df, columns):
    """Extracts lists from specified DataFrame columns, excluding the last specified column.

    Args:
        df (DataFrame): The pandas DataFrame from which to extract the data.
        columns (list of str): List of column names from which to extract lists.

    Returns:
        list of list: A list containing lists of data from each specified DataFrame column,
                      excluding the last one in the provided list.

    Raises:
        KeyError: If any column specified in 'columns' does not exist in 'df'.
        ValueError: If 'columns' is empty or only contains one column (since the last column is excluded).
    """
    if not columns:
        raise ValueError(
            "The 'columns' list must contain at least two column names.")

    # Check if all specified columns exist in the DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            f"The following columns are missing in the DataFrame: {missing_columns}")

    # Extract lists for all but the last specified column
    column_lists = [df[col].tolist() for col in columns[:-1]]

    return column_lists


# Load the datasets
Forest_area = pd.read_csv('Forest_area.csv', skiprows=4)
Arable_land = pd.read_csv('Arable_land.csv', skiprows=4)
Forest_area.head()
Manufacturing_value_added_USD = pd.read_csv(
    'Manufacturing_value_added_USD.csv',
    skiprows=4)  # Assuming this CSV is correctly named and exists
# Assuming this CSV is correctly named and exists
CO2_emissions = pd.read_csv('CO2_emissions.csv', skiprows=4)

# Define the columns and years
cols = ['Pakistan', 'Nepal', 'Sri Lanka', 'Afghanistan', 'Bangladesh', 'Years']
start_year = 1980  # Assuming the year should be int type
end_year = 2021

# Data processing and visualization for Forest Area
refined_forest_data = refine_data(Forest_area, cols, start_year, end_year)
years = refined_forest_data['Years'].tolist()
forest_lists = extract_column_lists(refined_forest_data, cols)
create_line_plot(years,
                 forest_lists,
                 'Years',
                 'Percentage of Land',
                 'Forest Area Coverage',
                 cols[:-1])

# Data processing and visualization for Arable Land
refined_arable_data = refine_data(Arable_land, cols, start_year, end_year)
arable_lists = extract_column_lists(refined_arable_data, cols)
create_line_plot(years,
                 arable_lists,
                 'Years',
                 'Percentage of Land',
                 'Arable Land Coverage',
                 cols[:-1])

# Load the datasets
Manufacturing_value_added_USD = pd.read_csv(
    'Manufacturing_value_added_USD.csv', skiprows=4)
CO2_emissions = pd.read_csv('CO2_emissions.csv', skiprows=4)

# Define the columns and years
cols = ['Pakistan', 'Nepal', 'Sri Lanka', 'Afghanistan', 'Bangladesh', 'Years']
start_year = 1990  # Adjusted for integer type
end_year = 2021

# Define years of interest for the bar plots
years_of_interest = [1995, 2000, 2005, 2010,
                     2015, 2020]  # Adjusted for integer type

# Process and visualize manufacturing data
manufacturing_data = refine_data(
    Manufacturing_value_added_USD,
    cols,
    start_year,
    end_year)
manufacturing_data_for_plot = data_for_bar(
    manufacturing_data, years_of_interest)
create_bar_plot(manufacturing_data_for_plot, 'Years',
                'USD', 'Manufacturing Value Added (USD)')

# Process and visualize CO2 emissions data
co2_emissions_data = refine_data(CO2_emissions, cols, start_year, end_year)
co2_emissions_data_for_plot = data_for_bar(
    co2_emissions_data, years_of_interest)
create_bar_plot(
    co2_emissions_data_for_plot,
    'Years',
    'CO2 Emissions (kt)',
    'CO2 Emissions')

Energy_use = pd.read_csv('Energy_use.csv', skiprows=4)
Electric_power_consumption = pd.read_csv(
    'Electric_power_consumption.csv', skiprows=4)
Agricultural_land = pd.read_csv('Agricultural_land.csv', skiprows=4)
Urban_population = pd.read_csv('Urban_population.csv', skiprows=4)
names = [
    'Forest_area',
    'Urban_population',
    'Manufacturing_GDP',
    'CO2_emissions',
    'Arable_land',
    'Electric_power_consumption',
    'Energy_use']
data_frames = [
    Forest_area,
    Urban_population,
    Manufacturing_value_added_USD,
    CO2_emissions,
    Arable_land,
    Electric_power_consumption,
    Energy_use]
country_name = ['Pakistan', 'Years']
create_correlation_heatmap(
    get_data_for_specific_country(
        data_frames,
        country_name,
        names,
        1990,
        2020),
    'Pakistan',
    'twilight_shifted')
country_name = ['Bangladesh', 'Years']
create_correlation_heatmap(
    get_data_for_specific_country(
        data_frames,
        country_name,
        names,
        1990,
        2020),
    'Bangladesh',
    'Dark2')
country_name = ['Nepal', 'Years']
create_correlation_heatmap(
    get_data_for_specific_country(
        data_frames,
        country_name,
        names,
        1990,
        2020),
    'Nepal',
    'brg')

country_name = ['Pakistan', 'Years']
get_data_description(data_frames, country_name, names, 1990, 2020).describe()
