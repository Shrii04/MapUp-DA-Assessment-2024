import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    def get_discount_factor(day, time_val):
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        weekends = ['Saturday', 'Sunday']
        
        if day in weekdays:
            if time(0, 0, 0) <= time_val <= time(10, 0, 0):
                return 0.8
            elif time(10, 0, 0) < time_val <= time(18, 0, 0):
                return 1.2
            else:
                return 0.8
        elif day in weekends:
            return 0.7
        return 1.0
    
    for index, row in df.iterrows():
        start_time = time.fromisoformat(row['startTime'])
        end_time = time.fromisoformat(row['endTime'])
        
        start_factor = get_discount_factor(row['startDay'], start_time)
        end_factor = get_discount_factor(row['endDay'], end_time)
        
        toll_columns = ['able2Hov2', 'able2Hov3', 'able3Hov2', 'able3Hov3', 'able5Hov2', 'able5Hov3', 'able4Hov2', 'able4Hov3']
        for col in toll_columns:
            if pd.notnull(row[col]) and row[col] != -1:
                df.at[index, col] *= (start_factor + end_factor) / 2
    
    df['start_day'] = df['startDay']
    df['start_time'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
    df['end_day'] = df['endDay']
    df['end_time'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time
    
    return df


df = pd.read_csv('C:\\Users\\adars\\OneDrive\\Attachments\\Desktop\\dataset-1 (1).csv')

df_with_toll_rates = calculate_time_based_toll_rates(df)

df_with_toll_rates.head()  

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
     toll_locations = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    print("Toll locations:", toll_locations)  
    
    distance_matrix = pd.DataFrame(float('inf'), index=toll_locations, columns=toll_locations)
    
    for location in toll_locations:
        distance_matrix.loc[location, location] = 0
    
    print("Initialized distance matrix:")
    print(distance_matrix)
    
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        distance_matrix.loc[id_start, id_end] = distance
        distance_matrix.loc[id_end, id_start] = distance  
    
    print("Distance matrix after filling known distances:")
    print(distance_matrix)
    
    for k in toll_locations:
        for i in toll_locations:
            for j in toll_locations:
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j], distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    
    print("Final distance matrix:")
    print(distance_matrix)
    
    return distance_matrix

df = pd.read_csv(""C:\Users\shri0009\Downloads\dataset-2.csv"")
result = calculate_distance_matrix(df)

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
     reference_rows = df[df['id_start'] == reference_id]
    
\    if reference_rows.empty:
        print(f"No data found for reference ID {reference_id}.")
        return pd.DataFrame(columns=['id_start'])

    reference_avg_distance = reference_rows['distance'].mean()
    
    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1
    
    grouped = df.groupby('id_start')['distance'].mean().reset_index()
    filtered_ids = grouped[(grouped['distance'] >= lower_threshold) & (grouped['distance'] <= upper_threshold)]
    
    filtered_ids = filtered_ids.sort_values(by='id_start')
    
    return filtered_ids[['id_start']]

data = {
    'id_start': [1, 1, 2, 2, 3, 3],
    'distance': [100, 120, 110, 115, 130, 140]
}
df = pd.DataFrame(data)

reference_id = 1
result = find_ids_within_ten_percentage_threshold(df, reference_id)

print(result)



def calculate_toll_rate(df)->pd.DataFrame():
    
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    data = {'id_start': [1, 2, 3],
        'distance': [100, 150, 200]}
df = pd.DataFrame(data)

def calculate_toll_rate(df) -> pd.DataFrame:
    """
    Calculate toll rates for different vehicle types based on distance.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    df['moto'] = df['distance'] * rates['moto']
    df['car'] = df['distance'] * rates['car']
    df['rv'] = df['distance'] * rates['rv']
    df['bus'] = df['distance'] * rates['bus']
    df['truck'] = df['distance'] * rates['truck']
    
    return df
df_with_toll_rates = calculate_toll_rate(df)
df_with_toll_rates

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
def get_discount_factor(day, time_val):
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        weekends = ['Saturday', 'Sunday']
        
        if day in weekdays:
            if time(0, 0, 0) <= time_val <= time(10, 0, 0):
                return 0.8
            elif time(10, 0, 0) < time_val <= time(18, 0, 0):
                return 1.2
            else:
                return 0.8
        elif day in weekends:
            return 0.7
        return 1.0
    
    for index, row in df.iterrows():
        start_time = time.fromisoformat(row['startTime'])
        end_time = time.fromisoformat(row['endTime'])
        
        start_factor = get_discount_factor(row['startDay'], start_time)
        end_factor = get_discount_factor(row['endDay'], end_time)
        
        toll_columns = ['able2Hov2', 'able2Hov3', 'able3Hov2', 'able3Hov3', 'able5Hov2', 'able5Hov3', 'able4Hov2', 'able4Hov3']
        for col in toll_columns:
            if pd.notnull(row[col]) and row[col] != -1:
                df.at[index, col] *= (start_factor + end_factor) / 2
    
    df['start_day'] = df['startDay']
    df['start_time'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
    df['end_day'] = df['endDay']
    df['end_time'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time
    
    return df


df = pd.read_csv('C:\Users\shri0009\Downloads\dataset-1 (1).csv')

df_with_toll_rates = calculate_time_based_toll_rates(df)

df_with_toll_rates.head()  

