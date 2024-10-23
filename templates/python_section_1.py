from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    
    for i in range(0, len(lst), n):
        group = lst[i:i+n]  
        result.extend(group[::-1])  
    
    return result
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3)) 
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))           
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  



from typing import List, Dict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    return dict(sorted(length_dict.items()))

print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
print(group_by_length(["one", "two", "three", "four"]))


from typing import Dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened = {}

    def _flatten(current_dict, parent_key=""):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                _flatten(value, new_key)  
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        _flatten(item, f"{new_key}[{i}]") 
                    else:
                        flattened[f"{new_key}[{i}]"] = item
            else:
                flattened[new_key] = value

    _flatten(nested_dict)
    return flattened

nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

result = flatten_dict(nested_dict)
print(result)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    result = list(set(permutations(nums)))
    pass
    return result

input_list = [1, 1, 2]
result = unique_permutations(input_list)
for perm in result:
    print(list(perm))
    


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
     patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{4}\.\d{2}\.\d{2}\b'  
    ]
    
    combined_pattern = '|'.join(patterns)
    dates = re.findall(combined_pattern, text)
    
    return dates

    pass  

if __name__ == "__main__":
    text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
    output = find_all_dates(text)
    print(output)
    


def haversine(lat1, lon1, lat2, lon2):
    
    R = 6371000  
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    distances = [0]  
    for i in range(1, len(df)):
        dist = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                         df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(dist)  

    df['distance'] = distances  
    
    return df
if __name__ == "__main__":
    
    polyline_str = "u{~vHk`hPj@k@wD~Z|Y~@d@jA"  
    df = polyline_to_dataframe(polyline_str)
    print(df)


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
     n = len(matrix)
    return [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    rotated_matrix = rotate_matrix(matrix)
    transformed_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            original_row = j
            original_col = n - 1 - i
            multiplication_factor = original_row + original_col
            transformed_matrix[i][j] = rotated_matrix[i][j] * multiplication_factor
            
    return transformed_matrix
if __name__ == "__main__":
    matrix = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]
    final_matrix = rotate_and_multiply_matrix(matrix)
    print(final_matrix)

    


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
      days_of_week = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                    'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    df['startDay_num'] = df['startDay'].map(days_of_week)
    df['endDay_num'] = df['endDay'].map(days_of_week)

    grouped = df.groupby(['id', 'id_2'])
    coverage_check = []

    for (id_val, id_2_val), group in grouped:
        week_coverage = {i: [] for i in range(7)}
        
        for _, row in group.iterrows():
            start_day = row['startDay_num']
            end_day = row['endDay_num']
            start_time = row['startTime']
            end_time = row['endTime']

            current_day = start_day
            while current_day != end_day:
                week_coverage[current_day].append((start_time, '23:59:59'))
                current_day = (current_day + 1) % 7
                start_time = '00:00:00'
                
            week_coverage[end_day].append((start_time, end_time))

        full_week = True
        for day, times in week_coverage.items():
            if len(times) == 0:  
                full_week = False
                break
            times = sorted(times)
            if times[0][0] != '00:00:00' or times[-1][1] != '23:59:59':
                full_week = False
                break
            for i in range(1, len(times)):
                if times[i-1][1] != times[i][0]:
                    full_week = False
                    break
            if not full_week:
                break
        
        coverage_check.append(((id_val, id_2_val), not full_week))
    
    index = pd.MultiIndex.from_tuples([x[0] for x in coverage_check], names=['id', 'id_2'])
    return pd.Series([x[1] for x in coverage_check], index=index)

    return pd.Series()
