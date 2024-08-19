import os
import pandas as pd

import re


def combine_csv_results(csv_folder_path: str, output_file_path: str) -> None:
    assert os.path.exists(csv_folder_path)

    csv_file_names = []
    for file_name in os.listdir(csv_folder_path):
        csv_file_names += [file_name]
    csv_file_names.sort(key=natural_keys)

    combined_results = pd.DataFrame()

    for file_name in csv_file_names:
        df_temp = pd.read_csv(os.path.join(csv_folder_path, file_name))
        combined_results = pd.concat([combined_results, df_temp], ignore_index=True)

    combined_results = combined_results.drop(columns="arena_name")

    combined_results.to_csv(output_file_path)


# from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text


# from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == "__main__":
    combine_csv_results(csv_folder_path="results/foragingTask/ppo",
                        output_file_path="results/foragingTask/ppo/all.csv")

    combine_csv_results(csv_folder_path="results/foragingTask/rppo",
                        output_file_path="results/foragingTask/rppo/all.csv")
