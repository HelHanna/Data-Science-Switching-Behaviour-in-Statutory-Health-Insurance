#!/usr/bin/env python3

import pathlib
import re

import pandas as pd


def sanitize_sheet_name(name):
    # Replace invalid characters with an underscore
    return re.sub(r'[\\/*?:$$$$]', '_', name)


def extract_questions_and_tables(filepath):
    # Load the file
    df = pd.read_excel(filepath, header=None)

    # Store results here
    questions_dict = {}  # key: question string, value: list of tables (as DataFrames)
    current_question = None
    current_table = []

    for idx, row in df.iterrows():
        row_str = row.dropna().astype(str).str.strip().tolist()

        # Identify question row (heuristic: short string, often 1-2 cells)
        if len(row_str) == 1 and not row_str[0].startswith(('Gesamt', 'n gesamt', 'Summe', 'Privat', 'Nicht')):
            # Save previous table if one exists
            if current_question and current_table:
                table_df = pd.DataFrame(current_table).dropna(how='all', axis=1)
                questions_dict.setdefault(current_question, []).append(table_df)
                current_table = []
            current_question = row_str[0]

        # Otherwise, treat it as part of a table
        elif current_question:
            current_table.append(row.tolist())

    # Save the final table
    if current_question and current_table:
        table_df = pd.DataFrame(current_table).dropna(how='all', axis=1)
        questions_dict.setdefault(current_question, []).append(table_df)

    return questions_dict

def write_to_excel_grouped(questions_dict, output_file):
    with pd.ExcelWriter(output_file) as writer:  # , engine='xlsxwriter'
        for i, v in enumerate(questions_dict):
            question, tables = v, questions_dict[v]

            first_two_columns = tables[0].iloc[:, :2]
            remaining_columns = pd.concat([df.iloc[:, 2:] for df in tables], axis=1)
            combined_df = pd.concat([first_two_columns, remaining_columns], axis=1)

            # Sanitize the sheet name
            sanitized_sheet_name = sanitize_sheet_name(f'{i:03} {question[:27]}')

            # Write the combined table to a new sheet
            combined_df.to_excel(writer, sheet_name=sanitized_sheet_name, index=False, header=False)



in_file = pathlib.Path('../data/Kundenmonitor_GKV_2023.xlsx')
# in_file = pathlib.Path('/dev/shm/u.xlsx')
questions_data = extract_questions_and_tables(in_file)
write_to_excel_grouped(questions_data, "processed_output.xlsx")
