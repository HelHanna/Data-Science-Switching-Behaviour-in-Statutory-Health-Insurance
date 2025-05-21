import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

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
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    import pandas as pd

    wb = Workbook()
    wb.remove(wb.active)  # remove default sheet

    for question, tables in questions_dict.items():
        ws = wb.create_sheet(title=question[:31])  # Excel sheet name max 31 chars
        ws.append([question])
        ws.append([])

        combined_tables = []

        for i, df in enumerate(tables):
            df = df.copy()

            df.columns = [str(col).strip() for col in df.columns]

            if i == 0:
                # Keep full table including first two columns
                combined_tables.append(df.reset_index(drop=True))
            else:
                # Remove the first two columns for all following subtables
                df_trimmed = df.iloc[:, 2:].reset_index(drop=True)
                combined_tables.append(df_trimmed)


        # Horizontales Zusammenfügen der bereinigten Tabellen
        combined = pd.concat(combined_tables, axis=1)

        # Schreibe die Daten ins Excel-Blatt
        for row in dataframe_to_rows(combined, index=False, header=False):
            ws.append(row)

    wb.save(output_file)



questions_data = extract_questions_and_tables("C:/Users/hanna/OneDrive - Universität des Saarlandes/Dokumente/Semester_10/data science/mini_test_data.xlsx"
)
write_to_excel_grouped(questions_data, "processed_output.xlsx")
