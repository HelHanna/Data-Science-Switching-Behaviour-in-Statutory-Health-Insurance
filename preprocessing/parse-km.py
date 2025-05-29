#!/usr/bin/env python3

import pathlib
from collections import Counter
from io import StringIO

import pandas as pd


def convert():
    df = pd.read_excel(in_file, sheet_name=0, header=None)
    df.to_csv(tmp_file, index=False, header=False)


def count_lines():
    lines = tmp_file.read_text().splitlines()
    line_counts = Counter(lines)
    for line, count in line_counts.items():
        print(f"{count:3}: {line.strip()}\n")


def compress_data():
    lines = tmp_file.read_text().splitlines()
    name = []
    data = []
    data_df = []
    state = 0
    doubleel = 0
    contd = False
    res = []
    for line in lines:
        if line == ',,,,,,,,,,,,,,,,,,':
            if data:
                df = pd.read_csv(StringIO("\n".join(data)), header=None)
                data = []
                # a = pd.DataFrame(data)
                if data_df:
                    df = df.iloc[:, 2:]
                df = df.dropna(axis=1, how='all')
                data_df.append(df)
            doubleel += 1
            if doubleel > 1:
                state = 0
            continue
        doubleel = 0
        if state == 0:
            line = line.strip(',')
            if len(name) == 0 or line not in name:
                if name and data_df:
                    res.append((name, data_df))
                else:
                    print(f'warn {name}')
                name = [line]
                data_df = []
                contd = False
            else:
                contd = True
            state += 1
            continue
        elif state == 1:
            if contd == False:
                line = line.strip(',')
                name.append(line)
            state += 1
            continue
        else:
            state += 1
            data.append(line)
    res.append((name, data_df))
    ret = []
    for i, (questions, dataframes) in enumerate(res):
        csv_file_path = out_file.with_name(f'out_{i + 1:03}.csv')
        ret.append('\n'.join([csv_file_path.name, *questions]))
        # csv_file_path.write_text('\n'.join(table_df))
        complete = pd.concat(dataframes, axis=1)
        complete.to_csv(csv_file_path, index=False, header=False)
        print(f"Table {questions} has been saved to {csv_file_path}")
    out_file.write_text('\n\n'.join(ret))


in_file = pathlib.Path('../data/Kundenmonitor_GKV_2023.xlsx')
tmp_file = pathlib.Path('tmp.csv')
out_file = pathlib.Path('./tmp-km23/out.txt')
out_file.parent.mkdir(parents=True, exist_ok=True)

convert()
count_lines()
compress_data()
