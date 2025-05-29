import pandas as pd

# Load sheets
questions_df = pd.read_excel("C:/Users/hanna/OneDrive - Universität des Saarlandes/Dokumente/Semester_10/data science/Topic 2/230807_Survey.xlsx", sheet_name=0)
answers_df = pd.read_excel("C:/Users/hanna/OneDrive - Universität des Saarlandes/Dokumente/Semester_10/data science/Topic 2/230807_Survey.xlsx", sheet_name=1)

# Forward fill to propagate Question, Type, Name to all rows
questions_df["Question"] = questions_df["Question"].fillna(method="ffill")
questions_df["Type"] = questions_df["Type"].fillna(method="ffill")
questions_df["Name"] = questions_df["Name"].fillna(method="ffill")

# Build mapping: Question -> {value -> label} for answer labels
question_map = {}
for _, row in questions_df.iterrows():
    qid = row["Question"]
    val = row.get("Value")
    label = row.get("Label")
    if pd.notnull(val) and pd.notnull(label):
        question_map.setdefault(qid, {})[str(val).strip()] = str(label).strip()

# Extract question labels from rows where Value is NaN (these are the actual question texts)
question_labels = questions_df[questions_df["Value"].isna()].set_index("Question")["Label"].to_dict()

# Fallback: if Label is missing, use the Name column
for qid in questions_df["Question"].unique():
    if qid not in question_labels or pd.isna(question_labels[qid]):
        question_labels[qid] = questions_df.loc[questions_df["Question"] == qid, "Name"].iloc[0]

# Build question types dictionary (first per question)
question_types = questions_df.groupby("Question")["Type"].first().to_dict()

# Build full question labels/types dict (including subquestions)
full_question_types = questions_df.set_index("Question")["Type"].dropna().to_dict()
full_question_labels = questions_df.set_index("Question")["Label"].dropna().to_dict()

# Build subquestion_texts dictionary for subquestions (IDs containing a dot)
subquestion_texts = questions_df[questions_df["Question"].str.contains(r"\.")].set_index("Question")["Label"].to_dict()

def generate_prompt(row):
    parts = ["Participant Summary:"]
    multi_select_answers = {}
    grouped_subquestions = {}

    for col in row.index:
        if not col.startswith("Q"):
            continue

        val = row[col]
        if pd.isnull(val):
            continue

        qtype = full_question_types.get(col, question_types.get(col, ""))
        qname = question_labels.get(col, col)
        main_qid = col.split(".")[0]

        # search for Multiple Select columns
        if qtype == "Multiple Select":  
            if int(val) == 1:
                label = question_map.get(col, {}).get("1", full_question_labels.get(col, qname))
                multi_select_answers.setdefault(main_qid, {"name": question_labels.get(main_qid, main_qid), "labels": []})
                multi_select_answers[main_qid]["labels"].append(label)

        # group subquestions together
        elif "." in col:
            sub_label = subquestion_texts.get(col, full_question_labels.get(col, qname))

            if qtype in ["Single Select", "Dropdown", "Rating Scale", "Single Matrix", "Rating Matrix"]:
                answer_label = question_map.get(col, {}).get(str(int(val)), f"(Unknown code {val})")
            else:
                answer_label = val

            if main_qid not in grouped_subquestions:
                grouped_subquestions[main_qid] = {
                    "main_label": question_labels.get(main_qid, main_qid),
                    "answers": []
                }
            grouped_subquestions[main_qid]["answers"].append((col, sub_label, answer_label))

        # questions without subquestions
        else:
            prefix = f"{col} ({qname})"
            if qtype in ["Single Select", "Dropdown"]:
                label = question_map.get(col, {}).get(str(int(val)), f"(Unknown code {val})")
                parts.append(f"- {prefix}: {label}")
            elif qtype in ["Number Input", "Net Promoter Score®"]:
                parts.append(f"- {prefix}: {val}")
            elif qtype == "Open Text":
                parts.append(f"- {prefix}: \"{val}\"")
            elif qtype == "Rating Scale":
                label = question_map.get(col, {}).get(str(int(val)), f"(Unknown rating {val})")
                parts.append(f"- {prefix}: {label}")
            elif qtype in ["Single Matrix", "Rating Matrix"]:
                label = question_map.get(col, {}).get(str(int(val)), f"(Unknown response {val})")
                parts.append(f"- {prefix}: {label}")
            elif qtype == "Ranking":
                try:
                    rank_num = int(val)
                    parts.append(f"- {prefix}: Rank {rank_num}")
                except ValueError:
                    parts.append(f"- {prefix}: (invalid rank)")
            else:
                parts.append(f"- {prefix}: {val}")

    for qid, info in multi_select_answers.items():
        labels_joined = ", ".join(info["labels"])
        parts.append(f"- {qid} ({info['name']}): {labels_joined}")

    for qid, info in grouped_subquestions.items():
        parts.append(f"{qid} ({info['main_label']}):")
        for col, sub_label, answer_label in info["answers"]:
            parts.append(f"  - {col} ({sub_label}): {answer_label}")

    parts.append("Answer summary complete.")
    return "\n".join(parts)


# Apply the function to each row (participant)
answers_df["prompt"] = answers_df.apply(generate_prompt, axis=1)

# Save output
answers_df[["pid", "prompt"]].to_csv("participant_prompts.csv", index=False)

print("Prompts generated and saved to 'participant_prompts.csv'")
