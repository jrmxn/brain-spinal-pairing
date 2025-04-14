import re
import pandas as pd
from pathlib import Path
from config import CONFIG as cfg


def parse_file(log_path, pattern_):
    """
    Parse the log file by grouping lines for each participant's session.
    Within each block capture:
        EPs at start: events = <number>
        EPs at start (SCAP): events = <number>
        EPs at start (REC): events = <number>

        EPs at end: events = <number>
        EPs at end (SCAP): events = <number>
        EPs at end (REC): events = <number>

    The block ends either at the next 'Session:' line or end of file.
    """

    # Patterns
    session_pattern = re.compile(r"Session:\s+(\S+)\s+V(\d+)")

    # We'll parse each block of lines belonging to a single session.
    def parse_session_block(block_lines, pattern):
        participant = ""
        visit = ""
        all_start = 0
        scap_start = 0
        rec_start = 0
        all_end = 0
        scap_end = 0
        rec_end = 0

        # The first line in the block should match 'Session:'
        if block_lines:
            match_session = session_pattern.search(block_lines[0])
            if match_session:
                participant = match_session.group(1)
                visit = match_session.group(2)

        for line in block_lines:
            line_str = line.strip()
            match = pattern.search(line_str)
            if not match:
                continue

            events_val = float(match.group(1))

            if "EPs at start (ALL)" in line_str and "(SCAP)" not in line_str and "(REC)" not in line_str:
                all_start = events_val
            elif "EPs at start (SCAP)" in line_str:
                scap_start = events_val
            elif "EPs at start (REC)" in line_str:
                rec_start = events_val
            elif "EPs at end (ALL)" in line_str and "(SCAP)" not in line_str and "(REC)" not in line_str:
                all_end = events_val
            elif "EPs at end (SCAP)" in line_str:
                scap_end = events_val
            elif "EPs at end (REC)" in line_str:
                rec_end = events_val

        return (
            participant,
            visit,
            all_start,
            scap_start,
            rec_start,
            all_end,
            scap_end,
            rec_end,
        )

    results = []
    assert log_path.exists(), "Incorrect path?"

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_block = []

    for line in lines:
        if session_pattern.search(line):
            # If there's a previous block, parse it first
            if current_block:
                parsed = parse_session_block(current_block, pattern_)
                # If there's a participant name, we append
                if parsed[0]:
                    results.append(parsed)
                current_block = [line]
            else:
                current_block = [line]
        else:
            if current_block:
                current_block.append(line)
            else:
                # ignore lines before first session
                continue

    # Parse the final block
    if current_block:
        parsed = parse_session_block(current_block, pattern_)
        if parsed[0]:
            results.append(parsed)

    return results


def parse_final_lines(filepath, group, key):
    """
    Parses the log file at `filepath` and extracts the value for the specified `group` and `key`.
    """
    # Ensure filepath is a Path object.
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File {filepath} not found.")

    # Compile a regex to match the line with the requested group and key.
    # Example matching line:
    # "2025-04-02 13:08:18,242 - INFO - Final count is (SCAP): ... | total = 33004 | ..."
    pattern = re.compile(rf"Final count is \({group}\):.*?\|\s*{key}\s*=\s*([\d\.]+)")

    found_value = None
    with file_path.open("r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                found_value = match.group(1)
                # Continue iterating to get the last occurrence in the file.

    if found_value is None:
        raise ValueError(f"No match found for group '{group}' and key '{key}' in file {filepath}")

    # Return the appropriate type based on the key.
    if key == "fr.muscles":
        return float(found_value)
    else:
        return int(found_value)


def main(input_log, target):
    print('----------')
    print(target)
    print('----------')
    events_pattern = re.compile(rf"{re.escape(target)}\s*=\s*(\d+(?:\.\d+)?)")
    parsed_data = parse_file(input_log, events_pattern)
    output_csv = input_log.with_suffix('.csv')
    output_csv_full = output_csv.with_stem(f'summary_{input_log.stem}_{target}')
    output_csv_merged = output_csv.with_stem(f'summary_{input_log.stem}_{target}_merged')

    # Create a pandas DataFrame
    columns = [
        "Participant",
        "Visit",
        "ALLEvents_start",
        "SCAPEvents_start",
        "RECEvents_start",
        "ALLEvents_end",
        "SCAPEvents_end",
        "RECEvents_end",
    ]
    df = pd.DataFrame(parsed_data, columns=columns)
    df = df.sort_values('Participant')
    df.to_csv(output_csv_full, index=False)  # n.b. after combining visits

    if target == 'fr.muscles':
        dp_p = df.groupby("Participant").mean(numeric_only=True)
    else:
        dp_p = df.groupby("Participant").sum(numeric_only=True)

    dp_p = dp_p.sort_index()

    dp_p.to_csv(output_csv_merged)  # n.b. after combining visits

    # Summaries for 'start' columns
    dict_show = {
        "ALLEvents_start": "\n=== ALL EPs at start (All Visits) ===",
        "SCAPEvents_start": "\n=== SCAP EPs at start (All Visits) ===",
        "RECEvents_start": "\n=== REC EPs at start (All Visits) ===",
        "ALLEvents_end": "\n=== ALL EPs at end (All Visits) ===",
        "SCAPEvents_end": "\n=== SCAP EPs at end (All Visits) ===",
        "RECEvents_end": "\n=== REC EPs at end (All Visits) ===",
    }

    for k, v in dict_show.items():
        f_mea = dp_p[k].mean()
        f_std = dp_p[k].std()
        f_sum = dp_p[k].sum()
        print(v)
        print(f"Mean = {f_mea:.3f}, StdDev = {f_std:.3f} (total = {f_sum:.0f})")

    if target == "events":
        print('=============')
        percentage_reduced = (1 - dp_p['SCAPEvents_end'].sum()/dp_p['SCAPEvents_start'].sum()) * 100
        print(f'This resulted in a reduction of {percentage_reduced:0.1f}% of the total stimulation events used for analysis of paired stimulation data')
        print('=============')

    if target == "total":
        scap_total_final_after_manual_rej = parse_final_lines(input_log, "SCAP", "total")
        final_count_scap = 100 - (100 * (1-scap_total_final_after_manual_rej/dp_p['SCAPEvents_end'].sum()))
        print('=============')
        print(f"Manual rejection step keeping: {final_count_scap:0.3f}%")
        print('=============')



if __name__ == "__main__":
    '''This can be used to look at the files made by just_load.py'''
    # es = 'filtered_justload_ECR_FCR_APB_ADM_FDI_notieonly'
    es = 'filtered_justload_auc_target_notieonly'
    f_data = cfg['DATA_FOLDER']['noninvasive'].parts[-1]
    input_log = Path(f"/mnt/hdd3/va_data_temp/proc/preproc_tables/reproc/{es}.log")
    vec_target = ['events', 'fr.muscles', 'total']

    for target in vec_target:
        main(input_log, target)
