import pandas as pd
import toml
from pathlib import Path
import numpy as np
import h5py
from src.bsp.core.utils import bidir_dict
import logging

DEBUGGING = False


def download_noninvasive_data(cfg):
    print(f"The non-invasive data folder does not exist:\n{cfg['DATA_FOLDER']['noninvasive']}")
    print("Data can auto-download from zenodo to that location. If that is not where you want the data, edit DATAFOLDER['noninvasive'] in config.py before fetching.")
    get_data = input("Fetch data automatically? (y/n)").lower() == 'y'
    if get_data:
        from zenodo_get import zenodo_get
        import zipfile
        print('OK, getting data...')
        doi = '10.5281/zenodo.15225065'
        zenodo_get(['--doi', doi, '--output-dir', str(cfg['DATA_FOLDER']['noninvasive'])])
        p_zip = [x for x in cfg['DATA_FOLDER']['noninvasive'].glob('*.zip')][0]
        print('Unzipping...')
        with zipfile.ZipFile(p_zip, 'r') as zip_ref:
            zip_ref.extractall(p_zip.parent)
        print('Cleaning up. And continuing with analysis.')
        p_zip.unlink()


def intensity_mapping_inverted(x):
    return {label: idx for idx, label in enumerate(x)}


def classify_row(row, cx_string, sc_string):
    cx_int_pct = row[cx_string]
    sc_int_pct = row[sc_string]

    op = np.nan
    if cx_int_pct > 100:
        if sc_int_pct > 100:
            op = 'supra-supra'
        elif 0 < sc_int_pct < 100:
            op = 'supra-sub'
        elif sc_int_pct == 0:
            op = 'supra-zero'
    elif 0 < cx_int_pct < 100:
        if sc_int_pct > 100:
            op = 'sub-supra'
        elif 0 < sc_int_pct < 100:
            op = 'sub-sub'
        elif sc_int_pct == 0:
            op = 'sub-zero'
    elif cx_int_pct == 0:
        if sc_int_pct > 100:
            op = 'zero-supra'
        elif 0 < sc_int_pct < 100:
            op = 'zero-sub'

    # print(f'cx:{cx_int_pct}, sc:{sc_int_pct} --> {op}')
    return op


def make_logger(p_log, console_output=True):
    print(p_log)
    logging.basicConfig(
        filename=p_log,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def count_summary_to_log(df_chunk, stage, muscles_final, scap_mask=None, rec_mask=None):
    df_fin = np.isfinite(df_chunk.loc[:, muscles_final])
    counts = np.sum(df_fin, axis=0)
    fr_muscles = np.mean(np.mean(df_fin, axis=1), axis=0)
    logging.info(f"{stage} (ALL): {', '.join(f'{muscle} {count}' for muscle, count in counts.items())} | total = {np.sum(counts)} | events = {df_fin.shape[0]} | fr.muscles = {fr_muscles}")
    if scap_mask is not None:
        df_fin = np.isfinite(df_chunk.loc[scap_mask, muscles_final])
        counts = np.sum(df_fin, axis=0)
        fr_muscles = np.mean(np.mean(df_fin, axis=1), axis=0)
        logging.info(f"{stage} (SCAP): {', '.join(f'{muscle} {count}' for muscle, count in counts.items())} | total = {np.sum(counts)} | events = {df_fin.shape[0]} | fr.muscles = {fr_muscles}")
    if rec_mask is not None:
        df_fin = np.isfinite(df_chunk.loc[rec_mask, muscles_final])
        counts = np.sum(df_fin, axis=0)
        fr_muscles = np.mean(np.mean(df_fin, axis=1), axis=0)
        logging.info(f"{stage} (REC): {', '.join(f'{muscle} {count}' for muscle, count in counts.items())} | total = {np.sum(counts)} | events = {df_fin.shape[0]} | fr.muscles = {fr_muscles}")
    logging.info('\n')


def safe_nanmean(series):
    # just to suppress a nanmean warning when the entire column is nan
    if series.isna().all():
        return np.nan  # Return NaN if all values are NaN
    else:
        return np.nanmean(series)


def marge_and_calculate_convergence(cfg, df, df_latency_participant, columns_of_interest, emg_channels, emg_channels_unsided, adjust_for_spi):
    # Create unique groups where columns of interest are the same
    unique_combinations_X = df[columns_of_interest].drop_duplicates()
    matching_set_groups = []
    for _, combination in unique_combinations_X.iterrows():
        mask = (df[columns_of_interest] == combination).all(axis=1)
        matching_set_groups.append(df.loc[mask, 'set_group'].unique())

    # Replace underscores in group names with hyphens
    df_latency_participant.loc[:, 'group_name'] = df_latency_participant['group_name'].str.replace('_', '-')

    # Calculate averages for each matching set group
    averaged_results = []
    for group in matching_set_groups:
        mask = df_latency_participant['group_name'].isin(group)
        numeric_columns = df_latency_participant.select_dtypes(include=[np.number]).columns
        non_numeric_columns = df_latency_participant.select_dtypes(exclude=[np.number]).columns
        numeric_averages = df_latency_participant.loc[mask, numeric_columns].apply(safe_nanmean)
        non_numeric_values = df_latency_participant.loc[mask, non_numeric_columns].iloc[0]
        combined_result = pd.concat([numeric_averages, non_numeric_values])
        for i in range(sum(mask)):
            combined_result['group_name'] = group[i]
            averaged_results.append(combined_result.copy())

    df_latency_participant_averaged = pd.DataFrame(averaged_results)

    # Create the convergence latency vector sided
    columns_to_merge = [f'convergence_latency_{i + 1}' for i in range(len(emg_channels))]

    df_latency_participant_averaged['convergence_latency_vector_sided'] = df_latency_participant_averaged[
            columns_to_merge].apply(lambda row: np.array(row), axis=1)

    # Extract the correct side of the muscles
    df_latency_participant_averaged['convergence_latency'] = df_latency_participant_averaged.apply(
        lambda row: row['convergence_latency_vector_sided'][:len(emg_channels_unsided)] if row[
                                                                                               'main_targeted_side'] == 'L'
        else row['convergence_latency_vector_sided'][len(emg_channels_unsided):],
        axis=1
    )

    vec_response = cfg['DATA_OPTIONS']['response'].copy()
    if vec_response == ['auc_target']:
        # if target is selected, then the set is reduced in the loop later -
        # vec_response = emg_channels_unsided
        vec_response = ['ECR', 'FCR', 'APB', 'ADM', 'FDI']

    # Step 6: Get only for the 'response' muscles
    df_latency_participant_averaged['convergence_latency_cut'] = df_latency_participant_averaged.apply(
        lambda row: [row['convergence_latency'][i] for i in range(len(emg_channels_unsided))
                     if emg_channels_unsided[i] in vec_response],
        axis=1
    )

    df_latency_participant_averaged = df_latency_participant_averaged.set_index('group_name')

    # Extract convergence
    convergence = [None] * len(df)
    i = 0
    for _, row in df.iterrows():
        x = df_latency_participant_averaged.loc[row['set_group'], 'convergence_latency_cut']
        if np.all(np.isnan(x)):
            y = np.nan
        else:
            if (adjust_for_spi == 'average') | (adjust_for_spi == 'global_average'):
                y = np.nanmean(x)  # or some other operation...
            elif (adjust_for_spi == 'target') | (adjust_for_spi == 'global_target'):
                str_target = df_latency_participant_averaged.loc[row['set_group'], 'muscle_targeted']

                ix_target = np.where([str_target == s for s in vec_response])[0]
                if ix_target.shape[0]==0:
                    y = np.nan
                else:
                    y = x[ix_target[0]]
            else:
                raise Exception('Badly specified adjust_for_spi.')
        convergence[i] = y
        i = i + 1

    return convergence


def drop_inconsistent_rows(df, columns_of_interest):
    grouped = df.groupby('set_group')

    filtered_dfs = []

    for group_name, group_df in grouped:
        most_common_combination = group_df[columns_of_interest].mode().iloc[0]
        consistent_rows = group_df[group_df[columns_of_interest].eq(most_common_combination).all(axis=1)]
        filtered_dfs.append(consistent_rows)

    filtered_df = pd.concat(filtered_dfs, ignore_index=True)


    return filtered_df


def append_replicate_minimum_spi_as_extreme(df):
    """
    """
    replicated_rows = pd.DataFrame(columns=df.columns)
    unique_sequences = df[['set_sequence']].drop_duplicates()

    for _, row in unique_sequences.iterrows():
        df_seq = df[df['set_sequence'] == row['set_sequence']].copy()
        m = np.min(df_seq['sccx_latency'])

        if m <= 4.0:
            for adj in [-100, +100]:
                replicate = df_seq[df_seq['sccx_latency'] == m].copy()
                replicate.loc[:, 'is_replicate'] = True
                replicate.loc[:, 'sccx_latency'] = replicate.loc[:, 'sccx_latency'] - adj
                replicated_rows = pd.concat([replicated_rows, replicate])

    df = pd.concat([df, replicated_rows])

    return df


def append_replicate_sccx_only_spi_as_extreme(df, emg_channels_unsided):
    """
    """
    replicated_rows = pd.DataFrame(columns=df.columns)
    unique_sequences = df[['set_sequence']].drop_duplicates()

    for _, row in unique_sequences.iterrows():
        df_seq = df[df['set_sequence'] == row['set_sequence']].copy()

        replicate_cx = df_seq[df_seq['sc_pct_local'] == 0].copy()
        replicate_sc = df_seq[df_seq['cx_pct_local'] == 0].copy()

        numeric_columns = replicate_sc.select_dtypes(include=[np.number]).columns
        non_numeric_columns = replicate_sc.select_dtypes(exclude=[np.number]).columns
        numeric_averages = replicate_sc.loc[:, numeric_columns].apply(safe_nanmean)
        non_numeric_values = replicate_sc.loc[:, non_numeric_columns].iloc[0]
        replicate_sc_combined = pd.concat([numeric_averages, non_numeric_values])

        for ch in emg_channels_unsided:
            if ch in replicate_cx.columns and 'APB' in replicate_sc_combined.index:
                replicate_cx.loc[:, ch] = replicate_cx.loc[:, ch] + replicate_sc_combined[ch]

        for adj in [-100, +100]:
            replicate = replicate_cx.copy()
            replicate.loc[:, 'is_replicate'] = True
            replicate.loc[:, 'sc_pct_local'] = replicate.loc[:, 'sc_pct']
            replicate.loc[:, 'sccx_latency'] = replicate.loc[:, 'sccx_latency'] - adj
            replicated_rows = pd.concat([replicated_rows, replicate])

    df = pd.concat([df, replicated_rows])
    df.reset_index(drop=True, inplace=True)

    return df


def process_participant_ni(cfg, participant_path, ie_only=True):
    muscles_final = cfg['DATA_OPTIONS']['response']

    csv_file = list(participant_path.glob('*_SCAP_TMS_TSS_table.csv'))[0]
    toml_file = list(participant_path.glob('*_SCAP_TMS_TSS_cfg_proc.toml'))[0]
    matlab_file = list(participant_path.glob('*_ep_matrix.mat'))[0]

    # Load CSV and TOML files
    df = pd.read_csv(csv_file)
    config = toml.load(toml_file)
    with h5py.File(matlab_file, 'r') as mat_file:
        mep = mat_file['ep_sliced'][:]

    df['datetime'] = pd.to_datetime(df['datetime_posix'], unit='s')
    df['ix_visit'] = 1
    df.loc[:, 'average_count'] = 1

    ix_visit_counts = df['ix_visit'].value_counts()
    max_count_entries = ix_visit_counts[ix_visit_counts == ix_visit_counts.max()]
    # selected_ix_visit = max_count_entries.index.max()

    # Rename columns according to channel names in TOML file
    channel_names = config['st']['channel']
    column_mapping = {f'auc_{i + 1}': channel_names[i] for i in range(len(channel_names))}
    df.rename(columns=column_mapping, inplace=True)

    column_mapping_pkpk = {f'pkpk_{i + 1}': channel_names[i] + '_pkpk' for i in range(len(channel_names))}
    df.rename(columns=column_mapping_pkpk, inplace=True)

    emg_channels = [ch for ch, ch_type in zip(config['st']['channel'], config['st']['channel_type']) if
                    ch_type == 'EMG']
    mep_channels = [ch[1:] for ch in emg_channels if ch.startswith('L')]
    mep_channels.append('auc_target')

    modified_chunks = []
    mep_chunks = []
    for ix in df['ix_visit'].unique():
        case_chunk = df['ix_visit'] == ix
        df_chunk = df[case_chunk].copy()
        mep_chunk = mep[:, :, case_chunk]

        participant = df_chunk['participant'].values[0]
        visit = int(df_chunk['ix_visit'].values[0])

        target_muscle = df_chunk['target_muscle'].iloc[0]
        side = target_muscle[0]

        emg_channels_on_side = [ch for ch in emg_channels if ch.startswith(side)]
        new_columns = {ch: ch[1:] for ch in emg_channels_on_side}

        case_mep_target = [target_muscle == x for x in config['st']['channel']]
        mep_chunk_target = mep_chunk[case_mep_target, :, :]
        mep_chunk_remapped = np.zeros((len(new_columns.values()), mep_chunk.shape[1], mep_chunk.shape[2]))
        for original, new in new_columns.items():
            if original in df_chunk.columns:
                df_chunk[new] = df_chunk[original]
                df_chunk[new + '_pkpk'] = df_chunk[original + '_pkpk']
                # df_chunk[new + '_saturated'] = df_chunk[original + '_saturated']
            if original in config['st']['channel']:
                case_original = [original == x for x in config['st']['channel']]
                case_new = [new == x for x in new_columns.values()]
                mep_chunk_remapped[case_new, :, :] = mep_chunk[case_original, :, :]
        mep_chunk = mep_chunk_remapped

        # stick the target muscle on the end
        mep_chunk = np.concatenate((mep_chunk, mep_chunk_target), axis=0)

        if cfg['DATA_OPTIONS']['response_transform'] == "log":
            for ch in (emg_channels + list(new_columns.values())):
                df_chunk.loc[:, ch] = np.log(df_chunk.loc[:, ch])
                df_chunk.loc[:, ch + '_pkpk'] = np.log(df_chunk.loc[:, ch + '_pkpk'])

        elif cfg['DATA_OPTIONS']['response_transform'] == "log10":
            for ch in (emg_channels + list(new_columns.values())):  # n.b. here the auc target is set below so it's ok that it's not in here
                df_chunk.loc[:, ch] = np.log10(df_chunk.loc[:, ch])
                df_chunk.loc[:, ch + '_pkpk'] = np.log10(df_chunk.loc[:, ch + '_pkpk'])

        elif cfg['DATA_OPTIONS']['response_transform'] == "log2":
            for ch in (emg_channels + list(new_columns.values())):  # n.b. here the auc target is set below so it's ok that it's not in here
                df_chunk.loc[:, ch] = np.log2(df_chunk.loc[:, ch])
                # if this gives a 0 warning, it is OK. It's an invalid row.
                df_chunk.loc[:, ch + '_pkpk'] = np.log2(df_chunk.loc[:, ch + '_pkpk'])

        elif cfg['DATA_OPTIONS']['response_transform'] == None:
            pass
        elif cfg['DATA_OPTIONS']['response_transform'] == 'linear':
            pass
        else:
            raise Exception('Response transform not specified')

        df_chunk['auc_target'] = df_chunk[target_muscle]
        df_chunk['auc_target_pkpk'] = df_chunk[target_muscle + '_pkpk']

        df_chunk.reset_index(drop=True, inplace=True)
        scap_mask = df_chunk['stim_type'] == "SCAP"
        rec_mask = (df_chunk['stim_type'] == 'TSS') | (df_chunk['stim_type'] == 'TMS')

        logging.info(f'Session: {participant} V{visit}')
        count_summary_to_log(df_chunk, 'EPs at start', muscles_final, scap_mask=scap_mask, rec_mask=rec_mask)

        case_valid_ = df_chunk['case_invalid'] == 0
        mep_chunk = mep_chunk[:, :, case_valid_]
        df_chunk = df_chunk[case_valid_]
        df_chunk.reset_index(drop=True, inplace=True)
        scap_mask = df_chunk['stim_type'] == "SCAP"
        rec_mask = (df_chunk['stim_type'] == 'TSS') | (df_chunk['stim_type'] == 'TMS')

        count_summary_to_log(df_chunk, 'EPs after removing invalid', muscles_final, scap_mask=scap_mask, rec_mask=rec_mask)

        df_chunk.loc[scap_mask, 'cxsc_intensity'] = df_chunk.loc[scap_mask].apply(
            classify_row, cx_string='TMSIntPct', sc_string='TSCSIntPct', axis=1
        )
        df_chunk.loc[scap_mask, 'cxsc_index'] = df_chunk.loc[scap_mask, 'cxsc_intensity'].map(
            intensity_mapping_inverted(cfg['DATA_OPTIONS']['intensities'])
        )

        # Identify SCAP rows with a valid intensity mapping.
        rows_to_drop = scap_mask & df_chunk['cxsc_index'].isna()
        mep_chunk = mep_chunk[:, :, ~rows_to_drop]
        df_chunk = df_chunk[~rows_to_drop]
        df_chunk.reset_index(drop=True, inplace=True)
        scap_mask = df_chunk['stim_type'] == "SCAP"
        rec_mask = (df_chunk['stim_type'] == 'TSS') | (df_chunk['stim_type'] == 'TMS')

        str_kept_intensities = ', '.join(cfg['DATA_OPTIONS']['intensities'])
        count_summary_to_log(df_chunk, f"Reduced SCAP trials here to only {str_kept_intensities} intensities", muscles_final, scap_mask=scap_mask, rec_mask=rec_mask)

        # verified visually that if we are at SPI < 15 then if there is a TSCS response it gets left out of the AUC window
        # At -15 ms, the stim artifact is not yet in the window, but the artifact suppression mainly takes care of this
        # df['SPI_target'].isna() is correct: it includes the conditions that do not have an SPI
        t_cutoff = -15
        rows_to_drop = scap_mask & ~((df_chunk['SPI_target'] >= t_cutoff) | (df_chunk['SPI_target'].isna()))
        df_chunk = df_chunk[~rows_to_drop]
        mep_chunk = mep_chunk[:, :, ~rows_to_drop]
        df_chunk.reset_index(drop=True, inplace=True)
        scap_mask = df_chunk['stim_type'] == "SCAP"
        rec_mask = (df_chunk['stim_type'] == 'TSS') | (df_chunk['stim_type'] == 'TMS')

        count_summary_to_log(df_chunk, f"Reduced SCAP trials to >= {t_cutoff} ms PI", muscles_final, scap_mask=scap_mask, rec_mask=rec_mask)

        # Handle specific SCAP rows that have non-valid SPI intensity values.
        case_no_valid_spi = scap_mask & (
                (df_chunk['cxsc_intensity'] == 'sub-zero') |
                (df_chunk['cxsc_intensity'] == 'zero-sub') |
                (df_chunk['cxsc_intensity'] == 'supra-zero')
        )
        if np.sum(case_no_valid_spi) > 0:
            raise Exception('Not expecting this to happen')
            df_chunk.loc[case_no_valid_spi, 'SPI_target'] = np.nan

        if DEBUGGING:
            unique_rows = df_chunk[['TMSIntPct', 'TSCSIntPct']].drop_duplicates()
            print(unique_rows)
            print('---')

        min_scap_time = df_chunk.loc[scap_mask, 'datetime_posix'].min()
        df_chunk['time'] = df_chunk['datetime_posix'] - min_scap_time

        recompute_runs = any((entry["participant"] == participant) and (entry["visit"] == visit) for entry in cfg['NI_RECOMPUTE_RUNS'])
        if recompute_runs:
            # should have corrected this before removing invalids - but anyway.
            str_correct = f"Correcting {participant} V{visit} run edges."
            print(str_correct)
            logging.info(str_correct)

            # Compute the difference in 'datetime' only for SCAP rows.
            stim_time = df_chunk.loc[scap_mask, 'datetime'].diff().astype('timedelta64[s]')
            stim_time.iloc[0] = np.timedelta64(0, 's')  # Set the first difference to 0

            # Identify breaks: if the time difference exceeds 150 seconds.
            is_break = stim_time > np.timedelta64(150, 's')
            # Detect the transition points when a break ends.
            is_break = np.diff(is_break.astype(int), append=0) == -1

            # Re-compute the run index as a cumulative sum of the breaks.
            df_chunk.loc[scap_mask, 'ix_run'] = np.cumsum(is_break) + 1

        count_summary_to_log(df_chunk, f"EPs at end", muscles_final, scap_mask=scap_mask, rec_mask=rec_mask)

        if ie_only:
            scap_mask = df_chunk['stim_type'] == "SCAP"
            df_chunk = df_chunk[scap_mask]
            mep_chunk = mep_chunk[:, :, scap_mask]
            df_chunk.reset_index(drop=True, inplace=True)
            df_chunk['cxsc_index'] = df_chunk['cxsc_index'].astype(int)  # this has to be here, because there are nans if it's a rec

        logging.info('------------')
        modified_chunks.append(df_chunk)
        mep_chunks.append(mep_chunk)

    logging.info('------------')

    df_modified = pd.concat(modified_chunks)
    mep_chunks = np.concatenate(mep_chunks, axis=2)

    df_modified = df_modified.reset_index(drop=True).sort_index()
    df_modified = df_modified.sort_index()
    sorted_index = df_modified.index
    mep_chunks = mep_chunks[:, :, sorted_index]

    if cfg['DATA_OPTIONS']['visit'] == "all":
        pass
    else:
        raise Exception('Bad visit selection')

    if DEBUGGING:
        unique_groupings = df_modified[['participant', 'ix_visit']].drop_duplicates()
        for _, row in unique_groupings.iterrows():
            print(f"Appending {row['participant']} V{row['ix_visit']}.")

    return df_modified, mep_chunks, mep_channels


def filter_ni(cfg, overwrite=True, ie_only=True, es=''):
    if not cfg['DATA_FOLDER']['noninvasive'].exists():
        download_noninvasive_data(cfg)

    all_participants = [d for d in cfg['DATA_FOLDER']['noninvasive'].iterdir() if d.is_dir()]
    es_extend = '_notieonly' if not ie_only else ''
    es = es + es_extend
    p_out = cfg['DATA_FOLDER']['noninvasive'].parent / 'reproc' / f'filtered{es}.csv'
    p_par = p_out.with_suffix('.parquet')
    p_npa = p_out.with_suffix('.npz')  # using .npy to store the numpy array
    p_log = p_out.with_suffix('.log')
    p_out.parent.mkdir(exist_ok=True)

    make_logger(p_log, console_output=False)

    muscles_final = cfg['DATA_OPTIONS']['response']
    if not p_par.exists() or overwrite:
        all_data = []
        mep_list = []
        for participant in all_participants:
            df_p, mep_p, mep_ch = process_participant_ni(cfg, participant, ie_only=ie_only)
            all_data.append(df_p)
            mep_list.append(mep_p)

        df = pd.concat(all_data, ignore_index=True)
        mep = np.concatenate(mep_list, axis=2)

        df.to_parquet(p_par, engine='pyarrow', index=False)
        np.savez(p_npa, mep=mep, mep_ch=mep_ch)
    else:
        print('Loading pre-processed parquet.')
        df = pd.read_parquet(p_par, engine='pyarrow')
        npzfile = np.load(p_npa)
        mep = npzfile['mep']
        mep_ch = npzfile['mep_ch']

    if ie_only:
        logging.info(f"Keeping only SCAP trials!")
    scap_mask = df['stim_type'] == "SCAP"
    rec_mask = (df['stim_type'] == 'TSS') | (df['stim_type'] == 'TMS')
    count_summary_to_log(df, f"Combined participants", muscles_final, scap_mask=scap_mask, rec_mask=rec_mask)

    if cfg['DATA_OPTIONS']['participants'] == 'all':
        pass
    else:
        df = df[df['participant'].isin(cfg['DATA_OPTIONS']['participants'])]
        counts = np.sum(np.isfinite(df.loc[:, muscles_final]), axis=0)
        logging.info(f"Filtered out some pariticpants: {', '.join(f'{muscle} {count}' for muscle, count in counts.items())}")

    df.loc[:, 'participant_condition_original'] = df.loc[:, 'participant_condition'].copy()

    df.loc[:, 'time'] = df['time'] / 3600  # easier to set the priors if it's in terms of 0 to 1 hour
    df.loc[:, 'run_index'] = df['ix_run'] - 1

    stable_categories = np.sort(df['ix_run'].unique())
    df['run_index'] = pd.Categorical(df['ix_run'], categories=stable_categories, ordered=True).codes
    run_mapping = {i: category for i, category in enumerate(stable_categories)}

    stable_categories = np.sort(df['ix_visit'].unique())
    df['visit_index'] = pd.Categorical(df['ix_visit'], categories=stable_categories, ordered=True).codes
    visit_mapping = {i: category for i, category in enumerate(stable_categories)}

    stable_categories = np.sort(df['participant_condition'].unique())
    df['condition_index'] = pd.Categorical(df['participant_condition'], categories=stable_categories,
                                           ordered=True).codes
    condition_mapping = {i: category for i, category in enumerate(stable_categories)}

    logging.info(f"\n\n--- Manual exclusions ---\n")
    for entry in cfg['NI_EXCLUDE']:
        participant, ix_visit, run, muscles = entry['participant'], entry['visit'], entry['run'], entry['muscles']
        if 'datetime_posix' in entry:
            datetime_posix = entry['datetime_posix']
        else:
            datetime_posix = None

        if np.any((df['participant'] == participant) & (df['ix_visit'] == ix_visit)):
            if run == 'all':
                mask = (df['participant'] == participant) & (df['ix_visit'] == ix_visit)
            else:
                ix_run = run
                mask = (df['participant'] == participant) & (df['ix_visit'] == ix_visit) & (df['ix_run'] == ix_run)
            if datetime_posix:
                datetime_index_mask = [(df.loc[mask, 'datetime_posix'] - datetime_posix[ix]).abs().idxmin() for ix in range(len(datetime_posix))]
                mask[:] = False
                mask.iloc[datetime_index_mask] = True
            # if muscle in target, then include auc_target in muscles
            target_muscle_rel = np.unique(df.loc[mask, 'target_muscle'])[0][1:]
            if target_muscle_rel in muscles:
                muscles.append('auc_target')
            muscles_set = set()
            muscles = [x for x in muscles if not (x in muscles_set or muscles_set.add(x))]
            logging.info(f"Removing {participant} V{ix_visit} run/s:{run} muscle/s:{', '.join(muscles)}")

            count_summary_to_log(df, f"Which takes us from", muscles_final)
            df.loc[mask, muscles] = np.nan
            count_summary_to_log(df, f"To", muscles_final)

            # the mask/muscles combo is applying regardless of whether it's IE or RC so also just apply it to the MEPs:
            muscle_indices = [list(mep_ch).index(m) for m in muscles]
            for muscle_index in muscle_indices:
                mep[muscle_index, :, mask.to_numpy()] = np.nan

    df.reset_index(drop=True, inplace=True)
    df = df.sort_values(by=['participant', 'visit_index', 'run_index'])
    sorted_index = df.index
    mep = mep[:, :, sorted_index]
    df.reset_index(drop=True, inplace=True)

    df['participant_filler'] = 'x'
    df['participant_index'] = df['participant'].factorize()[0]

    intensity_mapping = {v: k for k, v in intensity_mapping_inverted(cfg['DATA_OPTIONS']['intensities']).items()}

    unique_participant_pairs = df.drop_duplicates('participant_index')[['participant_index', 'participant']]
    unique_condition_pairs = df.drop_duplicates('participant_index')[['participant_index', 'participant_condition']]
    unique_filler_pairs = df.drop_duplicates('participant_index')[['participant_index', 'participant_filler']]
    unique_condition_original_pairs = df.drop_duplicates('participant_index')[['participant_index', 'participant_condition_original']]
    mapping = bidir_dict()
    mapping.add_mapping('alias', dict(zip(unique_participant_pairs['participant_index'], unique_participant_pairs['participant'])))
    mapping.add_mapping('participant', dict(zip(unique_participant_pairs['participant_index'], unique_participant_pairs['participant'])))
    mapping.add_mapping('participant_condition', dict(zip(unique_condition_pairs['participant_index'], unique_condition_pairs['participant_condition'])))
    mapping.add_mapping('participant_condition_original', dict(zip(unique_condition_original_pairs['participant_index'], unique_condition_original_pairs['participant_condition_original'])))
    mapping.add_mapping('condition', condition_mapping)
    mapping.add_mapping('intensity', intensity_mapping)
    mapping.add_mapping('visit', visit_mapping)
    mapping.add_mapping('run', run_mapping)
    mapping.add_mapping('muscle', dict(zip(range(len(cfg['DATA_OPTIONS']['response'])), cfg['DATA_OPTIONS']['response'])))
    mapping.add_mapping('participant_filler', dict(zip(unique_filler_pairs['participant_index'], unique_filler_pairs['participant_filler'])))

    # reject late SCS times
    # assert muscles_final == ['auc_target'], 'This dataset only makes sense if you are looking specifically at auc_target'
    # the 5.5 matches the time used in the matlab SCS artifact suppression code
    # case_late_scs = (df['TSCSDelay'] + 5.5 > df['onset_time_tms'])
    # scap_mask = (df['stim_type'] == "SCAP")
    # df.loc[case_late_scs & scap_mask, muscles_final] = np.nan

    # note that visit_index and run_index are re-numbered versions (0-indexed, no missing) versions of ix_visit and ix_run
    df.to_csv(p_out, index=False)

    logging.info("\n--------------------\n")
    scap_mask = df['stim_type'] == "SCAP"
    rec_mask = (df['stim_type'] == 'TSS') | (df['stim_type'] == 'TMS')
    count_summary_to_log(df, f"Final count is", muscles_final, scap_mask=scap_mask, rec_mask=rec_mask)

    return df, mapping, mep, mep_ch


def process_participant_io(cfg, dfx, cfg_data, df_latency_participant):
    variables_to_keep = ['datetime', 'cx_count', 'cx_pct', 'cx_voltage', 'is_valid', 'main_targeted_side', 'mode',
                         'muscle_targeted', 'participant', 'position', 'sc_approach', 'sc_biphasic', 'sc_count',
                         'sc_current', 'sc_depth', 'sc_displacement', 'sc_electrode', 'sc_electrode_configuration',
                         'sc_electrode_type', 'sc_frequency', 'sc_impedance1', 'sc_impedance2', 'sc_ipi',
                         'sc_laterality', 'sc_level', 'sc_misc', 'sc_pct', 'sc_polarity', 'sc_pw', 'sc_voltage',
                         'sccx_latency', 'set_group', 'set_sequence', 'alias',
                         'case_valid_sc_pairing', 'case_valid_cx_pairing', 'case_sc0_pairing', 'case_cx0_pairing', 'is_replicate', 'average_count']

    columns_of_interest = ['muscle_targeted', 'sc_approach', 'sc_biphasic', 'sc_count', 'sc_depth', 'sc_electrode',
        'sc_electrode_configuration', 'sc_electrode_type', 'sc_frequency', 'sc_ipi', 'sc_laterality',
        'sc_level', 'sc_polarity', 'sc_pw']

    df = dfx.copy()

    df = df[df['mode'].isin(['research_paired_repeat', 'research_paired_averaged'])].copy()
    if 'average_count' not in list(df.columns):
        df.loc[(df['mode'] == 'research_paired_averaged'), 'average_count'] = 5
        df.loc[np.invert(df['mode'] == 'research_paired_averaged'), 'average_count'] = 1

    if df.empty:
        print('no averaged or repeat mode ->')
        return None

    participant = df['participant'].unique()[0]
    if participant not in cfg_data['mcintosh2024']:
        print("not in cfg_data['mcintosh2024']->")
        return None

    if cfg['DATA_OPTIONS']['visit'] == "mcintosh2024":
        cond = cfg_data['mcintosh2024'][participant]
        case_set_group = df.loc[:, 'set_group'] == cond['set_group']
        df = df[case_set_group].copy()

    elif cfg['DATA_OPTIONS']['visit'] == "all":
        pass
    else:
        raise Exception('Bad visit selection')

    if df.empty:
        print("set_group not in cfg_data['mcintosh2024']->")
        return None

    df = df.copy()  # to avoid copy warning...
    df.loc[:, 'is_replicate'] = False

    df = drop_inconsistent_rows(df, columns_of_interest)

    df['datetime'] = pd.to_datetime(df['datetime_posix'], unit='s')
    list_col = ['is_valid', 'case_valid_sc_pairing', 'case_valid_cx_pairing', 'case_sc0_pairing', 'case_cx0_pairing']
    for col in list_col:
        df[col] = df[col].astype(bool)

    # Rename columns according to channel names in TOML file
    channel_names = cfg_data['channels']
    column_mapping = {f'auc_sccx_{i + 1}': channel_names[i] for i in range(len(channel_names))}
    df.rename(columns=column_mapping, inplace=True)

    emg_channels = [ch for ch, ch_type in zip(cfg_data['channels'], cfg_data['channel_type']) if
                    ch_type == 'EMG']
    emg_channels_unsided = [ch[1:] for ch in emg_channels if ch.startswith('L')]

    def update_columns(row, emg_channels):
        # strip the side from the targeted side
        side = row['main_targeted_side']
        emg_channels_on_side = [ch for ch in emg_channels if ch.startswith(side)]
        new_columns = {ch: ch[1:] for ch in emg_channels_on_side}
        for original, new in new_columns.items():
            if original in row.index:
                row[new] = row[original]
        return row

    # Apply the function to each row in the DataFrame
    df = df.apply(lambda row: update_columns(row, emg_channels), axis=1)

    for entry in cfg['IO_EXCLUDE']:
        participant, set_group, run, muscles = entry['participant'], entry['set_group'], entry['set_sequence'], entry['muscles']
        if run == 'all':
            mask = (df['participant'] == participant) & (df['set_group'] == set_group)
        else:
            set_sequence = run
            mask = (df['participant'] == participant) & (df['set_group'] == set_group) & (df['set_sequence'] == set_sequence)
        df.loc[mask, muscles] = np.nan

    # Create the 'auc_target' column
    auc_target_series = df.apply(
        lambda row: row[row['muscle_targeted']]
        if pd.notna(row['muscle_targeted'])
        else np.nan,
        axis=1
    )

    # Concatenate the new Series to the DataFrame in one go
    df = pd.concat([df, auc_target_series.rename('auc_target')], axis=1)

    variables_to_keep_local = variables_to_keep + emg_channels_unsided
    variables_to_keep_local = variables_to_keep_local + emg_channels
    variables_to_keep_local = variables_to_keep_local + ['auc_target']

    df = df[variables_to_keep_local].copy()

    df['participant_condition'] = df['sc_approach']

    df.loc[:, 'set_group'] = df['set_group'].astype(str)
    df = df[df['set_group'].str.startswith(('gpa', 'gpr'), na=False)]

    df['cx_pct_local'] = df['cx_pct']
    df['sc_pct_local'] = df['sc_pct']
    df.loc[df['case_sc0_pairing'], 'sc_pct_local'] = 0
    df.loc[df['case_cx0_pairing'], 'cx_pct_local'] = 0
    df.loc[df['cx_voltage'] < 10, 'cx_pct_local'] = 0  # 2025-04-01 sometimes you set 50V and it delivers substanitally less - so put cut off at 10
    df.loc[df['sc_voltage'] < 5e-4, 'sc_pct_local'] = 0

    df = df[(df['is_valid']) & (df['case_valid_sc_pairing']) & (df['case_valid_cx_pairing'])].copy()
    if df.empty:
        return None

    augment = cfg['DATA_OPTIONS']['intraoperative'][1]
    if augment == 'replicate_minimum_spi_as_extreme':
        df = append_replicate_minimum_spi_as_extreme(df)

    if augment == 'replicate_sccx_only_spi_as_extreme':
        df = append_replicate_sccx_only_spi_as_extreme(df, emg_channels_unsided)

    adjust_for_spi = cfg['DATA_OPTIONS']['intraoperative'][0]
    if adjust_for_spi:
        convergence = marge_and_calculate_convergence(
            cfg=cfg,
            df=df,
            df_latency_participant=df_latency_participant,
            columns_of_interest=columns_of_interest,
            emg_channels=emg_channels,
            emg_channels_unsided=emg_channels_unsided,
            adjust_for_spi=adjust_for_spi
        )
        df.loc[:, 'convergence_adjustment'] = convergence
    else:
        df.loc[:, 'convergence_adjustment'] = 0.0

    df['cxsc_intensity'] = df.apply(classify_row, cx_string='cx_pct_local', sc_string='sc_pct_local', axis=1)
    df['cxsc_index'] = df['cxsc_intensity'].map(intensity_mapping_inverted(cfg['DATA_OPTIONS']['intensities']))
    df = df.dropna(subset=['cxsc_index'])  # if they are not in the mapping they get na
    df['cxsc_index'] = df['cxsc_index'].astype(int)

    min_samples_in = 1
    remove_from = 'set_group'  # set_sequence or set_group
    filtered_df = df[(df['sc_pct_local'] != 0) | (df['cx_pct_local'] != 0) | (~df['is_replicate'])].copy()
    set_counts = filtered_df[remove_from].value_counts()
    to_remove = set_counts[set_counts < min_samples_in]
    df = df[~df[remove_from].isin(list(to_remove.index.values))]
    if df.empty:
        print(f'Removing because no data left after removing groups with < {min_samples_in} samples ->')
        return None

    df['visit_index'] = pd.factorize(df['set_group'])[0].astype(int)
    df['run_index'] = df.groupby('set_group')['set_sequence'].transform(lambda x: pd.factorize(x)[0]).astype(int)

    df = df.sort_values(by=['datetime'])
    df.reset_index(drop=True, inplace=True)

    df['time'] = 0.0
    unique_groups = df[['set_group']].drop_duplicates()
    for _, row in unique_groups.iterrows():
        subset = df[(df['set_group'] == row['set_group'])]
        time_in_minutes = (subset['datetime'] - subset['datetime'].iloc[0]).dt.total_seconds() / 600  # units are... tens of minutes!
        df.loc[time_in_minutes.index, 'time'] = time_in_minutes

    if cfg['DATA_OPTIONS']['response_transform'] == "log":
        for ch in (emg_channels + emg_channels_unsided + ['auc_target']):
            df.loc[:, ch] = np.log(df.loc[:, ch])
    elif cfg['DATA_OPTIONS']['response_transform'] == "log10":
        for ch in (emg_channels + emg_channels_unsided + ['auc_target']):
            df.loc[:, ch] = np.log10(df.loc[:, ch])
    elif cfg['DATA_OPTIONS']['response_transform'] == "log2":
        for ch in (emg_channels + emg_channels_unsided + ['auc_target']):
            df.loc[:, ch] = np.log2(df.loc[:, ch])
    elif cfg['DATA_OPTIONS']['response_transform'] == None:
        pass
    elif cfg['DATA_OPTIONS']['response_transform'] == 'linear':
        pass
    else:
        raise Exception('Response transform not specified')

    return df


def filter_io(cfg, overwrite=False, es=''):
    p_info = Path(cfg['DATA_FOLDER']['intraoperative'] / f'out_info.csv')
    p_latency = Path(cfg['DATA_FOLDER']['intraoperative'] / f'out_latency.csv')
    p_par = Path(cfg['DATA_FOLDER']['intraoperative'] / f'out_info{es}.parquet')
    config = toml.load(p_info.with_suffix('.toml'))
    df_o = pd.read_csv(p_info, low_memory=False)
    df_latency = pd.read_csv(p_latency, low_memory=False)

    vec_participant = df_o['participant'].unique()
    vec_alias = [df_o[df_o['participant'] == participant]['alias'].iloc[0] for participant in vec_participant]

    if not p_par.exists() or overwrite:
        modified_chunks = []
        for ix_p, participant in enumerate(vec_participant):
            df_participant = df_o[df_o['participant'] == participant].copy()
            df_latency_participant = df_latency[df_latency['participant'] == participant].copy()
            df_participant = process_participant_io(cfg, df_participant, config, df_latency_participant)
            if df_participant is not None:
                print(f'{participant}, {vec_alias[ix_p]}')
                print('')
                modified_chunks.append(df_participant)
            else:
                print(f'SKIPPING: {participant}, {vec_alias[ix_p]}')
                print('')
        df = pd.concat(modified_chunks)
        df.to_parquet(p_par, engine='pyarrow', index=False)
    else:
        print('Loading pre-processed parquet.')
        df = pd.read_parquet(p_par, engine='pyarrow')

    if cfg['DATA_OPTIONS']['participants'] == 'all':
        pass
    else:
        df = df[df['participant'].isin(cfg['DATA_OPTIONS']['participants'])]

    adjust_for_spi = cfg['DATA_OPTIONS']['intraoperative'][0]
    if (adjust_for_spi == 'average') | (adjust_for_spi == 'target'):
        df.loc[:, 'SPI_target'] = df['sccx_latency'] - df['convergence_adjustment']
    elif (adjust_for_spi == 'global_average') | (adjust_for_spi == 'global_target'):
        df.loc[:, 'SPI_target'] = df['sccx_latency'] - np.nanmean(df['convergence_adjustment'])
    elif adjust_for_spi is None:
        df.loc[:, 'SPI_target'] = df['sccx_latency']

    df = df.dropna(subset=['SPI_target'])

    df.loc[:, 'participant_condition_original'] = df.loc[:, 'participant_condition'].copy()

    df = df.sort_values(by=['participant', 'datetime'])

    df['participant_index'] = df['participant'].factorize()[0]
    unique_pairs = df.drop_duplicates('participant_index')[['participant_index', 'participant']]

    stable_categories = np.sort(df['participant_condition'].unique())
    df['condition_index'] = pd.Categorical(df['participant_condition'], categories=stable_categories,
                                           ordered=True).codes
    condition_mapping = {i: category for i, category in enumerate(stable_categories)}

    intensity_mapping = {v: k for k, v in intensity_mapping_inverted(cfg['DATA_OPTIONS']['intensities']).items()}

    mapping_visit = {}
    mapping_run = {}
    for participant in df['participant'].unique():
        df_participant = df[df['participant'] == participant].copy()
        visit_index_to_group_ = dict(enumerate(pd.factorize(df_participant['set_group'])[1]))
        run_index_to_sequence_ = {
            set_group: dict(enumerate(pd.factorize(df_participant[df_participant['set_group'] == set_group]['set_sequence'])[1]))
            for set_group in df_participant['set_group'].unique()
        }
        mapping_visit[participant] = visit_index_to_group_
        mapping_run[participant] = run_index_to_sequence_
    visit_mapping = {int(i): f'{i}' for i in df['visit_index'].unique()}
    run_mapping = {int(i): f'{i}' for i in df['run_index'].unique()}
    mapping = bidir_dict()
    unique_pairs_alias = df.drop_duplicates('participant_index')[['participant_index', 'alias']]
    unique_condition_pairs = df.drop_duplicates('participant_index')[['participant_index', 'participant_condition']]
    unique_condition_original_pairs = df.drop_duplicates('participant_index')[['participant_index', 'participant_condition_original']]
    mapping.add_mapping('alias', dict(zip(unique_pairs_alias['participant_index'], unique_pairs_alias['alias'])))
    mapping.add_mapping('participant', dict(zip(unique_pairs['participant_index'], unique_pairs['participant'])))
    mapping.add_mapping('participant_condition', dict(zip(unique_condition_pairs['participant_index'], unique_condition_pairs['participant_condition'])))
    mapping.add_mapping('participant_condition_original', dict(zip(unique_condition_original_pairs['participant_index'], unique_condition_original_pairs['participant_condition_original'])))
    mapping.add_mapping('condition', condition_mapping)
    mapping.add_mapping('intensity', intensity_mapping)
    mapping.add_mapping('visit', visit_mapping)
    mapping.add_mapping('run', run_mapping)
    mapping.add_mapping('muscle', dict(zip(range(len(cfg['DATA_OPTIONS']['response'])), cfg['DATA_OPTIONS']['response'])))

    df.reset_index(drop=True, inplace=True)

    return df, mapping, None, None

def filter_data(cfg, overwrite=True, es=''):
    if cfg['DATA_OPTIONS']['type'] == 'intraoperative':
        df, mapping, mep, mep_ch = filter_io(cfg, overwrite=overwrite, es=es)
    elif cfg['DATA_OPTIONS']['type'] == 'noninvasive':
        df, mapping, mep, mep_ch = filter_ni(cfg, overwrite=overwrite, es=es)
    else:
        raise Exception('???')
    return df, mapping, mep, mep_ch


if __name__ == "__main__":
    pass
