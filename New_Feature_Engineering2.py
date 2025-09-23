#!/usr/bin/env python3
# coding: utf-8

import cudf
import pandas as pd
import numpy as np
import os
import io
from tqdm import tqdm
import time
import re
import gc
from collections import defaultdict, Counter

# -----------------------
# Config
# -----------------------
WORK_START = 8   # 8:00 AM
WORK_END   = 18  # 6:00 PM

# -----------------------
# Chunked CSV reader (works for cudf and pandas)
# -----------------------
def read_csv_chunks(file_path, chunk_bytes=1024**3, **kwargs):
    """
    Read large CSV in chunks, returning pandas or cudf frames depending on engine param.
    """
    engine = kwargs.pop('engine', 'cudf')
    try:
        header_df = cudf.read_csv(file_path, nrows=0) if engine == 'cudf' else pd.read_csv(file_path, nrows=0)
    except (FileNotFoundError, StopIteration):
        return
    cols = list(header_df.columns)

    leftover = b''
    with open(file_path, 'rb') as f:
        f.readline()  # Skip header
        while True:
            data = f.read(chunk_bytes)
            if not data and not leftover:
                break
            buf = leftover + data
            nl = buf.rfind(b"\n")
            if nl >= 0:
                part, leftover = buf[:nl+1], buf[nl+1:]
            else:
                part, leftover = b"", buf
            if part:
                buf_io = io.BytesIO(part)
                # Use consistent header/names parameters
                chunk_kwargs = kwargs.copy()
                chunk_kwargs.update({'names': cols, 'header': None})
                yield cudf.read_csv(buf_io, **chunk_kwargs) if engine == 'cudf' else pd.read_csv(buf_io, **chunk_kwargs)
    if leftover.strip():
        buf_io = io.BytesIO(leftover)
        chunk_kwargs = kwargs.copy()
        chunk_kwargs.update({'names': cols, 'header': None})
        yield cudf.read_csv(buf_io, **chunk_kwargs) if engine == 'cudf' else pd.read_csv(buf_io, **chunk_kwargs)

# -----------------------
# Helpers
# -----------------------
def is_offhour(ts_series):
    if isinstance(ts_series, pd.Series):
        return (ts_series.dt.hour < WORK_START) | (ts_series.dt.hour >= WORK_END)
    else:
        return (ts_series.dt.hour < WORK_START) | (ts_series.dt.hour >= WORK_END)

def extract_domain(url_series):
    s = url_series.fillna('').astype('str')
    # use pandas string ops for robustness
    if isinstance(s, pd.Series):
        domains = s.str.extract(r'^[^:/]+://([^/]+)')[0].fillna('')
        missing = domains == ''
        if missing.any():
            parts = s[missing].str.split('/', 2)
            first = parts.str.get(0).fillna('')
            second = parts.str.get(1).fillna('')
            fallback = first.mask(first.eq(''), second).fillna('')
            domains.loc[missing] = fallback
        return domains.fillna('')
    else:
        s_pd = s.to_pandas()
        domains = s_pd.str.extract(r'^[^:/]+://([^/]+)')[0].fillna('')
        missing = domains == ''
        if missing.any():
            parts = s_pd[missing].str.split('/', 2)
            first = parts.str.get(0).fillna('')
            second = parts.str.get(1).fillna('')
            fallback = first.mask(first.eq(''), second).fillna('')
            domains.loc[missing] = fallback
        return cudf.from_pandas(domains)

def safe_requests_per_hour(requests, duration_hours):
    """Requests per hour; return 0 if span is zero to avoid distortion"""
    return (requests / duration_hours) if duration_hours > 0 else 0.0

# -----------------------
# Process Logon (chunked, cudf -> pandas roundtrip where needed)
# -----------------------
def process_logon(logon_path, chunk_bytes=1024**3):
    parts = []
    raw_parts = []
    if not os.path.exists(logon_path):
        return cudf.DataFrame(), pd.DataFrame()
    for chunk in tqdm(read_csv_chunks(logon_path, chunk_bytes, engine='cudf'), desc='Logon'):
        # ensure date (pandas roundtrip as before)
        chunk['date'] = cudf.from_pandas(pd.to_datetime(chunk['date'].to_pandas(), errors='coerce'))
        chunk = chunk.dropna(subset=['date'])
        chunk['day'] = chunk['date'].dt.floor('D')
        chunk['user'] = chunk['user'].astype('str')
        # counts
        act_series = chunk['activity'].fillna('').str.lower()
        logon_mask = act_series.str.contains('logon')
        logoff_mask = act_series.str.contains('logoff')
        logon_count = chunk[logon_mask].groupby(['user','day']).size().reset_index(name='logon_count')
        logoff_count = chunk[logoff_mask].groupby(['user','day']).size().reset_index(name='logoff_count')
        offhour_mask = is_offhour(chunk['date']) & act_series.str.contains('logon|logoff')
        offhour_events = chunk[offhour_mask]
        offhour_count = offhour_events.groupby(['user','day']).size().reset_index(name='offhour_logonlogoff_count')

        agg = logon_count.merge(logoff_count, on=['user','day'], how='outer')
        agg = agg.merge(offhour_count, on=['user','day'], how='outer')
        agg = agg.fillna(0)
        parts.append(agg)
        raw_parts.append(chunk[['user','day','date','pc','activity']].to_pandas())
        del chunk, logon_count, logoff_count, offhour_count, agg
        gc.collect()
    if not parts:
        return cudf.DataFrame(), pd.DataFrame()
    agg_all = cudf.concat(parts).groupby(['user','day']).agg({
        'logon_count':'sum',
        'logoff_count':'sum',
        'offhour_logonlogoff_count':'sum'
    }).reset_index()

    raw_all = pd.concat(raw_parts).sort_values(['user','day','date']).reset_index(drop=True)

    # distinct PCs per user-day from raw
    distinct_pcs_correct = raw_all.groupby(['user','day'])['pc'].nunique().reset_index(name='distinct_pcs')
    agg_all = agg_all.to_pandas()
    agg_all = agg_all.merge(distinct_pcs_correct, on=['user','day'], how='left').fillna({'distinct_pcs': 0})
    agg_all = cudf.from_pandas(agg_all)

    # pair logon->logoff per PC to compute duration (pandas loop as before)
    from collections import deque
    durations = []
    for (u, d), g in raw_all.groupby(['user','day']):
        total_seconds = 0.0
        for pc, gp in g.sort_values('date').groupby('pc'):
            stack = deque()
            for _, row in gp.iterrows():
                act = str(row['activity']).lower()
                ts = row['date']
                if 'logon' in act:
                    stack.append(ts)
                elif 'logoff' in act and stack:
                    start = stack.popleft()
                    total_seconds += (ts - start).total_seconds()
        durations.append({'user': u, 'day': pd.Timestamp(d), 'online_duration_hours': total_seconds / 3600.0})

    dur_df = pd.DataFrame(durations) if durations else pd.DataFrame(columns=['user','day','online_duration_hours'])
    agg_pd = agg_all.to_pandas()
    agg_pd = agg_pd.merge(dur_df, on=['user','day'], how='left').fillna({'online_duration_hours':0})
    # ensure numeric types
    for c in ['logon_count','logoff_count','distinct_pcs','offhour_logonlogoff_count']:
        if c in agg_pd.columns:
            agg_pd[c] = agg_pd[c].fillna(0).astype(int)
    agg_pd['online_duration_hours'] = agg_pd['online_duration_hours'].astype(float)

    return cudf.from_pandas(agg_pd), agg_pd

# -----------------------
# Process HTTP (chunked, cudf -> pandas roundtrip where needed)
# -----------------------
def process_http(http_path, chunk_bytes=1024**3):
    parts = []
    raw_parts = []
    if not os.path.exists(http_path):
        return cudf.DataFrame(), pd.DataFrame()
    for chunk in tqdm(read_csv_chunks(http_path, chunk_bytes, engine='cudf'), desc='HTTP'):
        chunk['date'] = cudf.from_pandas(pd.to_datetime(chunk['date'].to_pandas(), errors='coerce'))
        chunk = chunk.dropna(subset=['date'])
        chunk['day'] = chunk['date'].dt.floor('D')
        chunk['user'] = chunk['user'].astype('str')
        # domain extraction via pandas roundtrip
        chunk_pd = chunk.to_pandas()
        chunk_pd['domain'] = extract_domain(chunk_pd['url'])
        chunk_pd['is_wikileaks'] = chunk_pd['domain'].fillna('').str.lower().str.contains('wikileaks').astype(int)
        # only collect summable counts
        total_req = chunk_pd.groupby(['user','day']).size().reset_index(name='total_http_requests')
        wik = chunk_pd.groupby(['user','day'])['is_wikileaks'].max().reset_index(name='wikileaks_flag')

        agg = total_req.merge(wik, on=['user','day'], how='outer')
        parts.append(cudf.from_pandas(agg))
        raw_parts.append(chunk_pd[['user','day','date','domain']])
        del chunk, chunk_pd, total_req, wik, agg
        gc.collect()

    if not parts:
        return cudf.DataFrame(), pd.DataFrame()
    http_partial = cudf.concat(parts).groupby(['user','day']).agg({'total_http_requests':'sum', 'wikileaks_flag':'max'}).reset_index()
    raw_all = pd.concat(raw_parts).sort_values(['user','day','date']).reset_index(drop=True)

    # unique domains per user-day
    unique_domains_correct = raw_all.groupby(['user','day'])['domain'].nunique().reset_index(name='unique_domains')

    # session duration as span per user-day (pandas)
    session_spans = raw_all.groupby(['user','day'])['date'].agg(['min','max']).reset_index()
    session_spans['session_duration_hours'] = (session_spans['max'] - session_spans['min']).dt.total_seconds() / 3600.0
    session_spans = session_spans[['user','day','session_duration_hours']]

    # merge corrected metrics
    http_all = http_partial.to_pandas()
    http_all = http_all.merge(unique_domains_correct, on=['user','day'], how='left')
    http_all = http_all.merge(session_spans, on=['user','day'], how='left')
    http_all = http_all.fillna({'unique_domains': 0, 'session_duration_hours': 0})

    # compute avg_requests_per_session (pandas apply)
    http_all['avg_requests_per_session'] = http_all.apply(
        lambda r: safe_requests_per_hour(r['total_http_requests'], r['session_duration_hours']),
        axis=1
    ).astype('float32')

    http_all = cudf.from_pandas(http_all)
    return http_all, raw_all

# -----------------------
# Process Device (USB) (chunked, cudf -> pandas roundtrip where needed)
# -----------------------
def process_device(device_path, chunk_bytes=1024**3):
    parts = []
    raw_parts = []
    afterhours_parts = []
    if not os.path.exists(device_path):
        return cudf.DataFrame(), pd.DataFrame()
    for chunk in tqdm(read_csv_chunks(device_path, chunk_bytes, engine='cudf'), desc='Device'):
        chunk['date'] = cudf.from_pandas(pd.to_datetime(chunk['date'].to_pandas(), errors='coerce'))
        chunk = chunk.dropna(subset=['date'])
        chunk['day'] = chunk['date'].dt.floor('D')
        chunk['user'] = chunk['user'].astype('str')
        chunk_pd = chunk.to_pandas()
        # counts of connect/disconnect
        acts = chunk_pd['activity'].fillna('').str.lower().str.strip()
        conn_mask = (acts == 'connect')
        disc_mask = (acts == 'disconnect')

        usb_count = chunk_pd[conn_mask | disc_mask].groupby(['user','day']).size().reset_index(name='usb_connect_disconnect_count')
        # after-hours usb connects
        offhour_conn = chunk_pd[conn_mask & ((chunk_pd['date'].dt.hour < WORK_START) | (chunk_pd['date'].dt.hour >= WORK_END))]
        offhour_conn = offhour_conn.groupby(['user','day']).size().reset_index(name='afterhours_usb_connects')
        # collect raw for session pairing
        parts.append(cudf.from_pandas(usb_count))
        raw_parts.append(chunk_pd[['user','day','date','pc','activity']])
        afterhours_parts.append(cudf.from_pandas(offhour_conn))
        del chunk, chunk_pd, usb_count, offhour_conn
        gc.collect()
    if not parts:
        return cudf.DataFrame(), pd.DataFrame()
    dev_agg = cudf.concat(parts).groupby(['user','day']).sum().reset_index()
    if afterhours_parts:
        afterhours_agg = cudf.concat(afterhours_parts).groupby(['user','day']).sum().reset_index()
        dev_agg = dev_agg.merge(afterhours_agg, on=['user','day'], how='left').fillna({'afterhours_usb_connects': 0})
    else:
        dev_agg['afterhours_usb_connects'] = 0
    raw_all = pd.concat(raw_parts).sort_values(['user','day','date']).reset_index(drop=True)

    # pair connect->disconnect per user-day and per PC (pandas loop)
    from collections import deque
    durations = []
    for (u, d), g in raw_all.groupby(['user','day']):
        total_seconds = 0.0
        session_counts = 0
        for pc, gp in g.sort_values('date').groupby('pc'):
            connect_q = deque()
            for _, row in gp.iterrows():
                act = str(row['activity']).strip().lower()
                if act == 'disconnect':  # check disconnect first or use word-boundary regex
                    if connect_q:
                        start = connect_q.popleft()
                        total_seconds += (row['date'] - start).total_seconds()
                        session_counts += 1
                elif act == 'connect':
                    connect_q.append(row['date'])
        avg_dur = (total_seconds / session_counts / 3600.0) if session_counts > 0 else 0.0
        durations.append({
            'user': u, 'day': pd.Timestamp(d),
            'total_usb_session_hours': total_seconds / 3600.0,
            'avg_usb_session_hours': avg_dur,
            'usb_session_count': session_counts
        })

    dur_df = pd.DataFrame(durations) if durations else pd.DataFrame(columns=['user','day','total_usb_session_hours','avg_usb_session_hours','usb_session_count'])
    dev_pd = dev_agg.to_pandas()
    dev_pd = dev_pd.merge(dur_df, on=['user','day'], how='left').fillna({'total_usb_session_hours':0,'avg_usb_session_hours':0,'usb_session_count':0})
    return cudf.from_pandas(dev_pd), raw_all

# -----------------------
# Scenario mapping from answers4.csv
# -----------------------
def process_scenarios(answers_path):
    if not os.path.exists(answers_path):
        return pd.DataFrame(columns=['user','day','scenario'])
    df = pd.read_csv(answers_path, parse_dates=['date'], low_memory=True)
    df = df.dropna(subset=['date'])
    df['day'] = df['date'].dt.floor('D')
    df['scenario'] = pd.to_numeric(df.get('scenario', 0), errors='coerce').fillna(0).astype(int)
    out = df[['user','day','scenario']].drop_duplicates()
    return out

# -----------------------
# Time-aware baselines: helpers (pandas)
# -----------------------
def per_user_expanding_stats(df, col):
    grp = df.groupby('user')[col]
    mu_prev = grp.expanding().mean().shift(1).reset_index(level=0, drop=True)
    sd_prev = grp.expanding().std().shift(1).reset_index(level=0, drop=True)
    return mu_prev, sd_prev

def per_user_rolling_stats(df, col, window):
    base = df.groupby('user')[col].shift(1)
    roll_mu = base.groupby(df['user']).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_sd = base.groupby(df['user']).rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)
    return roll_mu, roll_sd

# -----------------------
# Global baselines (per day)
# -----------------------
def day_level_global_stats(df, col):
    # compute global expanding mean/std up to t-1 aggregated by day
    tmp = df[['day', col]].copy()
    tmp['sumsq'] = tmp[col].astype(float)**2
    daily = tmp.groupby('day').agg(count=(col,'size'), sum=(col,'sum'), sumsq=('sumsq','sum')).sort_index().reset_index()
    daily['cum_count_prev'] = daily['count'].cumsum().shift(1).fillna(0.0)
    daily['cum_sum_prev']   = daily['sum'].cumsum().shift(1).fillna(0.0)
    daily['cum_sumsq_prev'] = daily['sumsq'].cumsum().shift(1).fillna(0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mu_prev = daily['cum_sum_prev'] / daily['cum_count_prev'].replace(0, np.nan)
        var_prev = daily['cum_sumsq_prev'] / daily['cum_count_prev'].replace(0, np.nan) - (mu_prev**2)
        std_prev = np.sqrt(np.maximum(var_prev, 0.0))
    daily['global_mu_prev'] = mu_prev
    daily['global_sd_prev'] = std_prev
    return daily[['day','global_mu_prev','global_sd_prev','count']]

def day_level_global_rolling(df, col, window):
    tmp = df[['day', col]].copy()
    tmp['sumsq'] = tmp[col].astype(float)**2
    daily = tmp.groupby('day').agg(count=(col,'size'), sum=(col,'sum'), sumsq=('sumsq','sum')).sort_index()
    roll_sum   = daily['sum'].shift(1).rolling(window=window, min_periods=1).sum()
    roll_sumsq = daily['sumsq'].shift(1).rolling(window=window, min_periods=1).sum()
    roll_count = daily['count'].shift(1).rolling(window=window, min_periods=1).sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        mu_prev = roll_sum / roll_count.replace(0, np.nan)
        var_prev = roll_sumsq / roll_count.replace(0, np.nan) - (mu_prev**2)
        sd_prev = np.sqrt(np.maximum(var_prev, 0.0))
    out = pd.DataFrame({
        'day': daily.index,
        f'global_rollmu_prev_{window}d': mu_prev.values,
        f'global_rollsd_prev_{window}d': sd_prev.values,
        f'global_rollcount_prev_{window}d': roll_count.values
    })
    return out

# -----------------------
# Merge and compute time-aware per-user and global z-scores only (past-only)
# -----------------------
def merge_all(data_dir, answers_path, chunk_bytes=1024**3):
    t0 = time.time()
    print('Processing logon/http/device (chunked cudf) ...')
    logon_cudf, logon_raw_pd = process_logon(os.path.join(data_dir,'logon.csv'), chunk_bytes)
    http_cudf, http_raw_pd   = process_http(os.path.join(data_dir,'http.csv'), chunk_bytes)
    dev_cudf, dev_raw_pd     = process_device(os.path.join(data_dir,'device.csv'), chunk_bytes)

    # convert cudf aggregates to pandas for merging/rolling
    parts = []
    for df in [logon_cudf, http_cudf, dev_cudf]:
        if isinstance(df, cudf.DataFrame) and len(df)>0:
            parts.append(df.to_pandas())
    if not parts:
        print('No data found')
        return pd.DataFrame(), pd.DataFrame()
    merged = parts[0]
    for df in parts[1:]:
        merged = merged.merge(df, on=['user','day'], how='outer')

    # ensure day dtype
    merged['day'] = pd.to_datetime(merged['day'])
    required_zero_cols = [
        'logon_count','logoff_count','online_duration_hours','distinct_pcs',
        'offhour_logonlogoff_count','total_http_requests','unique_domains',
        'session_duration_hours','usb_connect_disconnect_count','afterhours_usb_connects',
        'avg_usb_session_hours','total_usb_session_hours','wikileaks_flag',
        'offhour_http_count', 'avg_requests_per_session'
    ]
    for c in required_zero_cols:
        if c not in merged.columns:
            merged[c] = 0
        else:
            merged[c] = merged[c].fillna(0)

    # cross-modal features
    merged['total_activity_count'] = (
        merged['logon_count'] + merged['logoff_count'] + merged['total_http_requests'] + merged['usb_connect_disconnect_count']
    )
    merged['pc_switch_rate'] = (merged['distinct_pcs'] / merged['logon_count'].replace(0, np.nan)).fillna(0).astype(float)

    # HTTP offhour count from raw timestamps
    merged = merged.drop(columns=['offhour_http_count'], errors='ignore')
    if not http_raw_pd.empty:
        http_raw_pd['hour'] = http_raw_pd['date'].dt.hour
        off_http = http_raw_pd[(http_raw_pd['hour'] < WORK_START) | (http_raw_pd['hour'] >= WORK_END)]
        off_http_cnt = off_http.groupby(['user','day']).size().reset_index(name='offhour_http_count')
        merged = merged.merge(off_http_cnt, on=['user','day'], how='left')
    else:
        merged['offhour_http_count'] = 0
    merged['offhour_http_count'] = merged['offhour_http_count'].fillna(0).astype(int)
    merged['offhour_http_flag'] = (merged['offhour_http_count'] > 0).astype(int)
    merged['offhour_logon_flag'] = (merged.get('offhour_logonlogoff_count',0) > 0).astype(int)
    merged['offhour_usb_flag']   = (merged.get('afterhours_usb_connects',0) > 0).astype(int)

    merged = merged.sort_values(['user','day']).reset_index(drop=True)
    merged['offhour_http_frac'] = (merged['offhour_http_count'] / merged['total_http_requests'].replace(0, np.nan)).fillna(0).astype(float)

    # Attach scenario and derive is_malicious once
    scenarios = process_scenarios(answers_path)
    if not scenarios.empty:
        scenarios['day'] = pd.to_datetime(scenarios['day']).dt.floor('D')
    
    # Remove all rows for users that ever appear in scenario 2 or 3
    if not scenarios.empty:
        bad_users = scenarios.loc[scenarios['scenario'].isin([2, 3]), 'user'].unique()
        if len(bad_users) > 0:
            print(f"Removing {len(bad_users)} users with scenario 2 or 3")
            merged = merged[~merged['user'].isin(bad_users)]


    merged = merged.merge(scenarios, on=['user','day'], how='left')
    if 'scenario' not in merged.columns:
        merged['scenario'] = 0
    merged['scenario'] = pd.to_numeric(merged['scenario'], errors='coerce').fillna(0).astype(int)
    merged['is_malicious'] = (merged['scenario'] > 0).astype('int8')


    # -----------------------
    # Time-aware per-user z only (expanding and 7d)
    # -----------------------
    feature_cols = [
        'logon_count','logoff_count','online_duration_hours','distinct_pcs','offhour_logonlogoff_count',
        'total_http_requests','unique_domains','session_duration_hours','avg_requests_per_session',
        'usb_connect_disconnect_count','avg_usb_session_hours','total_usb_session_hours',
        'afterhours_usb_connects','offhour_http_count','offhour_http_frac','pc_switch_rate','total_activity_count'
    ]
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0.0)

    per_user = merged[['user','day'] + feature_cols].copy()
    per_user = per_user.sort_values(['user','day']).reset_index(drop=True)

    # Expanding z (per-user)
    # Expanding z (per-user)
    base_ids = per_user[['user','day']].copy()

    expand_cols = {}
    for col in feature_cols:
        mu_prev, sd_prev = per_user_expanding_stats(per_user, col)
        sd_prev = sd_prev.replace(0, np.nan)
        expand_cols[f'{col}_user_z'] = ((per_user[col] - mu_prev) / sd_prev).fillna(0.0).astype('float32')

    # Rolling z for 7d (per-user)
    windows = [7]
    roll_frames = []

    for w in windows:
        roll_cols_w = {}
        for col in feature_cols:
            roll_mu_prev, roll_sd_prev = per_user_rolling_stats(per_user, col, w)  # shift(1) is internal, so past-only
            roll_sd_prev = roll_sd_prev.replace(0, np.nan)
            roll_cols_w[f'{col}_user_z_{w}d'] = ((per_user[col] - roll_mu_prev) / roll_sd_prev).fillna(0.0).astype('float32')
        roll_frames.append(pd.DataFrame(roll_cols_w, index=per_user.index))

    per_user_out = pd.concat(
        [base_ids, pd.DataFrame(expand_cols, index=per_user.index)] + roll_frames,
        axis=1
    )


    # -----------------------
    # Time-aware global z only (expanding per day and 7d per day), attached at row level
    # -----------------------
    global_df = merged[['day'] + feature_cols].copy().sort_values('day').reset_index(drop=True)

    # 1) Build one-row-per-day expanding baselines for all features (internal only)
    day_index = pd.DataFrame({'day': pd.to_datetime(sorted(global_df['day'].unique()))})
    global_day_baselines = day_index.copy()
    for col in feature_cols:
        dstat = day_level_global_stats(global_df, col)  # ['day','global_mu_prev','global_sd_prev','count']
        global_day_baselines = global_day_baselines.merge(
            dstat[['day','global_mu_prev','global_sd_prev']].rename(
                columns={
                    'global_mu_prev': f'{col}_global_mu_prev',
                    'global_sd_prev': f'{col}_global_sd_prev'
                }
            ),
            on='day', how='left'
        )

    # 7d rolling baselines (per day)
    windows = [7]
    for col in feature_cols:
        for w in windows:
            droll = day_level_global_rolling(global_df[['day', col]], col, w)
            global_day_baselines = global_day_baselines.merge(
                droll[['day', f'global_rollmu_prev_{w}d', f'global_rollsd_prev_{w}d']].rename(
                    columns={
                        f'global_rollmu_prev_{w}d': f'{col}_global_rollmu_prev_{w}d',
                        f'global_rollsd_prev_{w}d': f'{col}_global_rollsd_prev_{w}d'
                    }
                ),
                on='day', how='left'
            )


    # 2) Attach per-day baselines to each row (many-to-one on day)
    row_level = merged[['user','day'] + feature_cols].merge(
        global_day_baselines, on='day', how='left', sort=False, validate='m:1'
    )

    # 3) Batch-compute row-level global z (expanding and 7d), do NOT emit baselines
    glob_cols = {}
    for col in feature_cols:
        # Expanding global z (past-only)
        mu = row_level[f'{col}_global_mu_prev']
        sd = row_level[f'{col}_global_sd_prev'].replace(0, np.nan)
        dev = (row_level[col] - mu).astype('float32')
        glob_cols[f'{col}_global_z'] = (dev / sd).fillna(0.0).astype('float32')

        # Rolling global z for 7d (past-only)
        for w in [7]:
            mu_r = row_level[f'{col}_global_rollmu_prev_{w}d']
            sd_r = row_level[f'{col}_global_rollsd_prev_{w}d'].replace(0, np.nan)
            glob_cols[f'{col}_global_z_{w}d'] = ((row_level[col] - mu_r) / sd_r).fillna(0.0).astype('float32')


    global_out = pd.concat([row_level[['user','day']], pd.DataFrame(glob_cols, index=row_level.index)], axis=1)

    # Keep raw values to attach later
    raw_keep_cols = ['user','day'] + feature_cols
    raw_keep = merged[raw_keep_cols].copy()

    # Flags from merged
    flags_keep = merged[['user','day','wikileaks_flag','offhour_logon_flag','offhour_usb_flag','offhour_http_flag']].copy()
    for f in ['wikileaks_flag','offhour_logon_flag','offhour_usb_flag','offhour_http_flag']:
        if f not in flags_keep.columns:
            flags_keep[f] = 0
        flags_keep[f] = pd.to_numeric(flags_keep[f], errors='coerce').fillna(0).astype(int)

    # Labels from merged (single source of truth)
    labels_keep = merged[['user','day','scenario','is_malicious']].copy()
    labels_keep['scenario'] = pd.to_numeric(labels_keep['scenario'], errors='coerce').fillna(0).astype(int)
    labels_keep['is_malicious'] = pd.to_numeric(labels_keep['is_malicious'], errors='coerce').fillna(0).astype('int8')

    def attach_and_fill(df):
        out = df.merge(raw_keep, on=['user','day'], how='left')
        out = out.merge(flags_keep, on=['user','day'], how='left')
        out = out.merge(labels_keep, on=['user','day'], how='left')
        # Fill NaNs for early rows: raw values -> 0, z-scores already 0
        out[feature_cols] = out[feature_cols].fillna(0)
        for f in ['wikileaks_flag','offhour_logon_flag','offhour_usb_flag','offhour_http_flag','scenario','is_malicious']:
            if f in out.columns:
                out[f] = out[f].fillna(0)
        # Enforce types
        out['scenario'] = out['scenario'].astype(int)
        out['is_malicious'] = out['is_malicious'].astype('int8')
        for f in ['wikileaks_flag','offhour_logon_flag','offhour_usb_flag','offhour_http_flag']:
            out[f] = out[f].astype(int)
        return out


    per_user_out = attach_and_fill(per_user_out)
    global_out   = attach_and_fill(global_out)

    # Enforce identical ordering for both frames
    per_user_out = per_user_out.sort_values(['user','day']).reset_index(drop=True)
    global_out   = global_out.sort_values(['user','day']).reset_index(drop=True)

    # Required identifiers/flags first
    front = ['user','day','is_malicious','scenario','wikileaks_flag','offhour_logon_flag','offhour_usb_flag','offhour_http_flag']
    def reorder(df):
        existing_front = [c for c in front if c in df.columns]
        others = [c for c in df.columns if c not in existing_front]
        return df[existing_front + others]
    per_user_out = reorder(per_user_out)
    global_out   = reorder(global_out)

    print(f"Done Merge and time-aware features (z only) in {time.time() - t0:.2f} seconds")
    return per_user_out, global_out

if __name__ == '__main__':
    tstart = time.time()
    # Adjust paths as needed
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(this_dir, '..', '..', 'data', 'r4.2'))
    answers_path = os.path.join(this_dir, 'answers4.csv')

    per_user_pd, global_pd = merge_all(data_dir, answers_path, chunk_bytes=256 * 1024 * 1024)

    # Write CSVs
    out_dir = this_dir
    per_user_path = os.path.join(out_dir, 'per_user_features_timeaware_pruned11.csv')
    global_path   = os.path.join(out_dir, 'global_features_timeaware_pruned11.csv')
    per_user_pd.to_csv(per_user_path, index=False)
    global_pd.to_csv(global_path, index=False)

    print(f"✅ Time-aware per-user features (z-only + raw + flags) written to {per_user_path}")
    print(f"✅ Time-aware global features (z-only + raw + flags) written to {global_path}")
    print(f"Done Everything in {time.time() - tstart:.2f} seconds")
