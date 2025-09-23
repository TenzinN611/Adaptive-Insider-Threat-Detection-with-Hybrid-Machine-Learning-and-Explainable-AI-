import os
import re
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from elasticsearch import Elasticsearch, exceptions
from elasticsearch.helpers import bulk

# --- Configuration ---
ES_URL = os.environ.get("ES_URL", "https://localhost:9200")  # 8.x uses HTTPS
ES_USER = os.environ.get("ES_USER", "elastic")
ES_PASSWORD = os.environ.get("ES_PASSWORD", "V29-CjdrKXy+qDBDms+D")  # set or export ES_PASSWORD
ES_CA_CERT = os.environ.get(
    "ES_CA_CERT",
    "C:/Users/namse/Downloads/elasticsearch-9.1.3-windows-x86_64/elasticsearch-9.1.3/config/certs/http_ca.crt"
)

DATA_DIR = Path("C:/Users/namse/insider-threat-detector/data/r4.2")
LDAP_DIR = DATA_DIR / "LDAP"
# --- 1. Define Mappings (Schema) for each index ---
MAPPINGS = {
    "cert-logon": {
        "properties": {
            "id": {"type": "keyword"},
            "date": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
            "user": {"type": "keyword"},
            "pc": {"type": "keyword"},
            "activity": {"type": "keyword"}
        }
    },
    "cert-device": {
        "properties": {
            "id": {"type": "keyword"},
            "date": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
            "user": {"type": "keyword"},
            "pc": {"type": "keyword"},
            "activity": {"type": "keyword"}
        }
    },
    "cert-http": {
        "properties": {
            "id": {"type": "keyword"},
            "date": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
            "user": {"type": "keyword"},
            "pc": {"type": "keyword"},
            "url": {"type": "keyword"},
            "content": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 2048}}}
        }
    },
    "cert-email": {
        "properties": {
            "id": {"type": "keyword"},
            "date": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
            "user": {"type": "keyword"},
            "pc": {"type": "keyword"},
            "to": {"type": "keyword"},
            "cc": {"type": "keyword"},
            "bcc": {"type": "keyword"},
            "from": {"type": "keyword"},
            "size": {"type": "long"},
            "attachments": {"type": "long"},
            "content": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 2048}}}
        }
    },
    "cert-file": {
        "properties": {
            "id": {"type": "keyword"},
            "date": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
            "user": {"type": "keyword"},
            "pc": {"type": "keyword"},
            "filename": {"type": "keyword"},
            "content": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 1024}}}
        }
    },
    "certp-psychometric": {
        "properties": {
            "employee_name": {"type": "keyword"},
            "user_id": {"type": "keyword"},
            "O": {"type": "float"}, "C": {"type": "float"},
            "E": {"type": "float"}, "A": {"type": "float"},
            "N": {"type": "float"}
        }
    },
    # Explicit LDAP mapping
    "certl-ldap": {
        "properties": {
            "employee_name":   {"type": "keyword"},
            "user":            {"type": "keyword"},   # keep if some files use 'user'
            "user_id":         {"type": "keyword"},   # primary LDAP user key
            "email":           {"type": "keyword"},
            "role":            {"type": "keyword"},
            "business_unit":   {"type": "keyword"},
            "functional_unit": {"type": "keyword"},
            "department":      {"type": "keyword"},
            "team":            {"type": "keyword"},
            "supervisor":      {"type": "keyword"},
            "snapshot_date":   {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
        }
    }
}

# --- 2. Map CSV filenames to their index and column names ---
FILES_TO_INDEX = {
    'logon.csv': {'index': 'cert-logon', 'user_col': 'user'},
    'device.csv': {'index': 'cert-device', 'user_col': 'user'},
    'file.csv': {'index': 'cert-file', 'user_col': 'user'},
    'http.csv': {'index': 'cert-http', 'user_col': 'user'},
    'email.csv': {'index': 'cert-email', 'user_col': 'user'},
    'psychometric.csv': {'index': 'certp-psychometric', 'user_col': 'user_id'}
}

# --- Utilities ---
def to_iso_utc(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors='coerce', utc=True)
    out = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").str[:-4]
    return out.where(~ts.isna(), None)

def split_multi(val):
    if pd.isna(val):
        return None
    parts = re.split(r'[;|,]\s*', str(val))
    return [p for p in parts if p]

def normalize_email(val):
    if pd.isna(val):
        return None
    s = str(val)
    m = re.search(r'\((?:mailto:)?([^)>\s]+)\)', s)
    if m:
        return m.group(1)
    s = re.sub(r'[<>]', '', s)
    s = re.sub(r'^(?:mailto:)', '', s, flags=re.I)
    return s.strip()

# --- Bulk helper ---
def generate_actions(df, index_name):
    for _, row in df.iterrows():
        yield {"_index": index_name, "_source": row.to_dict()}

# --- LDAP per-month indexing helpers ---
def month_from_filename(fname: str) -> str:
    return fname.replace(".csv", "")

def snapshot_iso_utc(fname: str) -> str:
    dt = datetime.strptime(month_from_filename(fname), "%Y-%m")
    return pd.Timestamp(dt, tz="UTC").isoformat()

def create_index_if_missing(es_client, index_name, mapping_key):
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, mappings=MAPPINGS[mapping_key])
        print(f"Created index {index_name} with mapping '{mapping_key}'")

def ingest_ldap_monthly_split_indices(es_client):
    monthly_files = sorted([p for p in LDAP_DIR.glob("*.csv") if re.match(r"\d{4}-\d{2}\.csv", p.name)])
    if not monthly_files:
        print(f"No LDAP monthly CSVs found in {LDAP_DIR}")
        return

    added = []
    for path in monthly_files:
        month_key  = month_from_filename(path.name)           # e.g., "2010-01"
        index_name = f"certlldap-{month_key}".lower()         # e.g., "certl-ldap-2010-01"
        create_index_if_missing(es_client, index_name, "certl-ldap")

        for i, chunk in enumerate(pd.read_csv(path, chunksize=10000, on_bad_lines="warn")):
            # Ensure both 'user' and 'user_id' exist where possible
            if "user_id" not in chunk.columns and "user" in chunk.columns:
                chunk["user_id"] = chunk["user"]
            if "user" not in chunk.columns and "user_id" in chunk.columns:
                chunk["user"] = chunk["user_id"]

            # Normalize email
            if "email" in chunk.columns:
                chunk["email"] = chunk["email"].map(normalize_email)

            # Attach month snapshot date
            chunk["snapshot_date"] = snapshot_iso_utc(path.name)

            # JSON-safe
            chunk = chunk.where(pd.notnull(chunk), None)

            def actions():
                for _, row in chunk.iterrows():
                    yield {"_index": index_name, "_source": row.to_dict()}

            ok, errors = bulk(
                es_client,
                actions(),
                request_timeout=120,
                raise_on_error=False
            )
            err_count = len(errors) if isinstance(errors, list) else int(errors or 0)
            print(f"{index_name} chunk {i+1}: ok={ok}, errors={err_count}")
            if err_count:
                print(f"  first error: {errors}")
        added.append(index_name)

    # Optional alias for convenience
    if added:
        actions = [{"add": {"index": idx, "alias": "certl-ldap"}} for idx in added]
        es_client.indices.update_aliases(body={"actions": actions})
        print(f"Alias 'certl-ldap' updated for {len(added)} indices")

# --- Main Script ---
def main():
    print("Connecting to Elasticsearch (HTTPS + CA)...")
    es_client = Elasticsearch(
        ES_URL,
        ca_certs=ES_CA_CERT,
        basic_auth=(ES_USER, ES_PASSWORD),
        request_timeout=60,
    )

    retries = 5
    for i in range(retries):
        if es_client.ping():
            print("Connection successful.")
            break
        print(f"Connection failed. Retrying in 5 seconds... ({i+1}/{retries})")
        time.sleep(5)
    else:
        raise ConnectionError("Could not connect to Elasticsearch after several retries.")

    # 1) LDAP per-month ingestion -> certl-ldap-YYYY-MM
    ingest_ldap_monthly_split_indices(es_client)

    # 2) Event and psychometric ingestion
    for filename, info in FILES_TO_INDEX.items():
        index_name = info['index']
        user_col_original = info['user_col']
        file_path = DATA_DIR / filename

        if not file_path.exists():
            print(f"File not found: {file_path}. Skipping.")
            continue

        print("-" * 50)
        print(f"Processing {filename} into index '{index_name}'")

        # Create index if needed
        try:
            if not es_client.indices.exists(index=index_name):
                print(f"Index '{index_name}' not found. Creating with mapping...")
                es_client.indices.create(index=index_name, mappings=MAPPINGS[index_name])
                print("Index created successfully.")
            else:
                print(f"Index '{index_name}' already exists. Skipping mapping creation.")
        except exceptions.RequestError as e:
            print(f"Error creating index {index_name}: {e}")
            if 'resource_already_exists_exception' not in str(e):
                continue

        # Read CSV and bulk ingest
        print(f"Reading {filename} and ingesting data...")
        try:
            chunk_size = 10000
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, on_bad_lines='warn')):
                # Keep 'user' for event indices since mappings expect 'user'; ensure presence if column differs
                if user_col_original != 'user' and 'user' not in chunk.columns and user_col_original in chunk.columns:
                    chunk.rename(columns={user_col_original: 'user'}, inplace=True)

                # For psychometric keep 'user_id' as mapped
                if index_name == 'certp-psychometric' and 'user_id' not in chunk.columns and 'user' in chunk.columns:
                    chunk.rename(columns={'user': 'user_id'}, inplace=True)

                # Date to ISO 8601 strings
                if 'date' in chunk.columns:
                    chunk['date'] = to_iso_utc(chunk['date'])

                # Split multi-recipient email fields into arrays
                if index_name == 'cert-email':
                    for col in ['to', 'cc', 'bcc']:
                        if col in chunk.columns:
                            chunk[col] = chunk[col].map(split_multi)

                # JSON-safe
                chunk = chunk.where(pd.notnull(chunk), None)

                success_count, errors = bulk(
                    es_client,
                    generate_actions(chunk, index_name),
                    request_timeout=120,
                    raise_on_error=False
                )
                err_count = len(errors) if isinstance(errors, list) else int(errors or 0)
                print(f"  - Chunk {i+1}: Indexed {success_count} documents, {err_count} errors.")
                if err_count:
                    print(f"    - First error: {errors}")
        except Exception as e:
            print(f"An error occurred during bulk indexing of {filename}: {e}")

    print("-" * 50)
    print("Script finished.")

if __name__ == "__main__":
    main()
