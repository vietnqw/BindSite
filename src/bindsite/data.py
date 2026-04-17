from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


PRO_FASTA_URLS = {
    "PRO_Train_335.fa": "https://raw.githubusercontent.com/WeiLab-Biology/DeepProSite/main/DeepProSite-main/datasets/Train_335.fa",
    "PRO_Test_60.fa": "https://raw.githubusercontent.com/WeiLab-Biology/DeepProSite/main/DeepProSite-main/datasets/Test_60.fa",
    "PRO_Test_315.fa": "https://raw.githubusercontent.com/WeiLab-Biology/DeepProSite/main/DeepProSite-main/datasets/Test_315.fa",
}

PEPBCL_TSV_URLS = {
    "Dataset1_train.tsv": "https://raw.githubusercontent.com/Ruheng-W/PepBCL/master/data/Dataset1_train.tsv",
    "Dataset1_test.tsv": "https://raw.githubusercontent.com/Ruheng-W/PepBCL/master/data/Dataset1_test.tsv",
    "Dataset2_train.tsv": "https://raw.githubusercontent.com/Ruheng-W/PepBCL/master/data/Dataset2_train.tsv",
    "Dataset2_test.tsv": "https://raw.githubusercontent.com/Ruheng-W/PepBCL/master/data/Dataset2_test.tsv",
}

SPRINT_STR_ZIP_URL = "https://raw.githubusercontent.com/GTaherzadeh/SPRINT-STR/master/Data.zip"

PEP_OUTPUT_MAPPING = {
    "Dataset1_train.tsv": "PEP_Train_1154.fa",
    "Dataset1_test.tsv": "PEP_Test_125.fa",
    "Dataset2_train.tsv": "PEP_Train_640.fa",
    "Dataset2_test.tsv": "PEP_Test_639.fa",
}

EXPECTED_COUNTS = {
    "PRO_Train_335.fa": 335,
    "PRO_Test_60.fa": 60,
    "PRO_Test_315.fa": 315,
    "PEP_Train_1154.fa": 1154,
    "PEP_Test_125.fa": 125,
    "PEP_Train_640.fa": 640,
    "PEP_Test_639.fa": 639,
}


@dataclass(frozen=True)
class BindingRecord:
    seq_id: str
    sequence: str
    labels: str


@dataclass(frozen=True)
class LocatedBindingRecord:
    seq_id: str
    sequence: str
    labels: str
    source_file: Path
    record_index: int


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BindSite-data-cli/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _download_text(url: str) -> str:
    return _download_bytes(url).decode("utf-8")


def _ensure_data_hierarchy(data_root: Path) -> None:
    for task in ("PEP", "PRO"):
        for leaf in ("fasta", "csv", "pdb"):
            (data_root / task / leaf).mkdir(parents=True, exist_ok=True)


def _clear_directory_files(directory: Path) -> None:
    for entry in directory.iterdir():
        if entry.is_file():
            entry.unlink()


def _validate_sequence_and_labels(sequence: str, labels: str) -> None:
    if len(sequence) != len(labels):
        raise ValueError("Sequence/label lengths do not match.")
    if not set(labels).issubset({"0", "1"}):
        raise ValueError("Label string must contain only 0/1.")


def _parse_three_line_fasta(content: str) -> list[BindingRecord]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) % 3 != 0:
        raise ValueError("Expected 3-line FASTA format (header/sequence/labels).")

    records: list[BindingRecord] = []
    for idx in range(0, len(lines), 3):
        header = lines[idx]
        sequence = lines[idx + 1]
        labels = lines[idx + 2]
        if not header.startswith(">"):
            raise ValueError(f"Invalid FASTA header: {header}")
        _validate_sequence_and_labels(sequence, labels)
        records.append(BindingRecord(seq_id=header[1:], sequence=sequence, labels=labels))
    return records


def _parse_pepbcl_tsv(content: str) -> list[tuple[str, str]]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise ValueError("TSV content is empty.")
    header_parts = lines[0].lower().split()
    if header_parts != ["seq", "label"]:
        raise ValueError("Unexpected TSV header. Expected 'seq label'.")

    pairs: list[tuple[str, str]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Unexpected TSV row format: {line}")
        sequence, labels = parts
        _validate_sequence_and_labels(sequence, labels)
        pairs.append((sequence, labels))
    return pairs


def _parse_sprint_zip(zip_bytes: bytes) -> list[BindingRecord]:
    records: list[BindingRecord] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in ("DataSet/Train.txt", "DataSet/Test.txt"):
            content = archive.read(member).decode("utf-8")
            records.extend(_parse_three_line_fasta(content))
    return records


def _build_sprint_lookup(records: list[BindingRecord]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for record in records:
        key = (record.sequence, record.labels)
        if key in lookup and lookup[key] != record.seq_id:
            raise ValueError(f"Conflicting IDs for sequence-label key: {record.seq_id}")
        lookup[key] = record.seq_id
    return lookup


def _assign_pep_ids(
    pairs: list[tuple[str, str]],
    sprint_lookup: dict[tuple[str, str], str],
    synthetic_ids: dict[tuple[str, str], str],
) -> list[BindingRecord]:
    assigned: list[BindingRecord] = []
    for sequence, labels in pairs:
        key = (sequence, labels)
        if key in sprint_lookup:
            seq_id = sprint_lookup[key]
        else:
            if key not in synthetic_ids:
                synthetic_ids[key] = f"PEP_ID_{len(synthetic_ids) + 1:04d}"
            seq_id = synthetic_ids[key]
        assigned.append(BindingRecord(seq_id=seq_id, sequence=sequence, labels=labels))
    return assigned


def _write_three_line_fasta(path: Path, records: list[BindingRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.seq_id}\n")
            handle.write(f"{record.sequence}\n")
            handle.write(f"{record.labels}\n")


def _ensure_expected_count(filename: str, records: list[BindingRecord]) -> None:
    expected = EXPECTED_COUNTS[filename]
    if len(records) != expected:
        raise ValueError(f"{filename}: expected {expected} records but got {len(records)}.")


def download_benchmark_datasets(data_root: Path = Path("data")) -> int:
    """Download and build benchmark datasets under data/{PEP,PRO}/{fasta,csv,pdb}."""
    try:
        _ensure_data_hierarchy(data_root)
        _clear_directory_files(data_root / "PEP" / "csv")
        _clear_directory_files(data_root / "PRO" / "csv")

        pep_fasta_dir = data_root / "PEP" / "fasta"
        pro_fasta_dir = data_root / "PRO" / "fasta"

        print("Downloading SPRINT-STR Data.zip...")
        sprint_records = _parse_sprint_zip(_download_bytes(SPRINT_STR_ZIP_URL))

        print("Downloading PepBCL TSV files...")
        pepbcl_rows: dict[str, list[tuple[str, str]]] = {}
        for tsv_name, url in PEPBCL_TSV_URLS.items():
            pepbcl_rows[tsv_name] = _parse_pepbcl_tsv(_download_text(url))

        print("Building PEP FASTA files...")
        sprint_lookup = _build_sprint_lookup(sprint_records)
        synthetic_ids: dict[tuple[str, str], str] = {}
        for source_tsv, output_fasta in PEP_OUTPUT_MAPPING.items():
            records = _assign_pep_ids(pepbcl_rows[source_tsv], sprint_lookup, synthetic_ids)
            _ensure_expected_count(output_fasta, records)
            _write_three_line_fasta(pep_fasta_dir / output_fasta, records)
            print(f"  Wrote PEP/fasta/{output_fasta} ({len(records)} records)")

        print("Downloading PRO files from DeepProSite...")
        for fasta_name, url in PRO_FASTA_URLS.items():
            records = _parse_three_line_fasta(_download_text(url))
            _ensure_expected_count(fasta_name, records)
            _write_three_line_fasta(pro_fasta_dir / fasta_name, records)
            print(f"  Wrote PRO/fasta/{fasta_name} ({len(records)} records)")

        print(
            "Done. Data hierarchy ready under data/{PEP,PRO}/{fasta,csv,pdb}. "
            "Only final FASTA files are saved. "
            f"Synthetic PEP IDs assigned: {len(synthetic_ids)}."
        )
        return 0
    except (ValueError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _parse_three_line_fasta_file(path: Path) -> list[LocatedBindingRecord]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(f"{path}: expected 3-line FASTA format.")

    records: list[LocatedBindingRecord] = []
    for idx in range(0, len(lines), 3):
        header = lines[idx]
        sequence = lines[idx + 1]
        labels = lines[idx + 2]
        if not header.startswith(">"):
            raise ValueError(f"{path}: invalid FASTA header '{header}' at record {idx // 3 + 1}.")
        _validate_sequence_and_labels(sequence, labels)
        records.append(
            LocatedBindingRecord(
                seq_id=header[1:],
                sequence=sequence,
                labels=labels,
                source_file=path,
                record_index=(idx // 3) + 1,
            )
        )
    return records


def _verify_task_fasta(records: list[LocatedBindingRecord], task_name: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    first_seen: dict[str, LocatedBindingRecord] = {}
    for record in records:
        if record.seq_id not in first_seen:
            first_seen[record.seq_id] = record
            continue
        previous = first_seen[record.seq_id]
        if (record.sequence, record.labels) != (previous.sequence, previous.labels):
            errors.append(
                "ID conflict in task "
                f"{task_name}: '{record.seq_id}' has inconsistent sequence/label.\n"
                f"  First:  {previous.source_file} (record {previous.record_index})\n"
                f"  Second: {record.source_file} (record {record.record_index})"
            )
    return (len(errors) == 0, errors)


def verify_fasta_integrity(
    data_root: Path = Path("data"),
    tasks: tuple[str, ...] = ("PEP", "PRO"),
) -> int:
    """Verify repeated IDs are internally consistent within each task."""
    all_ok = True
    for task in tasks:
        fasta_dir = data_root / task / "fasta"
        if not fasta_dir.exists():
            print(f"ERROR: missing directory {fasta_dir}", file=sys.stderr)
            return 1
        files = sorted(path for path in fasta_dir.glob("*.fa") if path.is_file())
        if not files:
            print(f"ERROR: no FASTA files found in {fasta_dir}", file=sys.stderr)
            return 1

        try:
            records: list[LocatedBindingRecord] = []
            for file_path in files:
                records.extend(_parse_three_line_fasta_file(file_path))
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        ok, errors = _verify_task_fasta(records, task)
        unique_ids = len({record.seq_id for record in records})
        print(
            f"[{task}] Checked {len(records)} records across {len(files)} files. "
            f"Unique IDs: {unique_ids}."
        )
        if ok:
            print(f"[{task}] PASS: No ID conflicts found.")
        else:
            all_ok = False
            print(f"[{task}] FAIL: Found {len(errors)} ID conflicts.", file=sys.stderr)
            for err in errors:
                print(err, file=sys.stderr)

    return 0 if all_ok else 1
