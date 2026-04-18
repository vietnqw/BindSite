import io
import csv
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from dataclasses import dataclass
from ..core.logger import logger
from ..core.config import (
    PRO_FASTA_URLS, PEPBCL_TSV_URLS, SPRINT_STR_ZIP_URL, 
    PEP_OUTPUT_MAPPING
)
from .io import BindingRecord, parse_3line_fasta

@dataclass(frozen=True)
class LocatedBindingRecord:
    seq_id: str
    sequence: str
    labels: str
    source_file: Path
    record_index: int

def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "BindSite-cli/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()

def _download_text(url: str) -> str:
    return _download_bytes(url).decode("utf-8")

def _parse_pepbcl_tsv(content: str) -> list[tuple[str, str]]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return []
    
    pairs = []
    for line in lines[1:]: # Skip header "seq label"
        parts = line.split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs

def _parse_sprint_zip(zip_bytes: bytes) -> list[BindingRecord]:
    records = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in ("DataSet/Train.txt", "DataSet/Test.txt"):
            content = archive.read(member).decode("utf-8")
            # Reuse logic from IO if we make it public or local
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            for idx in range(0, (len(lines)//3)*3, 3):
                records.append(BindingRecord(seq_id=lines[idx][1:], sequence=lines[idx+1], labels=lines[idx+2]))
    return records

def download_benchmark_datasets(data_root: Path = Path("data")) -> int:
    """Download and build benchmark datasets under data/{PEP,PRO}/{fasta,csv,pdb}."""
    try:
        # Hierarchy setup
        for task in ("PEP", "PRO"):
            for folder in ("fasta", "csv", "pdb", "features"):
                (data_root / task / folder).mkdir(parents=True, exist_ok=True)

        logger.info("Downloading SPRINT-STR Data...")
        sprint_records = _parse_sprint_zip(_download_bytes(SPRINT_STR_ZIP_URL))
        sprint_lookup = {(r.sequence, r.labels): r.seq_id for r in sprint_records}

        logger.info("Downloading and processing PEP datasets...")
        synthetic_ids = {}
        for source_tsv, output_fasta in PEP_OUTPUT_MAPPING.items():
            content = _download_text(PEPBCL_TSV_URLS[source_tsv])
            pairs = _parse_pepbcl_tsv(content)
            
            records = []
            for seq, labels in pairs:
                key = (seq, labels)
                if key in sprint_lookup:
                    seq_id = sprint_lookup[key]
                else:
                    if key not in synthetic_ids:
                        synthetic_ids[key] = f"PEP_ID_{len(synthetic_ids) + 1:04d}"
                    seq_id = synthetic_ids[key]
                records.append(BindingRecord(seq_id=seq_id, sequence=seq, labels=labels))
            
            # Write FASTA
            fasta_path = data_root / "PEP/fasta" / output_fasta
            with open(fasta_path, "w") as f:
                for r in records:
                    f.write(f">{r.seq_id}\n{r.sequence}\n{r.labels}\n")
            logger.info(f"  Wrote {output_fasta} ({len(records)} records)")

        logger.info("Downloading PRO datasets...")
        for name, url in PRO_FASTA_URLS.items():
            content = _download_text(url)
            # Simple write-through since they are already in 3-line FASTA
            with open(data_root / "PRO/fasta" / name, "w") as f:
                f.write(content)
            logger.info(f"  Wrote {name}")

        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1

def verify_fasta_integrity(data_root: Path, tasks=("PEP", "PRO")) -> int:
    """Verify that repeated IDs have consistent sequences/labels."""
    from rich.console import Console
    console = Console()
    
    all_ok = True
    for task in tasks:
        fasta_dir = data_root / task / "fasta"
        if not fasta_dir.exists():
            console.print(f"[yellow]Warning: Fasta directory not found for {task}: {fasta_dir}[/yellow]")
            continue
        
        id_map = {} # seq_id -> (seq, labels, file)
        files = sorted(list(fasta_dir.glob("*.fa")))
        
        task_ok = True
        for f in files:
            records = parse_3line_fasta(f)
            for r in records:
                if r.seq_id in id_map:
                    prev_seq, prev_labels, prev_file = id_map[r.seq_id]
                    if r.sequence != prev_seq or r.labels != prev_labels:
                        console.print(f"[red]Error: ID Conflict in {task}: {r.seq_id} mismatched between {prev_file} and {f.name}[/red]")
                        task_ok = False
                else:
                    id_map[r.seq_id] = (r.sequence, r.labels, f.name)
        
        status_text = "[green]PASS[/green]" if task_ok else "[red]FAIL[/red]"
        console.print(f"Verification [cyan]{task}[/cyan]: Checked [blue]{len(id_map)}[/blue] unique IDs. Status: {status_text}")
        if not task_ok:
            all_ok = False
            
    return 0 if all_ok else 1

def export_fasta_to_csv(input_fasta: Path, output_csv: Path) -> int:
    """Convert DeepProSite 3-line FASTA into CSV with columns ID,sequence,label."""
    try:
        records = parse_3line_fasta(input_fasta)
        if not records:
            logger.error(f"No records found in {input_fasta}")
            return 1

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ID", "sequence", "label"])
            for record in records:
                # Format labels as a bracketed list string to match original format
                label_list_repr = "[" + ", ".join(record.labels) + "]"
                writer.writerow([record.seq_id, record.sequence, label_list_repr])

        logger.info(f"Wrote {len(records)} rows to {output_csv}")
        return 0
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return 1

def get_data_info(data_root: Path, tasks=("PEP", "PRO")) -> int:
    """Check data-related information for each task using Rich for formatting."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    try:
        for task in tasks:
            fasta_dir = data_root / task / "fasta"
            pdb_dir = data_root / task / "pdb"
            
            if not fasta_dir.exists():
                console.print(f"[yellow]Warning: Fasta directory not found for task {task}: {fasta_dir}[/yellow]")
                continue
                
            fasta_files = sorted(list(fasta_dir.glob("*.fa")))
            
            console.print(f"Data Status: [cyan]{task}[/cyan]")
            table = Table(show_header=True, header_style="magenta")
            table.add_column("Dataset File", style="cyan", ratio=2)
            table.add_column("Items", justify="right", style="blue")
            table.add_column("PDBs", justify="right", style="green")
            table.add_column("Missing", justify="right", style="red")
            
            task_all_ids = set()
            
            for f in fasta_files:
                records = parse_3line_fasta(f)
                num_items = len(records)
                ids_in_file = {r.seq_id for r in records}
                task_all_ids.update(ids_in_file)
                
                pdbs_found = 0
                if pdb_dir.exists():
                    for seq_id in ids_in_file:
                        if (pdb_dir / f"{seq_id}.pdb").exists():
                            pdbs_found += 1
                
                missing = num_items - pdbs_found
                table.add_row(f.name, str(num_items), str(pdbs_found), str(missing))

            total_items = len(task_all_ids)
            total_pdbs_found = 0
            if pdb_dir.exists():
                for seq_id in task_all_ids:
                    if (pdb_dir / f"{seq_id}.pdb").exists():
                        total_pdbs_found += 1
            
            total_missing = total_items - total_pdbs_found
            
            table.add_section()
            table.add_row("TOTAL UNIQUE", str(total_items), str(total_pdbs_found), str(total_missing))
            
            if pdb_dir.exists():
                physical_pdb_count = len(list(pdb_dir.glob("*.pdb")))
                table.add_row("Physical PDB files", "", str(physical_pdb_count), "")
            
            console.print(table)
            console.print()

        return 0
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return 1
