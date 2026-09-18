#!/usr/bin/env python3
"""
Prepare random FoldDisco query-residue sets from AFDB mmCIF files.

Generated files:
  data/query_residue_list_v2.txt:
      Single list format (path \t residue_tags)
  data/folddisco_info_list_expanded_v2.txt:
      Batch format (path \t residue_tags \t output_file_path)
"""

from __future__ import annotations

import collections
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gemmi
import numpy as np


# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

CIF_DIR = Path("data/afdb_raw")

QUERY_RESIDUE_FILE = Path("data/query_residue_list_v2.txt")
LEGACY_QUERY_FILE = Path("data/chain_residue_list_v2.txt")
METADATA_FILE = Path("data/query_metadata_v2.tsv")
FOLDDISCO_INFO_FILE = Path("data/folddisco_info_list_expanded_v2.txt")
REJECTION_FILE = Path("data/query_sampling_rejections_v2.tsv")

FOLDDISCO_RESULT_DIR = Path("result/folddisco_results_raw")

MIN_QUERY_SIZE = 2
MAX_QUERY_SIZE = 32
QUERIES_PER_SIZE = 25

CA_CONTACT_CUTOFF = 12.0
MIN_PLDDT = 90.0
ALLOW_MISSING_PLDDT = False
ONE_QUERY_PER_STRUCTURE = True

REQUIRE_DISCONTINUOUS = False
MIN_SEQUENCE_SEPARATION = 0
GROWTH_ATTEMPTS = 100

RANDOM_SEED = 99999

STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
EMPTY_CIF_VALUES = {".", "?", ""}


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    auth_seq_id: int

    def folddisco_tag(self) -> str:
        return f"{self.chain}{self.auth_seq_id}"


@dataclass
class Residue:
    key: ResidueKey
    comp_id: str
    n: np.ndarray
    ca: np.ndarray
    c: np.ndarray
    plddt: Optional[float]
    chain_order: int


@dataclass
class ParsedStructure:
    path: Path
    chains: Dict[str, List[Residue]]


@dataclass
class QueryRecord:
    query_id: str
    structure_id: str
    cif_path: Path
    size: int
    residues: List[Residue]
    edge_count: int
    edge_density: float
    radius_gyration: float
    max_pairwise_ca: float
    sequence_runs: int
    max_sequence_separation: int


class ParseError(RuntimeError):
    pass


def normalize_cif_value(value: str) -> str:
    return "" if value in EMPTY_CIF_VALUES else value


def tag_index(tags: Sequence[str]) -> Dict[str, int]:
    return {tag.split(".", 1)[1]: i for i, tag in enumerate(tags)}


def structure_id_from_path(path: Path) -> str:
    return path.name[:-4] if path.name.lower().endswith(".cif") else path.stem


def altloc_priority(value: str) -> int:
    value = normalize_cif_value(value)
    if value == "":
        return 0
    if value == "A":
        return 1
    return 2


def parse_structure(path: Path) -> ParsedStructure:
    try:
        doc = gemmi.cif.read(str(path))
        block = doc.sole_block()
        cat = block.find_mmcif_category("_atom_site.")
    except Exception as exc:
        raise ParseError(f"failed_to_read:{exc}") from exc

    if len(cat) == 0:
        raise ParseError("missing_atom_site")

    tags = list(cat.tags)
    ix = tag_index(tags)
    required = {
        "group_PDB", "label_atom_id", "label_comp_id",
        "Cartn_x", "Cartn_y", "Cartn_z", "label_asym_id", "label_seq_id",
    }
    missing = sorted(required - ix.keys())
    if missing:
        raise ParseError("missing_columns:" + ",".join(missing))

    model_col = ix.get("pdbx_PDB_model_num")
    first_model: Optional[str] = None

    grouped: collections.OrderedDict[Tuple[str, str, str, str], Dict[str, Tuple[int, List[str]]]] = collections.OrderedDict()
    residue_names: Dict[Tuple[str, str, str, str], str] = {}

    for row_idx in range(len(cat)):
        row = list(cat[row_idx])
        model = row[model_col] if model_col is not None else "1"
        if first_model is None:
            first_model = model
        if model != first_model:
            continue

        if row[ix["group_PDB"]] != "ATOM":
            continue

        comp_id = row[ix["label_comp_id"]].strip().upper()
        if comp_id not in STD_AA:
            continue

        auth_chain = normalize_cif_value(row[ix["auth_asym_id"]]) if "auth_asym_id" in ix else ""
        label_chain = normalize_cif_value(row[ix["label_asym_id"]])
        chain = auth_chain or label_chain

        auth_seq = normalize_cif_value(row[ix["auth_seq_id"]]) if "auth_seq_id" in ix else ""
        label_seq = normalize_cif_value(row[ix["label_seq_id"]])
        seq_text = auth_seq or label_seq

        insertion_code = normalize_cif_value(row[ix["pdbx_PDB_ins_code"]]) if "pdbx_PDB_ins_code" in ix else ""

        key = (chain, seq_text, insertion_code, model)
        atom_name = row[ix["label_atom_id"]].strip()
        altloc = row[ix["label_alt_id"]] if "label_alt_id" in ix else ""
        priority = altloc_priority(altloc)

        if key not in grouped:
            grouped[key] = {}
            residue_names[key] = comp_id

        previous = grouped[key].get(atom_name)
        if previous is None or priority < previous[0]:
            grouped[key][atom_name] = (priority, row)

    xcol, ycol, zcol = ix["Cartn_x"], ix["Cartn_y"], ix["Cartn_z"]
    bcol = ix.get("B_iso_or_equiv")

    chains: Dict[str, List[Residue]] = collections.OrderedDict()
    chain_order: Dict[str, int] = collections.defaultdict(int)

    for raw_key, atoms in grouped.items():
        chain, seq_text, insertion_code, _model = raw_key
        if len(chain) != 1 or insertion_code:
            continue

        try:
            seq_id = int(seq_text)
        except ValueError:
            continue
        if seq_id < 0 or not {"N", "CA", "C"}.issubset(atoms):
            continue

        def coordinate(atom_name: str) -> np.ndarray:
            atom_row = atoms[atom_name][1]
            return np.array(
                [float(atom_row[xcol]), float(atom_row[ycol]), float(atom_row[zcol])],
                dtype=np.float64,
            )

        ca_row = atoms["CA"][1]
        plddt: Optional[float] = None
        if bcol is not None:
            val = normalize_cif_value(ca_row[bcol])
            if val:
                try:
                    plddt = float(val)
                except ValueError:
                    plddt = None

        residue = Residue(
            key=ResidueKey(chain=chain, auth_seq_id=seq_id),
            comp_id=residue_names[raw_key],
            n=coordinate("N"),
            ca=coordinate("CA"),
            c=coordinate("C"),
            plddt=plddt,
            chain_order=chain_order[chain],
        )
        chain_order[chain] += 1
        chains.setdefault(chain, []).append(residue)

    if not chains:
        raise ParseError("no_compatible_residues")

    return ParsedStructure(path=path, chains=chains)


def eligible_residues(structure: ParsedStructure) -> Dict[str, List[Residue]]:
    result: Dict[str, List[Residue]] = {}
    for chain, residues in structure.chains.items():
        selected = [r for r in residues if (r.plddt is not None and r.plddt >= MIN_PLDDT) or (r.plddt is None and ALLOW_MISSING_PLDDT)]
        if selected:
            result[chain] = selected
    return result


def build_adjacency(residues: Sequence[Residue]) -> List[Set[int]]:
    adjacency: List[Set[int]] = [set() for _ in residues]
    if len(residues) < 2:
        return adjacency

    coords = np.vstack([r.ca for r in residues])
    cells = np.floor(coords / CA_CONTACT_CUTOFF).astype(np.int64)
    bins: Dict[Tuple[int, int, int], List[int]] = collections.defaultdict(list)

    for idx, cell in enumerate(cells):
        bins[tuple(int(v) for v in cell)].append(idx)

    cutoff_squared = CA_CONTACT_CUTOFF ** 2
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    for i, cell in enumerate(cells):
        base = tuple(int(v) for v in cell)
        for dx, dy, dz in offsets:
            neighbor = (base[0] + dx, base[1] + dy, base[2] + dz)
            for j in bins.get(neighbor, []):
                if j <= i:
                    continue
                delta = coords[i] - coords[j]
                if float(np.dot(delta, delta)) <= cutoff_squared:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
    return adjacency


def connected_components(adjacency: Sequence[Set[int]]) -> List[List[int]]:
    seen: Set[int] = set()
    components: List[List[int]] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in adjacency[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        components.append(comp)
    return components


def sequence_run_count(residues: Sequence[Residue]) -> int:
    if not residues:
        return 0
    ordered = sorted(residues, key=lambda r: r.chain_order)
    runs = 1
    for prev, curr in zip(ordered, ordered[1:]):
        if curr.chain_order != prev.chain_order + 1:
            runs += 1
    return runs


def max_sequence_separation(residues: Sequence[Residue]) -> int:
    if len(residues) < 2:
        return 0
    pos = [r.chain_order for r in residues]
    return max(pos) - min(pos)


def satisfies_sequence_constraints(residues: Sequence[Residue]) -> bool:
    if REQUIRE_DISCONTINUOUS and sequence_run_count(residues) < 2:
        return False
    if MIN_SEQUENCE_SEPARATION > 0 and max_sequence_separation(residues) < MIN_SEQUENCE_SEPARATION:
        return False
    return True


def sample_connected_query(
    residues_by_chain: Dict[str, List[Residue]],
    query_size: int,
    rng: random.Random,
) -> Optional[List[Residue]]:
    candidates = []
    for residues in residues_by_chain.values():
        if len(residues) < query_size:
            continue
        adj = build_adjacency(residues)
        for comp in connected_components(adj):
            if len(comp) >= query_size:
                candidates.append((residues, adj, comp))

    if not candidates:
        return None

    for _ in range(GROWTH_ATTEMPTS):
        residues, adj, comp = rng.choice(candidates)
        comp_set = set(comp)
        seed = rng.choice(comp)
        selected = {seed}
        frontier = set(adj[seed]) & comp_set

        while len(selected) < query_size and frontier:
            node = rng.choice(tuple(frontier))
            frontier.remove(node)
            selected.add(node)
            frontier.update((adj[node] & comp_set) - selected)

        if len(selected) != query_size:
            continue

        result = [residues[i] for i in sorted(selected, key=lambda x: residues[x].chain_order)]
        if satisfies_sequence_constraints(result):
            return result
    return None


def query_geometry(residues: Sequence[Residue]) -> Tuple[int, float, float, float]:
    coords = np.vstack([r.ca for r in residues])
    edge_count = 0
    max_dist = 0.0
    for i in range(len(residues)):
        for j in range(i + 1, len(residues)):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            max_dist = max(max_dist, d)
            if d <= CA_CONTACT_CUTOFF:
                edge_count += 1
    num_pairs = len(residues) * (len(residues) - 1) // 2
    density = edge_count / num_pairs if num_pairs else 0.0
    rg = math.sqrt(float(np.mean(np.sum((coords - coords.mean(axis=0)) ** 2, axis=1))))
    return edge_count, density, rg, max_dist


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_output_files(records: Sequence[QueryRecord]) -> None:
    for path in (QUERY_RESIDUE_FILE, LEGACY_QUERY_FILE, METADATA_FILE, FOLDDISCO_INFO_FILE, REJECTION_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
    FOLDDISCO_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) query_residue_list_v2.txt: 2열 (cif_path \t tags) - 헤더 없이 기록
    with QUERY_RESIDUE_FILE.open("w") as out:
        for record in records:
            tags = ",".join(r.key.folddisco_tag() for r in record.residues)
            out.write(f"{record.cif_path}\t{tags}\n")

    # 2) chain_residue_list_v2.txt: 3열 레거시 (data/afdb_raw \t AF-....cif \t tags)
    with LEGACY_QUERY_FILE.open("w") as out:
        for record in records:
            tags = ",".join(r.key.folddisco_tag() for r in record.residues)
            out.write(f"data/afdb_raw\t{record.cif_path.name}\t{tags}\n")

    # 3) folddisco_info_list_expanded_v2.txt: 3열 배치 (cif_path \t tags \t output_path)
    # FoldDisco 가 각 쿼리 결과를 독립 파일로 저장하도록 유도
    with FOLDDISCO_INFO_FILE.open("w") as out:
        for record in records:
            tags = ",".join(r.key.folddisco_tag() for r in record.residues)
            out_file = FOLDDISCO_RESULT_DIR / f"output_{sanitize_filename(record.query_id)}.txt"
            out.write(f"{record.cif_path}\t{tags}\t{out_file}\n")

    metadata_cols = [
        "query_id", "pdb_id", "query_size", "cif_path", "chain", "selected_tags",
        "residue_names", "min_plddt", "mean_plddt", "edge_count", "edge_density",
        "radius_gyration", "max_pairwise_ca", "sequence_runs", "max_sequence_separation",
    ]
    with METADATA_FILE.open("w") as out:
        out.write("\t".join(metadata_cols) + "\n")
        for record in records:
            tags = ",".join(r.key.folddisco_tag() for r in record.residues)
            names = ",".join(r.comp_id for r in record.residues)
            plddts = [r.plddt for r in record.residues if r.plddt is not None]
            min_p = min(plddts) if plddts else float("nan")
            mean_p = float(np.mean(plddts)) if plddts else float("nan")
            row = [
                record.query_id, record.structure_id, str(record.size), str(record.cif_path),
                record.residues[0].key.chain, tags, names, f"{min_p:.3f}", f"{mean_p:.3f}",
                str(record.edge_count), f"{record.edge_density:.6f}", f"{record.radius_gyration:.6f}",
                f"{record.max_pairwise_ca:.6f}", str(record.sequence_runs), str(record.max_sequence_separation),
            ]
            out.write("\t".join(row) + "\n")


def main() -> None:
    if not CIF_DIR.is_dir():
        raise SystemExit(f"CIF directory does not exist: {CIF_DIR}")

    cif_paths = sorted(CIF_DIR.glob("*.cif"))
    if not cif_paths:
        raise SystemExit(f"No .cif files found in {CIF_DIR}")

    needed = (MAX_QUERY_SIZE - MIN_QUERY_SIZE + 1) * QUERIES_PER_SIZE
    if ONE_QUERY_PER_STRUCTURE and len(cif_paths) < needed:
        raise SystemExit(f"Need at least {needed} CIFs, but found {len(cif_paths)}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(cif_paths)

    tasks = [q_size for q_size in range(MAX_QUERY_SIZE, MIN_QUERY_SIZE - 1, -1) for _ in range(QUERIES_PER_SIZE)]
    records: List[QueryRecord] = []
    rejection_counts: collections.Counter[str] = collections.Counter()
    used_paths: Set[Path] = set()
    cursor = 0

    for task_no, q_size in enumerate(tasks, start=1):
        sampled = False
        while cursor < len(cif_paths):
            cif_p = cif_paths[cursor]
            cursor += 1

            if ONE_QUERY_PER_STRUCTURE and cif_p in used_paths:
                continue

            try:
                struct = parse_structure(cif_p)
            except ParseError as exc:
                rejection_counts[str(exc)] += 1
                continue

            elig = eligible_residues(struct)
            if max((len(v) for v in elig.values()), default=0) < q_size:
                rejection_counts[f"insufficient_residues_for_k{q_size}"] += 1
                continue

            sel = sample_connected_query(elig, q_size, rng)
            if sel is None:
                rejection_counts[f"no_connected_query_for_k{q_size}"] += 1
                continue

            s_id = structure_id_from_path(cif_p)
            q_id = f"{s_id}_k{q_size:02d}"
            e_cnt, e_dens, rg, max_d = query_geometry(sel)

            records.append(
                QueryRecord(
                    query_id=q_id, structure_id=s_id, cif_path=cif_p, size=q_size,
                    residues=sel, edge_count=e_cnt, edge_density=e_dens,
                    radius_gyration=rg, max_pairwise_ca=max_d,
                    sequence_runs=sequence_run_count(sel), max_sequence_separation=max_sequence_separation(sel),
                )
            )
            used_paths.add(cif_p)
            sampled = True
            break

        if not sampled:
            with REJECTION_FILE.open("w") as out:
                out.write("reason\tcount\n")
                for r, c in rejection_counts.most_common():
                    out.write(f"{r}\t{c}\n")
            raise SystemExit(f"Unable to produce query {task_no}/{len(tasks)} for size {q_size}")

        if task_no % 100 == 0 or task_no == len(tasks):
            print(f"Prepared {task_no}/{len(tasks)} queries (size={q_size})", flush=True)

    records.sort(key=lambda r: (r.size, r.query_id))
    write_output_files(records)

    with REJECTION_FILE.open("w") as out:
        out.write("reason\tcount\n")
        for r, c in rejection_counts.most_common():
            out.write(f"{r}\t{c}\n")

    print(f"Successfully prepared {len(records)} FoldDisco queries.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)