#!/usr/bin/env python3
"""
Randomly shuffle protein side chains within mmCIF (.cif / .cif.gz) files.

What it does
------------
For each structure, the backbone atoms (N, CA, C, O, OXT) stay exactly where
they are. The side chains (every other atom, including CB) are randomly
permuted among the residues. When a side chain is moved onto a new backbone,
it is rigidly transplanted by superimposing the donor residue's N-CA-C frame
onto the acceptor's N-CA-C frame, so the side chain's internal geometry is
preserved and it sits naturally on its new backbone. The residue name moves
with the side chain (e.g. a LEU side chain landing on position 5 makes
position 5 a LEU).

Only the 20 standard amino acids are shuffled; anything else (HETATM,
modified residues, ligands, waters) is left untouched and in place. Every
other mmCIF category (pLDDT / ma_qa_metric, headers, entities, ...) is
preserved because only the _atom_site loop is rewritten.

Usage
-----
Test on ONE file, writing the copy to a separate directory:
    python3 shuffle_sidechains.py one.cif --out-dir /tmp/shuffled --workers 1

Full run over a huge list (avoids ARG_MAX), overwriting in place:
    find /path/to/data -maxdepth 1 -name '*.cif' > files.txt
    python3 shuffle_sidechains.py --file-list files.txt --in-place --workers 16

SLURM job array (recommended for 550k+ files) -- each array task processes
its own stride of the file list, using multiprocessing inside the task:
    #SBATCH --array=0-199
    #SBATCH --cpus-per-task=8
    python3 shuffle_sidechains.py --file-list files.txt --in-place
        # --num-shards / --shard-id and --workers are auto-detected from
        # SLURM_ARRAY_TASK_* and SLURM_CPUS_PER_TASK

You can also shard manually (no SLURM):
    python3 shuffle_sidechains.py --file-list files.txt --in-place \
        --num-shards 200 --shard-id 7 --workers 8

Reproducibility: each file is shuffled with a seed derived from --seed and the
file name, so a given (seed, file) always produces the same output regardless
of how the work is sharded or how many workers run.
"""
import argparse
import gzip
import hashlib
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gemmi

STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
BACKBONE = {"N", "CA", "C", "O", "OXT"}


def _tag_index(tags):
    """Map short tag name -> column index for the _atom_site loop."""
    idx = {}
    for i, t in enumerate(tags):
        idx[t.split(".", 1)[1]] = i
    return idx


def kabsch(moving, fixed):
    """Rigid transform (R, t) mapping `moving` points onto `fixed` points.
    Returns R (3x3) and t (3,) so that (R @ p) + t maps moving->fixed."""
    mc = moving.mean(axis=0)
    fc = fixed.mean(axis=0)
    M = moving - mc
    F = fixed - fc
    H = M.T @ F
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = fc - R @ mc
    return R, t


def _fmt(v):
    """Format a coordinate like mmCIF (3 decimals), avoiding -0.000."""
    s = f"{v:.3f}"
    return "0.000" if s == "-0.000" else s


def shuffle_document(doc, seed):
    """Shuffle side chains in a gemmi cif Document in place.
    Returns number of standard-AA residues involved in the shuffle."""
    block = doc.sole_block()
    cat = block.find_mmcif_category("_atom_site.")
    tags = list(cat.tags)
    ix = _tag_index(tags)

    for req in ("group_PDB", "label_atom_id", "label_comp_id",
                "Cartn_x", "Cartn_y", "Cartn_z", "auth_seq_id", "auth_asym_id"):
        if req not in ix:
            raise ValueError(f"missing _atom_site.{req}")

    comp_cols = [ix[c] for c in ("label_comp_id", "auth_comp_id") if c in ix]
    pos_cols = [ix[c] for c in (
        "label_asym_id", "label_entity_id", "label_seq_id",
        "pdbx_PDB_ins_code", "auth_seq_id", "auth_asym_id",
        "pdbx_PDB_model_num") if c in ix]
    cx, cy, cz = ix["Cartn_x"], ix["Cartn_y"], ix["Cartn_z"]

    rows = [list(cat[r]) for r in range(len(cat))]

    key_cols = [ix["auth_asym_id"], ix["auth_seq_id"]]
    if "pdbx_PDB_ins_code" in ix:
        key_cols.append(ix["pdbx_PDB_ins_code"])
    if "pdbx_PDB_model_num" in ix:
        key_cols.append(ix["pdbx_PDB_model_num"])

    order = []
    groups = {}
    for row in rows:
        k = tuple(row[c] for c in key_cols)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(row)

    def coords(row):
        return np.array([float(row[cx]), float(row[cy]), float(row[cz])])

    def split(grp):
        bb_rows, sc_rows = [], []
        for row in grp:
            (bb_rows if row[ix["label_atom_id"]] in BACKBONE else sc_rows).append(row)
        return bb_rows, sc_rows

    shufflable = []
    frames = {}
    for pos, k in enumerate(order):
        grp = groups[k]
        comp = grp[0][ix["label_comp_id"]]
        if grp[0][ix["group_PDB"]] != "ATOM" or comp not in STD_AA:
            continue
        bb = {row[ix["label_atom_id"]]: row for row in grp
              if row[ix["label_atom_id"]] in ("N", "CA", "C")}
        if not {"N", "CA", "C"} <= bb.keys():
            continue
        frames[k] = np.array([coords(bb["N"]), coords(bb["CA"]), coords(bb["C"])])
        shufflable.append(pos)

    n_std = len(shufflable)
    if n_std < 2:
        return 0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_std)

    donor_info = {}
    for pos in shufflable:
        k = order[pos]
        comp = groups[k][0][ix["label_comp_id"]]
        _, sc = split(groups[k])
        donor_info[pos] = (comp, sc)

    new_groups = dict(groups)
    for i, pos in enumerate(shufflable):
        tgt_key = order[pos]
        donor_pos = shufflable[perm[i]]
        donor_comp, donor_sc = donor_info[donor_pos]

        tgt_bb, _ = split(groups[tgt_key])
        template = groups[tgt_key][0]
        new_rows = []
        for row in tgt_bb:
            r = list(row)
            for c in comp_cols:
                r[c] = donor_comp
            new_rows.append(r)

        if donor_sc:
            R, t = kabsch(frames[order[donor_pos]], frames[tgt_key])
            for row in donor_sc:
                r = list(row)
                p = coords(row)
                q = (R @ p) + t
                r[cx] = _fmt(q[0]); r[cy] = _fmt(q[1]); r[cz] = _fmt(q[2])
                for c in comp_cols:
                    r[c] = donor_comp
                for c in pos_cols:
                    r[c] = template[c]
                new_rows.append(r)
        new_groups[tgt_key] = new_rows

    id_col = ix.get("id")
    out_cols = [[] for _ in tags]
    next_id = 1
    for k in order:
        for row in new_groups[k]:
            if id_col is not None:
                row = list(row)
                row[id_col] = str(next_id)
                next_id += 1
            for c in range(len(tags)):
                out_cols[c].append(row[c])

    cat.loop.set_all_values(out_cols)
    return n_std


def process_file(path, out_path, seed_base):
    seed = int(hashlib.sha256(
        f"{seed_base}:{os.path.basename(path)}".encode()).hexdigest()[:16], 16)
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            doc = gemmi.cif.read_string(fh.read())
    else:
        doc = gemmi.cif.read(path)

    n_shuf = shuffle_document(doc, seed)
    out_text = doc.as_string()

    d = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        if out_path.endswith(".gz"):
            with os.fdopen(fd, "wb") as fh:
                fh.write(gzip.compress(out_text.encode()))
        else:
            with os.fdopen(fd, "w") as fh:
                fh.write(out_text)
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return n_shuf


def _worker(args):
    path, out_path, seed_base = args
    try:
        n = process_file(path, out_path, seed_base)
        return (path, True, f"{n} residues shuffled")
    except Exception as e:  # noqa: BLE001
        return (path, False, repr(e))


def gather_inputs(inputs, file_list, pattern):
    print(f"gathering inputs from {inputs} + {file_list} (pattern={pattern})", flush=True)
    import fnmatch
    paths = []
    if file_list:
        with open(file_list) as fh:
            paths += [ln.strip() for ln in fh if ln.strip()]
    for inp in inputs:
        if os.path.isdir(inp):
            for root, _, files in os.walk(inp):
                for f in files:
                    if fnmatch.fnmatch(f, pattern):
                        paths.append(os.path.join(root, f))
        elif inp:
            paths.append(inp)
    return paths


def _env_int(name):
    v = os.environ.get(name)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def resolve_sharding(num_shards, shard_id):
    """Resolve (num_shards, shard_id) from explicit args, falling back to
    SLURM job-array environment variables. Returns 0-based (shards, id)."""
    print(f"sharding: num_shards={num_shards} shard_id={shard_id}", flush=True)
    if num_shards <= 0:
        tmin = _env_int("SLURM_ARRAY_TASK_MIN")
        tmax = _env_int("SLURM_ARRAY_TASK_MAX")
        cnt = _env_int("SLURM_ARRAY_TASK_COUNT")
        if tmin is not None and tmax is not None:
            num_shards = tmax - tmin + 1
        elif cnt:
            num_shards = cnt
        else:
            num_shards = 1
    if shard_id < 0:
        tid = _env_int("SLURM_ARRAY_TASK_ID")
        tmin = _env_int("SLURM_ARRAY_TASK_MIN") or 0
        shard_id = (tid - tmin) if tid is not None else 0
    if shard_id >= num_shards:
        raise SystemExit(f"shard-id {shard_id} out of range for {num_shards} shards")
    return num_shards, shard_id


def resolve_workers(workers):
    if workers and workers > 0:
        return workers
    return _env_int("SLURM_CPUS_PER_TASK") or 1


def _report(i, total, path, good, msg, ok, fail):
    if good:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL {path}: {msg}", flush=True)
    if i % 1000 == 0 or i == total:
        print(f"  [{i}/{total}] ok={ok} fail={fail}", flush=True)
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description="Shuffle protein side chains in mmCIF files.")
    ap.add_argument("inputs", nargs="*", help="files and/or directories")
    ap.add_argument("--file-list", help="newline-separated list of file paths")
    ap.add_argument("--glob", default="*.cif", dest="pattern",
                    help="filename pattern when scanning directories (default *.cif)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in-place", action="store_true", help="overwrite originals")
    g.add_argument("--out-dir", help="write shuffled copies into this directory (flat)")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes per task (0=auto: SLURM_CPUS_PER_TASK or 1)")
    ap.add_argument("--num-shards", type=int, default=0,
                    help="split the file list into this many shards "
                         "(0=auto from SLURM_ARRAY_TASK_*)")
    ap.add_argument("--shard-id", type=int, default=-1,
                    help="0-based shard index to process (auto from SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--seed", default="0", help="global seed (default 0)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N files (0=all)")
    args = ap.parse_args()

    workers = resolve_workers(args.workers)
    num_shards, shard_id = resolve_sharding(args.num_shards, args.shard_id)

    paths = gather_inputs(args.inputs, args.file_list, args.pattern)
    paths.sort()  # stable order so sharding is deterministic across tasks
    if num_shards > 1:
        paths = paths[shard_id::num_shards]  # strided slice for this shard
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        sys.exit("no input files found (for this shard)")

    print(f"[shard {shard_id}/{num_shards}] found {len(paths)} file(s) to process", flush=True)

    def out_for(p):
        return p if args.in_place else os.path.join(args.out_dir, os.path.basename(p))

    tasks = [(p, out_for(p), args.seed) for p in paths]
    total = len(tasks)
    print(f"[shard {shard_id}/{num_shards}] processing {total} file(s) with "
          f"{workers} worker(s) -> {'in place' if args.in_place else args.out_dir}",
          flush=True)

    ok = fail = 0
    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            path, good, msg = _worker(t)
            ok, fail = _report(i, total, path, good, msg, ok, fail)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                path, good, msg = fut.result()
                ok, fail = _report(i, total, path, good, msg, ok, fail)

    print(f"[shard {shard_id}/{num_shards}] done: {ok} ok, {fail} failed", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()