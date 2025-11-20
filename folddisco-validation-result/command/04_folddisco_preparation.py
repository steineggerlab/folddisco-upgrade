#Python script to extract chain information from each pdbs correspondent to Foldmason results

from pathlib import Path
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

BASE_DIR = Path("data/classified_pdbs_3") 
DOMAIN_INFO = "domain_list.txt"
MSA_FILENAME = "result_aa.fa"
RESULT_FILE = Path("data/chain_residue_list3.txt")

MAP_3TO1 = {k.upper(): v for k, v in protein_letters_3to1.items()}
parser = PDBParser(QUIET=True)

def parse_target_idx(domain_info_path: Path):
    with domain_info_path.open("r") as f:
        lines = f.readlines()
    line2 = lines[1].strip()
    cleaned = line2.strip("[] ").replace(" ", "")
    nums = [int(x) for x in cleaned.split(",") if x]
    return nums

def load_msa(msa_path):
    msa = {}
    current_id = None
    buf = []

    with msa_path.open() as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    msa[current_id] = "".join(buf)
                current_id = line[1:].strip()
                buf = []
            else:
                buf.append(line)
        if current_id:
            msa[current_id] = "".join(buf)

    return msa

def alncol_to_seqidx(aln, col):
    if aln[col - 1] == "-":
        return None
    before = aln[:col - 1]
    return sum(1 for x in before if x != "-") + 1

def residue_list_per_chain(structure):
    model = next(structure.get_models())
    chain_res = {}
    for chain in model.get_chains():
        residues = []
        seen = set()
        for res in chain.get_residues():
            hetflag, resseq, icode = res.id
            if hetflag != " ":
                continue
            key = (resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            aa3 = res.get_resname().upper().strip()
            aa1 = MAP_3TO1.get(aa3, 'X')
            residues.append((resseq, icode.strip() or "", aa3, aa1))
        chain_res[chain.id] = residues
    return chain_res

def main():
    with RESULT_FILE.open("w") as out:
        out.write("cat_id\tpdb_id\tselected_tags\n")

    for cat_dir in sorted(BASE_DIR.glob("*")):
        domain_info = cat_dir / DOMAIN_INFO
        cat_key = cat_dir.name.replace("pdbs_", "")
        msa_dir = Path("data/foldmason_1") / cat_key
        msa_path = msa_dir / MSA_FILENAME
        if not domain_info.is_file():
            continue
        if not msa_path.exists():
            print(f"[WARN] No MSA file in {cat_dir}")
            continue

        target_idx = parse_target_idx(domain_info)
        msa = load_msa(msa_path)
        for pdb_path in sorted(cat_dir.glob("*.pdb")):
            pdb_id = pdb_path.stem

            if pdb_id not in msa:
                print(f"[WARN] {pdb_id} not found in MSA of {cat_dir.name}")
                continue

            aln = msa[pdb_id]

            structure = parser.get_structure(pdb_id, str(pdb_path))
            chains = residue_list_per_chain(structure)

            # chain selection: if multiple, choose chain with largest number of residues
            ch = max(chains.keys(), key=lambda c: len(chains[c]))
            residues = chains[ch]

            tags = []
            for col in target_idx:
                seq_idx = alncol_to_seqidx(aln, col)
                if seq_idx is None:
                    continue
                if seq_idx > len(residues):
                    continue
                resseq, icode, aa3, aa1 = residues[seq_idx - 1]
                tags.append(f"{ch}{resseq}{icode}".strip())

            tag_str = ",".join(tags)

            with RESULT_FILE.open("a") as out:
                out.write(f"{cat_dir.name}\t{pdb_id}\t{tag_str}\n")


if __name__ == "__main__":
    main()