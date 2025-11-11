#Python script to extract chain information from each pdbs correspondent to Foldmason results

from pathlib import Path
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

BASE_DIR = Path("data/classified_pdbs") 
DOMAIN_INFO = "domain_list.txt"
RESULT_FILE = Path("data/chain_residue_list.txt")

MAP_3TO1 = {k.upper(): v for k, v in protein_letters_3to1.items()}
parser = PDBParser(QUIET=True)

def parse_target_idx(domain_info_path: Path):
    with domain_info_path.open("r") as f:
        lines = f.readlines()
    line2 = lines[1].strip()
    cleaned = line2.strip("[] ").replace(" ", "")
    nums = [int(x) for x in cleaned.split(",") if x]
    return nums

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
        if not domain_info.is_file():
            continue

        target_idx = parse_target_idx(domain_info)
        for pdb_path in sorted(cat_dir.glob("*.pdb")):
            structure = parser.get_structure(pdb_path.stem, str(pdb_path))
            chains = residue_list_per_chain(structure)

            with RESULT_FILE.open("a") as out:
                for ch, residues in chains.items():
                    tags = []
                    for idx in target_idx:
                        if 1 <= idx <= len(residues):
                            resseq, icode, aa3, aa1 = residues[idx - 1]
                            tags.append(f"{ch}{resseq}{icode}".strip())
                        else:
                            continue
                    tag_str = ",".join(tags)
                    out.write(f"{cat_dir.name}\t{pdb_path.stem}\t{tag_str}\n")

if __name__ == "__main__":
    main()