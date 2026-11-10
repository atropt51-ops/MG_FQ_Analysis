"""
36_apply_S83I_mutation.py

Apply the S83I mutation (Ser83 -> Ile) to ParC structure.
Find residue 83 in the PDB file, change SER to ILE.
"""
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select

INPUT_PDB = Path("data/protein_structures/ParC_P47446.pdb")
OUTPUT_PDB = Path("data/protein_structures/ParC_S83I.pdb")
WT_PDB = Path("data/protein_structures/ParC_WT.pdb")

# Three-letter codes
SER = "SER"
ILE = "ILE"


def find_ser83(structure):
    """Locate residue 83 (Ser) in the structure."""
    print("[INFO] Searching for residue 83 (Ser)...")
    candidates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.id
                res_name = residue.get_resname()
                # residue id = (hetero_flag, seq_num, insertion_code)
                if res_id[1] == 83 and res_name == SER:
                    candidates.append((model.id, chain.id, res_id))
                    print(f"  Found: model={model.id}, chain={chain.id}, "
                          f"residue={res_name} {res_id[1]}")
    if not candidates:
        print("[WARN] No Ser83 found. Listing all residues near 83:")
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 80 <= residue.id[1] <= 86:
                        print(f"  {chain.id} {residue.get_resname()} "
                              f"{residue.id[1]}")
    return candidates


class MutateS83I(Select):
    """Select only residue 83 and change SER -> ILE."""
    def __init__(self, target_chain, target_resid):
        self.target_chain = target_chain
        self.target_resid = target_resid

    def accept_residue(self, residue):
        if (residue.get_parent().id == self.target_chain
                and residue.id[1] == self.target_resid):
            # Change the residue name to ILE
            residue.resname = ILE
        return 1


def main():
    print(f"[INFO] Loading: {INPUT_PDB}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ParC", str(INPUT_PDB))

    # Save a clean WT copy first (for later comparison)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(WT_PDB))
    print(f"[INFO] Saved WT copy: {WT_PDB}")

    candidates = find_ser83(structure)
    if not candidates:
        print("[ERROR] Could not find Ser83. Manual inspection needed.")
        return

    # Use first candidate
    model_id, chain_id, res_id = candidates[0]
    print(f"\n[INFO] Mutating {chain_id} SER{res_id[1]} -> ILE")

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(OUTPUT_PDB), MutateS83I(chain_id, res_id[1]))
    print(f"[INFO] Saved mutant: {OUTPUT_PDB}")

    # Verify
    verify = PDBParser(QUIET=True).get_structure("check", str(OUTPUT_PDB))
    for model in verify:
        for chain in model:
            for residue in chain:
                if residue.id[1] == 83:
                    print(f"[INFO] Verified residue 83: "
                          f"{residue.get_resname()}")


if __name__ == "__main__":
    main()
