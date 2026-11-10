"""
35_download_structures.py (v5)
Download ParC and GyrA from AlphaFold DB with proper User-Agent.
"""
import requests
from pathlib import Path

OUT_DIR = Path("data/protein_structures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIPROT_IDS = {"P47446": "ParC", "P47250": "GyrA"}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def download_via_api(uniprot_id, name):
    print(f"[INFO] {name} ({uniprot_id}) - trying API...")
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        r = requests.get(api_url, headers=HEADERS, timeout=30)
        print(f"  API status: {r.status_code}")
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        pdb_url = data[0].get("pdbUrl")
        if not pdb_url:
            return None
        print(f"  PDB URL: {pdb_url}")
        r2 = requests.get(pdb_url, headers=HEADERS, timeout=60)
        print(f"  Download status: {r2.status_code}")
        if r2.status_code != 200:
            return None
        out_path = OUT_DIR / f"{name}_{uniprot_id}.pdb"
        out_path.write_bytes(r2.content)
        print(f"  Saved: {out_path} ({len(r2.content)} bytes)")
        return out_path
    except Exception as e:
        print(f"  API error: {e}")
        return None

def download_via_3dbeacons(uniprot_id, name):
    print(f"[INFO] {name} - trying 3D-Beacons...")
    url = f"https://www.ebi.ac.uk/pdbe/pdbe-kb/3dbeacons/api/uniprot/summary/{uniprot_id}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        print(f"  3D-Beacons status: {r.status_code}")
        if r.status_code != 200:
            return None
        data = r.json()
        for s in data.get("structures", []):
            summary = s.get("summary", {})
            provider = summary.get("provider", "")
            if "AlphaFold" in provider or "alphafold" in provider.lower():
                model_url = s.get("model", {}).get("url") or s.get("model_url")
                if not model_url:
                    continue
                print(f"  3D-Beacons model URL: {model_url}")
                r2 = requests.get(model_url, headers=HEADERS, timeout=60)
                if r2.status_code != 200:
                    continue
                out_path = OUT_DIR / f"{name}_{uniprot_id}.pdb"
                out_path.write_bytes(r2.content)
                print(f"  Saved: {out_path} ({len(r2.content)} bytes)")
                return out_path
    except Exception as e:
        print(f"  3D-Beacons error: {e}")
    return None

def main():
    for uid, name in UNIPROT_IDS.items():
        result = download_via_api(uid, name)
        if result is None:
            result = download_via_3dbeacons(uid, name)
        if result is None:
            print(f"[WARN] All methods failed for {name}. Use manual download.")
    print("\n[INFO] Done:", OUT_DIR)

if __name__ == "__main__":
    main()
