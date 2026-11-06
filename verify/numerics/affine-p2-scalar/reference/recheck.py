"""Recheck retained independent-reference coefficients without any linear solve."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from refinement import certify


def encoded(value):
    def rational(x):
        if isinstance(x,Fraction):return str(x)
        raise TypeError(type(x).__name__)
    return json.loads(json.dumps(value,default=rational))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('data')/'manifest.json')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    for name,digest in manifest['files'].items():
        path=args.manifest.parent/name
        if '..' in Path(name).parts or Path(name).is_absolute():raise ValueError('invalid packet path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('packet byte digest mismatch')
    if hashlib.sha256((args.directory/'receipt.json').read_bytes()).hexdigest()!=manifest['files']['refinement/receipt.json']:
        raise ValueError('unbound generation receipt')
    receipt=json.loads((args.directory/'receipt.json').read_text())
    if receipt.get('status')!='certified':raise ValueError('incomplete reference generation')
    levels=receipt['levels']
    if [(r['dimension'],r['N']) for r in levels]!=[(d,n) for d in (2,3) for n in (2,4,8)]:raise ValueError('incomplete level set')
    for row in levels:
        path=args.directory/row['file']
        if path.name!=f"d{row['dimension']}-n{row['N']}.json" or path.parent!=args.directory:raise ValueError('invalid level path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('reference byte digest mismatch')
        saved=json.loads(path.read_text())
        checked=encoded(certify(row['dimension'],row['N'],saved['coefficients_binary64_hex']))
        if checked!=saved:raise ValueError('reference exact certificate mismatch')
        print(f"PASS dimension={row['dimension']} N={row['N']} exact-residual-and-full-field-norms",flush=True)

if __name__=='__main__':main()
