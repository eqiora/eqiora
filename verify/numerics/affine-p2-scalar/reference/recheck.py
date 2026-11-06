"""Recheck retained independent-reference coefficients without any linear solve."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from refinement import certify


def encoded(value):
    return json.loads(json.dumps(value,default=lambda x:str(x) if isinstance(x,Fraction) else None))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    args=parser.parse_args()
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
