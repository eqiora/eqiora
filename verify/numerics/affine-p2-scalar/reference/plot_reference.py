"""Plot retained independent intervals; no solve, candidate or reference fitting."""
import json
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator


def render(width):
    root=Path(__file__).parent
    mobile=width==320
    plt.rcParams.update({'font.size':7 if mobile else 11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2 if mobile else 1,1 if mobile else 2,figsize=(width/160,5.8 if mobile else 3.6),layout='constrained')
    for d,ax in zip((2,3),axes):
        for name,label,marker in [('L2','L2 norm','o'),('H1_seminorm','H1 seminorm','s')]:
            rows=[json.loads((root/'data'/'refinement'/f'd{d}-n{n}.json').read_text()) for n in (2,4,8)]
            low=[float(Fraction(r[name]['lower'])) for r in rows]
            high=[float(Fraction(r[name]['upper'])) for r in rows]
            mid=[(a+b)/2 for a,b in zip(low,high)]
            ax.errorbar([2,4,8],mid,yerr=[[m-a for m,a in zip(mid,low)],[b-m for b,m in zip(high,mid)]],label=label,marker=marker,capsize=3)
        ax.set(xscale='log',yscale='log',xlabel='Logical intervals N' if mobile else 'Intervals per logical axis N',ylabel='Normalized error' if mobile else 'Normalized error norm',title=f'{d}D independent\nP2 reference' if mobile else f'{d}D independent P2 reference')
        ax.set_xticks([2,4,8],labels=['2','4','8']);ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(alpha=.25);ax.legend()
    fig.savefig(root/('reference-errors-320.png' if mobile else 'reference-errors.png'),dpi=160)
    plt.close(fig)


def main():
    for width in (1440,320):render(width)

if __name__=='__main__':main()
