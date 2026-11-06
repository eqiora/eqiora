"""Plot original samples only after clone-only scientific and stored-audit replay."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from recheck import replay, HERE


def main():
    with replay() as (_, cases, reports, summary):
        by_tool = {(r['case_label'], r['tool']): r for r in reports['benchmark']}
        labels = [f'{case.label}\n{len(case.reference.data["cells"]):,} cells' for case in cases]
        for mobile in (False, True):
            width = 3.2 if mobile else 9.6
            dpi = 100 if mobile else 150
            fig, axes = plt.subplots(2, 1, figsize=(width, 8.7 if mobile else 6.6), layout='constrained')
            for backend, color, shift in (('eqiora', '#0f766e', -.14), ('fenicsx', '#b45309', .14)):
                medians, lows, highs, cold_memory, final_memory = [], [], [], [], []
                for i, case in enumerate(cases):
                    report = by_tool[case.label, backend]; m = report['execution_measurement']
                    samples = [s['wall_seconds'] for s in m['samples']]
                    medians.append(m['median_wall_seconds']); lows.append(m['min_wall_seconds']); highs.append(m['max_wall_seconds'])
                    axes[0].scatter(i+shift+np.linspace(-.045,.045,7), samples, s=18, color=color, alpha=.65)
                    cold_memory.append(report['memory_after_execution_before_accuracy']['process_peak_rss_kib']/1024)
                    final_memory.append(report['memory_at_end']['process_peak_rss_kib']/1024)
                x = np.arange(3)+shift
                axes[0].errorbar(x, medians, yerr=[np.array(medians)-lows,np.array(highs)-medians], fmt='_',
                                 ms=15, capsize=5, color=color, label=({'eqiora': 'Eqiora', 'fenicsx': 'FEniCSx'}[backend]), lw=1.4)
                axes[1].plot(x, cold_memory, 'o-', color=color, label=({'eqiora': 'Eqiora', 'fenicsx': 'FEniCSx'}[backend])+' cold HWM')
                axes[1].scatter(x, final_memory, facecolors='none', edgecolors=color, marker='s', s=40,
                                label=({'eqiora': 'Eqiora', 'fenicsx': 'FEniCSx'}[backend])+' final HWM')
            axes[0].set_yscale('log'); axes[0].set_ylabel('Ordinary wall time [s]')
            axes[0].set_title('Seven samples; median and full range', fontsize=10)
            axes[1].set_ylabel('Process peak RSS [MiB]'); axes[1].set_title('Process high-water marks\nCold return and final child' if mobile else 'Cold return and whole-child high-water marks', fontsize=9 if mobile else 10)
            for ax in axes:
                ax.set_xticks(range(3),labels); ax.grid(axis='y',alpha=.22); ax.legend(fontsize=7 if mobile else 8)
            fig.suptitle('Steady Stokes cylinder\nOne CPU thread • shared host\nFresh matrices and factors' if mobile else
                         'Steady Stokes cylinder: ordinary execution and process memory\nOne thread; shared host; fresh matrices and factors', fontsize=11)
            fig.savefig(HERE/('runtime-320.png' if mobile else 'runtime-1440.png'), dpi=dpi);plt.close(fig)
            fig, ax = plt.subplots(figsize=(width, 5.4 if mobile else 4.2),layout='constrained')
            metric_names = ('velocity','pressure','force_x','force_y')
            for index, case in enumerate(cases):
                accuracy = summary['benchmark'][case.label]['eqiora']['agreement']
                ratios = [accuracy[k]/case.reference.manifest['metrics'][k]['tolerance'] for k in metric_names]
                ax.plot(range(4), ratios, 'o-', label=case.label)
            ax.axhline(1,color='#475569',ls='--',label='Frozen gate');ax.set_yscale('log');ax.set_ylim(1e-10,3)
            ax.set_xticks(range(4),['u RMS','p RMS','x force','y force']);ax.set_ylabel('Difference / frozen allowance')
            ax.set_title('Eqiora cold fields vs designated A\nIndependent A/B/C budget per mesh',fontsize=10)
            ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
            fig.savefig(HERE/('agreement-320.png' if mobile else 'agreement-1440.png'),dpi=dpi);plt.close(fig)
    print('Rendered 320px and 1440px runtime and agreement figures after full replay')


if __name__ == '__main__':
    main()
