#!/usr/bin/env python3

import argparse
import os
import pdb

import numpy as np
import matplotlib.pyplot as plt
from rabbit import io_tools
from wums import logging, output_tools, plot_tools
from rabbit import debugdata, inputdata

sort_choices = []
sort_choices_abs = [f"abs {s}" for s in sort_choices]


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default=os.path.expanduser("./test"),
        help="Base path for output",
    )

    parser.add_argument(
        "inputFile",
        type=str,
        help="fitresults output",
    )
    parser.add_argument(
        "--keepNuisances",
        type=str,
        default=None,
        help="Match nuisances by regular expression",
    )
    parser.add_argument(
        "--excludeNuisances",
        type=str,
        default=None,
        help="Exclude nuisances by regular expression",
    )
        
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="fitresults key in file (e.g. 'asimov'). Leave empty for data fit result.",
    )
   
    parser.add_argument(
        "--nPoi",
        type=int,
        default=-1,
        help="number of pois to plot",
    )
    parser.add_argument(
        "--poiFormat",
        type=bool,
        default=True,
        help="formatting for my specific poi",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    return parser.parse_args()



def main():
    args = parseArgs()
    fitresult = io_tools.get_fitresult(args.inputFile, args.result)

    labels, pulls, constraints = io_tools.get_pulls_and_constraints(
        fitresult,
        keep_nuisances=args.keepNuisances,
        exclude_nuisances=args.excludeNuisances,
    )

    #dxxd, dxyd, dxzd, dyzd are calculated by hand. need to make the mapping
    fit_values = {"cxxu":  [23717.4965, ], 
        "cxxd": [213647.74311, ], 
        "dxxu": [19168.37659, ], 
        "dxxd": [0, ], 
        "cxxs": [139553.16388, ], 
        "dxxs": [1493309.21303, ], 
        "cxyu": [23795.83126, ], 
        "cxyd": [214362.81607, ], 
        "dxyu": [19229.5826, ], 
        "dxyd": [0, 240.24888], 
        "cxys": [140007.6221, ],
        "dxys": [1498216.68121, ], 
        "cxzu": [11364.09578, ], 
        "cxzd": [102368.71343, ], 
        "dxzu": [9183.32597, ], 
        "dxzd": [0, ],
        "cxzs": [66838.30456, ], 
        "dxzs": [715505.99146, ], 
        "cyzu": [11330.84679, ], 
        "cyzd": [102061.37775, ], 
        "dyzu": [9159.41109, ], 
        "dyzd": [0, ],
        "cyzs": [66672.48952, ], 
        "dyzs": [713439.79058, ]}

    literature_vals = {"cxyu": [-32e-5, 4.1e-5], "cxzu": [-5.4e-4, 1.4e-4], "cyzu": [-3.7e-4, 2.1e-4], "cxxu": [-2.1e-5, 2.4e-5], "cxyd": [-16e-4, 2.0e-4], "cxzd": [-27e-4, 7e-4], "cyzd": [-1.8e-3, 1e-3], "cxxd": [-1.0e-3, 1.2e-3], "cxys": [-26e-4, 3.3e-4], "cxzs": [-4.4e-3, 1.2e-3], "cyzs": [-3e-3, 1.7e-3], "cxxs": [-1.7e-3, 2.e-3]
    }
    # pdb.set_trace()
    # constraints = []
    # for val in labels:
    #     try: 
    #         constraints.append(fit_values[str(val)][1])
    #     except:
    #         pass
    # constraints = np.array(constraints)
    # y positions (top to bottom)
    y = np.arange(args.nPoi)[::-1]
    n = args.nPoi
    # --- Plot ---
    if args.nPoi == 20:
        fig, ax = plt.subplots(figsize=(7, 10))
    elif args.nPoi == 12:
        fig, ax = plt.subplots(figsize=(7, 9))
    elif args.nPoi == 8:
        fig, ax = plt.subplots(figsize=(7, 5))

    else:
        fig, ax = plt.subplots(figsize=(7, 3))
    # 95% CL (thin lines)
    ax.errorbar(pulls[:n]*1e-6, y, xerr=constraints[:n]*2*1e-6, fmt='none',
                ecolor='black', elinewidth=1, capsize=0,)
    # 68% CL (thicker lines + points)
    ax.errorbar(pulls[:n]*1e-6, y, xerr=constraints[:n]*1e-6, fmt='o',
                color='black', ecolor='black', elinewidth=3, capsize=0, label='68%, 95% CL')
    previously_labeled = False
    
    # for i in range(len(labels[:n])):
    #     if (labels[i] in literature_vals.keys()) and (literature_vals[labels[i]][0] != 0):
    #         this_val = np.abs(np.array(literature_vals[labels[i]])[:, None])
    #         if not previously_labeled:
    #             ax.errorbar(0, y[i]+0.25, xerr = this_val, color = 'C0', elinewidth=3, capsize=0, label = 'ZEUS \n 68%, 95% CL')
    #             ax.errorbar(0, y[i]+0.25, xerr = 2*this_val, color = 'C0', fmt='none', elinewidth=1, capsize=0)
    #             previously_labeled = True
    #         else:
    #             ax.errorbar(0, y[i]+0.25, xerr =this_val, color = 'C0', elinewidth=3, capsize=0)
    #             ax.errorbar(0, y[i]+0.25,fmt='none',  xerr = 2*this_val, color = 'C0', elinewidth=1, capsize=0)
    # zero reference line
    ax.axvline(0, color='black', linestyle='--', linewidth=1)

    # labels & ticks
    ax.set_yticks(y)
    if args.poiFormat:
        labels_mod = [f'${l[0]}^{{{l[1:3]}}}_{l[-1]}$' for l in labels[:n]]
        ax.set_yticklabels(labels_mod)
    else:
        ax.set_yticklabels(labels[:n])

    
    # ax.set_xlabel("SME coefficient value")
    # group separators (optional)
    if args.nPoi > 5:
        if args.nPoi == 20 or args.nPoi == 18:
            for sep in [7.5, 11.5 ]:
                ax.axhline(sep, color='gray', linewidth=1)
    xlim = 2 #5e-4
    ax.set_xlim([-xlim, xlim])
    # grid (vertical only)
    ax.xaxis.grid(True, linestyle=':', color='gray', alpha=0.7)
    ax.yaxis.grid(False)
    ax.yaxis.minorticks_off()
    ax.legend(frameon=True, fontsize = 16)

    # cleaner look
    ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)

    
    outdir = output_tools.make_plot_dir(args.outpath)

    # indata = inputdata.FitInputData(args.infile, pseudodata=args.pseudodata)
    
    outfile = 'fitted_values' + args.postfix
    plot_tools.save_pdf_and_png(outdir, outfile)
    output_tools.write_index_and_log(
            outdir,
            outfile,
            args=args,
        )
    
    logname = f"{outdir}/{outfile}.log"
    
    
    with open(logname, "a") as logf:
        logf.write("\n")
        logf.write("-------------------------------------------- \n")
        logf.write("process: # of events\n")
    
         

if __name__ == "__main__":
    main()
