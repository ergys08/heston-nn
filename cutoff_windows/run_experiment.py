"""Tail shape (not merely volatility) versus Heston MPS complexity.

Preserves the previous fixed and adaptive raw-return windows as controls.
Primary analysis: amplitude encoding on a common window in STANDARD DEVIATIONS.
Rerunning writes the same output filenames and replaces the previous results.
"""
from __future__ import annotations
import argparse, csv, json, warnings
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from heston.build_distributions import read_calibrations
from heston.distributions import heston_pdf, discretize_pdf
from MPS.mps_decomposition import decompose_state
from MPS.circuit import prepare_iterative_mps

HERE=Path(__file__).resolve().parent
DEFAULT_INPUT=HERE.parent/"heston"/"SPX_generated_heston_params_2019_2020.csv"

def cdf(x,p):
    dx=float(x[1]-x[0]); xx=np.r_[x,x[-1]+dx]; pp=np.r_[p,p[0]]
    cc=np.r_[0.,np.cumsum((pp[:-1]+pp[1:])*dx*.5)]
    return xx,cc/cc[-1]

def mass(xx,cc,r):
    return float(np.interp(r,xx,cc)-np.interp(-r,xx,cc))

def radius(xx,cc,m):
    if not 0<m<1:raise ValueError("retained fraction must be between 0 and 1")
    lo,hi=0.,min(-xx[0],xx[-1])*(1-1e-7)
    if mass(xx,cc,hi)<m:raise ArithmeticError("FFT window insufficient")
    for _ in range(56):
        mid=(lo+hi)/2
        if mass(xx,cc,mid)>=m:hi=mid
        else:lo=mid
    return hi


def full_moments(x,pdf):
    """Moments of the complete PDF, not the truncated encoding."""
    dx=float(x[1]-x[0])
    mu=float(np.sum(x*pdf)*dx)
    v=float(np.sum((x-mu)**2*pdf)*dx)
    if v<=0 or not np.isfinite(v):raise ArithmeticError("Bad full-PDF variance")
    sd=np.sqrt(v)
    skew=float(np.sum((x-mu)**3*pdf)*dx/sd**3)
    kurt=float(np.sum((x-mu)**4*pdf)*dx/sd**4-3)
    return mu,sd,v,skew,kurt

def tail_z(xx,cc,mu,sd,z):
    lo,hi=mu-z*sd,mu+z*sd
    if lo<=xx[0] or hi>=xx[-1]:raise ArithmeticError("FFT grid too narrow for z tails")
    return float(1-(np.interp(hi,xx,cc)-np.interp(lo,xx,cc)))

def z_radius(xx,cc,mu,sd,target):
    if not 0<target<1:raise ValueError("Bad retained mass")
    lo,hi=0.,min((mu-xx[0])/sd,(xx[-1]-mu)/sd)*(1-1e-7)
    if hi<=0 or tail_z(xx,cc,mu,sd,hi)>1-target+1e-10:
        raise ArithmeticError("FFT grid too narrow for standardised cutoff")
    for _ in range(55):
        middle=(lo+hi)/2
        if tail_z(xx,cc,mu,sd,middle)>1-target:lo=middle
        else:hi=middle
    return hi

def bin_probs(xx,cc,left,right,qubits):
    if left<=xx[0] or right>=xx[-1] or left>=right:
        raise ArithmeticError("Encoding window outside FFT grid")
    edges=np.linspace(left,right,2**qubits+1)
    masses=np.diff(np.interp(edges,xx,cc))
    if masses.min()<-1e-12:raise ArithmeticError("Negative discretised mass")
    masses=np.maximum(masses,0)
    kept=float(masses.sum())
    if not 0<kept<=1+1e-9:raise ArithmeticError("Bad retained probability")
    return masses/kept,kept

def diagnostics(probs,thresholds,layers):
    psi=np.sqrt(np.asarray(probs,float))
    n=(len(psi)-1).bit_length()
    ranks=[];drops=[];entropies=[]
    for b in range(1,n):
        singular=np.linalg.svd(psi.reshape(2**b,-1),compute_uv=False)
        w=singular**2
        ranks.append(int(np.sum(singular>singular[0]*1e-10)))
        drops.append(float(w[2:].sum()))
        positive=w[w>1e-16]
        entropies.append(float(-(positive*np.log2(positive)).sum()))
    chi=[float(decompose_state(psi,max_bond=k).fidelity)
         for k in range(1,2**(n//2)+1)]
    layer=[float(z) for z in prepare_iterative_mps(psi,n_layers=layers).fidelities]
    out=dict(max_exact_bond_dimension=max(ranks),bond_ranks_json=json.dumps(ranks),
        max_discarded_chi2=max(drops),discarded_chi2_by_bond_json=json.dumps(drops),
        max_bond_entropy_bits=max(entropies),fidelity_by_chi_json=json.dumps(chi))
    for i,f in enumerate(layer,1):out[f"mps_layer_{i}_fidelity"]=f
    for t in thresholds:
        tag=str(t).replace("0.","")
        out[f"chi_for_{tag}"]=next((i for i,f in enumerate(chi,1) if f>=t-1e-12),None)
        out[f"layers_for_{tag}"]=next((i for i,f in enumerate(layer,1) if f>=t-1e-12),None)
    return out

def write(path,rows):
    if rows:
        with path.open("w",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)


def calculate_corr(rows,thresholds,layers):
    predictors=["excess_kurtosis","standardized_tail_mass_3sigma",
                "standardized_tail_mass_4sigma","skewness",
                "tail_mass_fixed","variance","sigma","v0"]
    outcomes=[]
    for mode in ("standardized","fixed","adaptive"):
        outcomes += [f"{mode}_max_exact_bond_dimension",f"{mode}_max_discarded_chi2",
                     f"{mode}_max_bond_entropy_bits"]
        outcomes += [f"{mode}_mps_layer_{j}_fidelity" for j in range(1,layers+1)]
        for t in thresholds:
            tag=str(t).replace("0.","")
            outcomes += [f"{mode}_chi_for_{tag}",f"{mode}_layers_for_{tag}"]
    groups={"all":rows}
    groups.update({year:[r for r in rows if r["date"].startswith(year)]
                   for year in sorted({r["date"][:4] for r in rows})})
    result=[]
    for group,subset in groups.items():
        for analysis in ("levels","first_difference"):
            pairs=([(r,r) for r in subset] if analysis=="levels" else
                   [(subset[i-1],subset[i]) for i in range(1,len(subset))
                    if subset[i-1]["date"][:4]==subset[i]["date"][:4]])
            for predictor in predictors:
                for outcome in outcomes:
                    xy=[]
                    for before,after in pairs:
                        try:
                            if analysis=="levels":
                                x,y=float(after[predictor]),float(after[outcome])
                            else:
                                x=float(after[predictor])-float(before[predictor])
                                y=float(after[outcome])-float(before[outcome])
                            if np.isfinite(x) and np.isfinite(y):xy.append((x,y))
                        except (ValueError,TypeError):pass
                    pr=sr=None
                    if len(xy)>=3:
                        x,y=np.array(xy).T
                        if np.std(x)>1e-13 and np.std(y)>1e-13:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                pr=float(pearsonr(x,y).statistic)
                                sr=float(spearmanr(x,y).statistic)
                    result.append(dict(sample=group,analysis=analysis,
                                       predictor=predictor,outcome=outcome,n=len(xy),
                                       pearson_r=pr,spearman_rho=sr))
    return result


def make_plots(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dates=np.array([np.datetime64(r["date"]) for r in rows])
    kurt=np.array([r["excess_kurtosis"] for r in rows])
    mass=np.array([r["standardized_tail_mass_3sigma"] for r in rows])
    fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
    ax[0].plot(dates,kurt);ax[0].set_ylabel("Excess kurtosis (full PDF)")
    ax[1].plot(dates,mass);ax[1].set_ylabel("P(|X-mu| > 3 std)")
    for mode in ("standardized","fixed","adaptive"):
        ax[2].plot(dates,[r[f"{mode}_mps_layer_1_fidelity"] for r in rows],label=mode)
    ax[2].set_ylabel("One-layer MPS fidelity");ax[2].set_xlabel("Trading day")
    ax[2].legend()
    fig.savefig(out/"timeline.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for a,key in zip(ax,("standardized_max_discarded_chi2",
                         "standardized_chi_for_995","standardized_mps_layer_1_fidelity")):
        a.scatter(kurt,[r[key] for r in rows],s=10,alpha=.55)
        a.set_xlabel("Full-PDF excess kurtosis");a.set_ylabel(key)
    fig.savefig(out/"correlation.png",dpi=160);plt.close(fig)


def run(args):
    if not 0<args.retained_fraction<1:raise ValueError("Bad retained fraction")
    if not 2<=args.qubits<=12:raise ValueError("Qubits must be 2..12")
    if args.layers<1 or any(not 0<t<=1 for t in args.fidelities):
        raise ValueError("Bad layers or fidelities")
    samples=read_calibrations(args.input)
    if args.max_states:samples=samples[:args.max_states]
    if not samples:raise ValueError("Empty calibration file")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    first_date,p0=samples[0]
    x0,pdf0,_=heston_pdf(p0,n_fft=args.n_fft,halfwidth=args.fft_halfwidth,
                         maturity=args.maturity,drift=args.drift)
    xx0,cc0=cdf(x0,pdf0)
    raw_radius=radius(xx0,cc0,args.retained_fraction)
    mu0,sd0,_,_,_=full_moments(x0,pdf0)
    std_radius=z_radius(xx0,cc0,mu0,sd0,args.retained_fraction)
    print(f"First date {first_date}: raw cutoff +/-{raw_radius:.8f}; "
          f"standardized +/-{std_radius:.5f} SD",flush=True)
    results=[];failed=[]
    for i,(date,p) in enumerate(samples):
        try:
            if i==0:x,den,xx,cc=x0,pdf0,xx0,cc0
            else:
                x,den,_=heston_pdf(p,n_fft=args.n_fft,halfwidth=args.fft_halfwidth,
                                   maturity=args.maturity,drift=args.drift)
                xx,cc=cdf(x,den)
            mu,sd,var,skew,kurt=full_moments(x,den)
            r_adaptive=radius(xx,cc,args.retained_fraction)
            windows=dict(fixed=(-raw_radius,raw_radius),
                         adaptive=(-r_adaptive,r_adaptive),
                         standardized=(mu-std_radius*sd,mu+std_radius*sd))
            row=dict(date=date,theta=p.theta,kappa=p.kappa,sigma=p.xi,
                     rho=p.rho,v0=p.v0,mean=mu,std_dev=sd,variance=var,
                     skewness=skew,excess_kurtosis=kurt,
                     fixed_radius=raw_radius,adaptive_radius=r_adaptive,
                     standardized_z_radius=std_radius)
            for z in (2,3,4):
                row[f"standardized_tail_mass_{z}sigma"]=tail_z(xx,cc,mu,sd,z)
            for mode,(left,right) in windows.items():
                probabilities,kept=bin_probs(xx,cc,left,right,args.qubits)
                row[f"{mode}_retained_mass"]=kept
                row[f"{mode}_tail_mass"]=1-kept
                row[f"{mode}_bin_width"]=(right-left)/(2**args.qubits)
                row.update({f"{mode}_{k}":v for k,v in
                            diagnostics(probabilities,args.fidelities,args.layers).items()})
            row["tail_mass_fixed"]=row["fixed_tail_mass"]
            row["tail_mass_adaptive"]=row["adaptive_tail_mass"]
            results.append(row)
        except (ArithmeticError,ValueError,FloatingPointError,np.linalg.LinAlgError) as e:
            failed.append(dict(date=date,error=str(e)))
            if args.fail_fast:raise
        if (i+1)%20==0 or i+1==len(samples):
            print(f"{i+1}/{len(samples)}; succeeded {len(results)}, failed {len(failed)}",flush=True)
    write(args.output_dir/"daily_results.csv",results)
    write(args.output_dir/"failures.csv",failed)
    if not failed:(args.output_dir/"failures.csv").write_text("date,error\n")
    cor=calculate_corr(results,args.fidelities,args.layers)
    write(args.output_dir/"correlations.csv",cor)
    if results:make_plots(results,args.output_dir)
    selected=[r for r in cor if r["sample"] in ("all","2019","2020")
              and r["analysis"]=="levels"
              and r["predictor"] in ("excess_kurtosis","standardized_tail_mass_3sigma")
              and r["outcome"] in ("standardized_max_discarded_chi2",
                                   "standardized_chi_for_995",
                                   "standardized_mps_layer_1_fidelity",
                                   "standardized_layers_for_999")]
    summary=dict(source=str(args.input),n_input=len(samples),
      n_success=len(results),n_failed=len(failed),first_date=first_date,
      fixed_symmetric_cutoff=[-raw_radius,raw_radius],
      standardized_z_radius=std_radius,
      retained_fraction=args.retained_fraction,n_qubits=args.qubits,
      fidelity_thresholds=args.fidelities,n_layers=args.layers,
      n_fft=args.n_fft,fft_halfwidth=args.fft_halfwidth,
      maturity=args.maturity,drift=args.drift,
      key_correlations=selected,
      caveat="Kurtosis and standardized tails measure shape. Correlations are "
             "observational, serially dependent; standardized encoding changes "
             "the physical bins.")
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print("Overwrote previous results in",args.output_dir.resolve(),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    p.add_argument("--output-dir",type=Path,default=HERE)
    p.add_argument("--retained-fraction",type=float,default=.999)
    p.add_argument("--qubits",type=int,default=8)
    p.add_argument("--n-fft",type=int,default=131072)
    p.add_argument("--fft-halfwidth",type=float,default=25.)
    p.add_argument("--maturity",type=float,default=1.)
    p.add_argument("--drift",type=float,default=.05)
    p.add_argument("--fidelities",type=float,nargs="+",default=[.99,.995,.999])
    p.add_argument("--layers",type=int,default=3)
    p.add_argument("--max-states",type=int)
    p.add_argument("--fail-fast",action="store_true")
    return run(p.parse_args())

if __name__=="__main__":main()
