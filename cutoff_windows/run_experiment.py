"""Fixed and adaptive cutoff windows versus MPS complexity, 2019–2020 SPX."""
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

def diagnostics(probs,thresholds,layers):
    psi=np.sqrt(probs);n=(len(psi)-1).bit_length()
    ranks=[];drops=[];ents=[]
    for b in range(1,n):
        s=np.linalg.svd(psi.reshape(2**b,-1),compute_uv=False);w=s*s
        ranks.append(int((s>1e-10*s[0]).sum()));drops.append(float(w[2:].sum()))
        nz=w[w>1e-16];ents.append(float(-(nz*np.log2(nz)).sum()))
    ff=[float(decompose_state(psi,max_bond=k).fidelity) for k in range(1,2**(n//2)+1)]
    ll=[float(x) for x in prepare_iterative_mps(psi,n_layers=layers).fidelities]
    out=dict(max_exact_bond_dimension=max(ranks),bond_ranks_json=json.dumps(ranks),
             max_discarded_chi2=max(drops),discarded_chi2_by_bond_json=json.dumps(drops),
             max_bond_entropy_bits=max(ents),fidelity_by_chi_json=json.dumps(ff))
    for j,f in enumerate(ll,1):out[f"mps_layer_{j}_fidelity"]=f
    for threshold in thresholds:
        tag=str(threshold).replace("0.","")
        out[f"chi_for_{tag}"]=next((i for i,f in enumerate(ff,1) if f>=threshold-1e-12),None)
        out[f"layers_for_{tag}"]=next((i for i,f in enumerate(ll,1) if f>=threshold-1e-12),None)
    return out

def write(path,rows):
    if rows:
        with path.open("w",newline="") as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def calculate_corr(rows,thresholds,layers):
    xs=["tail_mass_fixed","adaptive_radius","excess_kurtosis","variance","sigma","v0"]
    ys=[]
    for w in ("fixed","adaptive"):
        ys += [f"{w}_max_exact_bond_dimension",f"{w}_max_discarded_chi2",f"{w}_max_bond_entropy_bits"]
        ys += [f"{w}_mps_layer_{j}_fidelity" for j in range(1,layers+1)]
        for t in thresholds:
            tag=str(t).replace("0.","");ys += [f"{w}_chi_for_{tag}",f"{w}_layers_for_{tag}"]
    out=[]
    for x in xs:
        for y in ys:
            xy=[]
            for r in rows:
                try:
                    a,b=float(r[x]),float(r[y])
                    if np.isfinite(a) and np.isfinite(b):xy.append((a,b))
                except (ValueError,TypeError):pass
            pr=sr=None
            if len(xy)>=3:
                a,b=np.array(xy).T
                if np.std(a)>1e-13 and np.std(b)>1e-13:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pr=float(pearsonr(a,b).statistic);sr=float(spearmanr(a,b).statistic)
            out.append(dict(predictor=x,outcome=y,n=len(xy),pearson_r=pr,spearman_rho=sr))
    return out

def make_plots(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d=np.array([np.datetime64(r["date"]) for r in rows])
    t=np.array([r["tail_mass_fixed"] for r in rows])
    f,a=plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
    a[0].plot(d,t);a[0].set_ylabel("Mass outside fixed cutoff")
    for w in ("fixed","adaptive"):
        a[1].plot(d,[r[f"{w}_chi_for_995"] for r in rows],label=w)
        a[2].plot(d,[r[f"{w}_mps_layer_1_fidelity"] for r in rows],label=w)
    a[1].set_ylabel("Bond dimension for 0.995 fidelity");a[1].legend()
    a[2].set_ylabel("One-layer MPS fidelity");a[2].legend()
    f.savefig(out/"timeline.png",dpi=160);plt.close(f)
    f,a=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,key in zip(a,("fixed_max_discarded_chi2","fixed_chi_for_995")):
        ax.scatter(t,[r[key] for r in rows],s=10,alpha=.5)
        ax.set_xlabel("Mass outside first-date cutoff");ax.set_ylabel(key)
    f.savefig(out/"correlation.png",dpi=160);plt.close(f)

def run(args):
    if not 2<=args.qubits<=12:raise ValueError("qubits must be between 2 and 12")
    if args.layers<1:raise ValueError("layers must be positive")
    if not args.fidelities or any(not 0<f<=1 for f in args.fidelities):raise ValueError("invalid fidelity thresholds")
    samples=read_calibrations(args.input)
    if args.max_states:samples=samples[:args.max_states]
    if not samples:raise ValueError("empty input")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    firstdate,p0=samples[0]
    x0,den0,_=heston_pdf(p0,n_fft=args.n_fft,halfwidth=args.fft_halfwidth,maturity=args.maturity,drift=args.drift)
    xx0,cc0=cdf(x0,den0);fixed_r=radius(xx0,cc0,args.retained_fraction)
    print(f"First date {firstdate}: fixed symmetric cutoff +/-{fixed_r:.8f}",flush=True)
    rows=[];failed=[]
    for i,(date,p) in enumerate(samples):
        try:
            if i==0:x,den,xx,cc=x0,den0,xx0,cc0
            else:
                x,den,_=heston_pdf(p,n_fft=args.n_fft,halfwidth=args.fft_halfwidth,maturity=args.maturity,drift=args.drift)
                xx,cc=cdf(x,den)
            adaptive_r=radius(xx,cc,args.retained_fraction)
            _,_,fp,fret=discretize_pdf(x,den,n_qubits=args.qubits,cutoff=fixed_r)
            _,_,ap,aret=discretize_pdf(x,den,n_qubits=args.qubits,cutoff=adaptive_r)
            dx=float(x[1]-x[0]);mu=float(np.sum(x*den)*dx)
            var=float(np.sum((x-mu)**2*den)*dx)
            kurt=float(np.sum((x-mu)**4*den)*dx/var**2-3)
            r=dict(date=date,theta=p.theta,kappa=p.kappa,sigma=p.xi,rho=p.rho,v0=p.v0,
                   fixed_radius=fixed_r,adaptive_radius=adaptive_r,
                   tail_mass_fixed=1-fret,tail_mass_adaptive=1-aret,
                   fixed_retained=fret,adaptive_retained=aret,
                   fixed_bin_width=2*fixed_r/2**args.qubits,
                   adaptive_bin_width=2*adaptive_r/2**args.qubits,
                   variance=var,excess_kurtosis=kurt)
            for w,pr in (("fixed",fp),("adaptive",ap)):
                r.update({f"{w}_{k}":v for k,v in diagnostics(pr,args.fidelities,args.layers).items()})
            rows.append(r)
        except (ValueError,ArithmeticError,FloatingPointError,np.linalg.LinAlgError) as e:
            failed.append(dict(date=date,error=str(e)))
            if args.fail_fast:raise
        if (i+1)%20==0 or i+1==len(samples):
            print(f"{i+1}/{len(samples)} dates, {len(rows)} successful, {len(failed)} failed",flush=True)
    write(args.output_dir/"daily_results.csv",rows)
    write(args.output_dir/"failures.csv",failed)
    cor=calculate_corr(rows,args.fidelities,args.layers)
    write(args.output_dir/"correlations.csv",cor)
    if rows:make_plots(rows,args.output_dir)
    summary=dict(source=str(args.input),n_input=len(samples),n_success=len(rows),n_failed=len(failed),
        first_date=firstdate,fixed_symmetric_cutoff=[-fixed_r,fixed_r],
        retained_fraction=args.retained_fraction,n_qubits=args.qubits,
        fidelity_thresholds=args.fidelities,n_layers=args.layers,
        n_fft=args.n_fft,fft_halfwidth=args.fft_halfwidth,
        maturity=args.maturity,drift=args.drift,
        tail_mass_median=float(np.median([r["tail_mass_fixed"] for r in rows])) if rows else None,
        tail_mass_range=[float(min(r["tail_mass_fixed"] for r in rows)),float(max(r["tail_mass_fixed"] for r in rows))] if rows else [],
        key_correlations=[r for r in cor if r["predictor"]=="tail_mass_fixed" and r["outcome"] in
            ("fixed_chi_for_995","adaptive_chi_for_995","fixed_max_discarded_chi2","fixed_mps_layer_1_fidelity")],
        caveat="Comparisons describe correlations, not causality; fixed and adaptive quantum states use different bins.")
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print("Results saved in",args.output_dir,flush=True)
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
