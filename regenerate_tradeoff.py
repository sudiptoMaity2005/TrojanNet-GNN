import pandas as pd, numpy as np, glob, re, sys, math
D=sys.argv[1] if len(sys.argv)>1 else '/tmp/gs2'
MODEL=sys.argv[2] if len(sys.argv)>2 else 'ExtraTrees'
levels=[0.3,0.5]
def at_recall(y,s,r):
    P=int(y.sum()); T=math.ceil(r*P)
    df=pd.DataFrame({'s':s,'y':y}).groupby('s').agg(n=('y','size'),tp=('y','sum')).sort_index(ascending=False)
    cn=df.n.cumsum().values; ct=df.tp.cumsum().values
    i=int(np.argmax(ct>=T)); prev_n=cn[i-1] if i>0 else 0; prev_t=ct[i-1] if i>0 else 0
    g_n=df.n.values[i]; g_t=df.tp.values[i]
    k=prev_n+g_n*(T-prev_t)/g_t
    return T/k, k/len(y)
rows=[]
for f in glob.glob(f'{D}/gate_scores_s*_{MODEL}_*.csv'):
    nm=f.split('/')[-1][:-4]; m=re.match(r'gate_scores_s(\d+)_'+MODEL+r'_(.+)$',nm)
    seed,circ=int(m.group(1)),m.group(2)
    d=pd.read_csv(f,usecols=['y_true','score']); y=d.y_true.values; s=d.score.values
    if y.sum()==0: continue
    rec={'seed':seed,'circuit':circ,'n':len(y),'P':int(y.sum())}
    for r in levels:
        p,k=at_recall(y,s,r); rec[f'prec{int(r*100)}']=p; rec[f'insp{int(r*100)}']=k
    rows.append(rec)
d=pd.DataFrame(rows)
d['family']=d.circuit.str.replace(r'[-_]T?\d+.*$','',regex=True).str.replace(r'^b19.*','b19',regex=True)
d.loc[d.circuit.str.startswith('b19'),'family']='b19'; d.loc[d.circuit.str.startswith('wb_conmax'),'family']='wb_conmax'
d['base']=d.P/d.n
fam=d.groupby(['family','seed'])[['base','prec30','prec50','insp30','insp50']].mean().reset_index().groupby('family')[['base','prec30','prec50','insp30','insp50']].mean()
fam.to_csv('tradeoff_family.csv')
order=['c3540','c6288','c2670','AES','s1423','c5315','RS232','s15850','s35932','s13207','EthernetMAC10GE','b19','wb_conmax']
fam=fam.loc[order]; print((fam*100).round(2).to_string())
mac=fam.mean(); print('macro',(mac*100).round(2).to_dict())
disp={'b19':'B19','wb_conmax':'wb-conmax'}
rows=[]
for f in order:
    v=fam.loc[f]
    rows.append(f"{disp.get(f,f)} & {100*v.base:.2f} & {100*v.prec30:.1f} & {100*v.prec50:.1f} & {100*v.insp50:.1f} \\\\")
rows.append("\\midrule"); rows.append(f"\\textbf{{Macro average}} & {100*mac.base:.2f} & {100*mac.prec30:.1f} & {100*mac.prec50:.1f} & {100*mac.insp50:.1f} \\\\")
open('tradeoff_rows.tex','w').write('\n'.join(rows))
