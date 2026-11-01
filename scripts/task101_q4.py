import sys,json,re
sys.path.insert(0,'scripts')
import task101_overlap as T
import task58_passages as P
from task58_measure import load,classify,words,EQ_IN_PARA,CITE_RE,MKR
L3=T.FILES['ch3'].read_text().split('\n'); L5=T.FILES['ch5'].read_text().split('\n')
def wr(lines,rs): return sum(len(lines[i-1].split()) for a,b in rs for i in range(a,b+1))
def tx(lines,rs): return ' '.join(lines[i-1] for a,b in rs for i in range(a,b+1))
def cov(a,b,n=8):
    ta,tb=T.norm_tokens(a),T.norm_tokens(b)
    c=set()
    for i,j,Lr in T.shared_runs(ta,tb,n): c.update(range(j,j+Lr))
    r=T.shared_runs(ta,tb,n)
    return len(c),max([x[2] for x in r],default=0),len(T.shared_runs(ta,tb,12))
Q=[
 ("Spatial resolution",(3102,3109),"5.8.2 Single-node spatial resolution",(825,830)),
 ("Perfect foresight",(3111,3118),"5.8.2 Perfect foresight",(817,823)),
 ("No integer structure",(3120,3134),"5.8.2 No integer structure",(832,839)),
 ("Storage: energy balance, not state of charge",(3136,3144),"5.8.2 Storage as an energy balance",(841,848)),
 ("Surplus generation freely disposable",(3146,3151),None,None),
 ("Outage treatment",(3153,3161),None,None),
 ("Demand exogenous and price-inelastic",(3163,3174),"5.8.2 Demand is exogenous and price-inelastic",(850,858)),
 ("System boundary (trade; busbar factor)",(3176,3187),"5.8.2 System boundary and network losses",(860,866)),
 ("Hydro carries no cost term of any kind",(3189,3197),"5.8.2 Hydro carries no operating cost",(873,879)),
 ("Technology learning exogenous",(3199,3205),"5.8.2 Technology learning is exogenous",(881,885)),
 ("Policy exogenous throughout",(3207,3215),"5.8.2 Policy is exogenous",(868,871)),
 ("Capacity credits are assumptions",(3217,3223),"5.8.3 Composition of the firm-capacity constraint",(935,939)),
 ("Two narrower omissions (gas price; hydrology)",(3225,3231),None,None),
 ("No single direction summarises these",(3233,3240),None,None),
]
print("| §3.7.3 limitation | Ch3 lines | Ch3 words | Chapter 5 passage | Ch5 lines | Ch5 words | 12-runs | longest run | Ch5 words inside ≥8-runs | kind |")
print("|---|---|---|---|---|---|---|---|---|---|")
for n,r3,n5,r5 in Q:
    w3=wr(L3,[r3])
    if n5:
        c,mx,n12=cov(tx(L3,[r3]),tx(L5,[r5])); w5=wr(L5,[r5])
        share=100*c//w5
        kind="L" if (n12>0 or c>=0.25*w5) else "X"
        print(f"| {n} | {r3[0]}-{r3[1]} | {w3} | {n5} | {r5[0]}-{r5[1]} | {w5} | {n12} | {mx} | {c} ({share}%) | {kind} |")
    else:
        print(f"| {n} | {r3[0]}-{r3[1]} | {w3} | none found | - | - | - | - | - | - |")
