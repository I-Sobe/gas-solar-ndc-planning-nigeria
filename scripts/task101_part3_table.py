import sys,json,re
sys.path.insert(0,'scripts')
import task58_passages as P
from task58_measure import load,classify,words,EQ_IN_PARA,CITE_RE,MKR,CH
R={r['id']:r for r in json.load(open(''+__import__('os').environ.get('T101_SCRATCH','/tmp/t101')+'/remap.json'))}
L=load(CH); cls=classify(L)
chg={'A1':'rewritten (TASK 93 A, 94 E.1): vintage figures; the "earlier state" sentence stands','A6':'register letter B→C','A7':'register letter B→C','A15':'register letter B→C','A18':'register letter B→C','A19':'register letter B→C','A20':'rewritten (TASK 87 Part 3): endpoint restored to the blended 24,986,620,949.18; VERIFY marker enlarged by 59 words','A24':'register letter B→C','A27':'register letter B→C','A29':'register letter C→B','A34':'rewritten (TASK 91 B.2): legs now "unconstrained"; "earlier state … bound" clause kept','A36':'clause "operating and capital cost" (TASK 88)'}
live={'A23','A26','A38'}
rows=[]
for tier,rows_in in (('A',P.A),('B',P.B)):
    for id_,sec,rng,q,note in rows_in:
        r=R[id_]
        txt=' '.join(L[i-1] for a,b in r['new'] for i in range(a,b+1))
        e=r['eq'];
        flag=lambda c: 'Y' if c else ''
        eq=flag(EQ_IN_PARA.search(txt) or e); ci=flag(CITE_RE.search(txt)); rg=flag(re.search(r"Appendix [A-D]|\bL\d{1,3}\b|commit `",txt)); va=flag('⟦' in txt); mk=flag(MKR.search(txt))
        nw=r['prose']+r['table']
        free= not (eq or ci or rg or va or mk)
        rows.append(dict(id=id_,tier=tier,sec=sec,old=r['old'],new=r['new'],oldw=r['oldw'],neww=nw,eq=eq,ci=ci,rg=rg,va=va,mk=mk,free=free,q=q,chg=chg.get(id_,''),live=id_ in live))
def rs(x): return '; '.join(f"{a}-{b}" for a,b in x)
print("| id | § | TASK 58 lines | current lines | TASK 58 words | current words | quantity | math/eq | cite | reg | ⟦⟧ | [VERIFY…] | change since TASK 58 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['id']} | {r['sec']} | {rs(r['old'])} | {rs(r['new'])} | {r['oldw']} | {r['neww']} | {r['q']} | {r['eq']} | {r['ci']} | {r['rg']} | {r['va']} | {r['mk']} | {r['chg'] or 'none'} |")
def tot(f): return sum(r['neww'] for r in rows if f(r))
def toto(f): return sum(r['oldw'] for r in rows if f(r))
print()
print('A new',tot(lambda r:r['tier']=='A'),'A old',toto(lambda r:r['tier']=='A'))
print('B new',tot(lambda r:r['tier']=='B'),'B old',toto(lambda r:r['tier']=='B'))
print('A+B new',tot(lambda r:True),'old',toto(lambda r:True))
print('live new',tot(lambda r:r['live']),'live old',toto(lambda r:r['live']))
print('ceiling new',tot(lambda r:not r['live']),'old',toto(lambda r:not r['live']))
print('free new',tot(lambda r:r['free'] and not r['live']),'A free',tot(lambda r:r['free'] and r['tier']=='A'),'B free',tot(lambda r:r['free'] and r['tier']=='B'))
print('free old(recompute on old text not done)')
body=27718
print('ceiling share',100*tot(lambda r:not r['live'])/body)
print('8 largest A:',sorted([(r['neww'],r['id']) for r in rows if r['tier']=='A'],reverse=True)[:8], sum(sorted([r['neww'] for r in rows if r['tier']=='A'],reverse=True)[:8]))
json.dump(rows,open(''+__import__('os').environ.get('T101_SCRATCH','/tmp/t101')+'/p3rows.json','w'))
