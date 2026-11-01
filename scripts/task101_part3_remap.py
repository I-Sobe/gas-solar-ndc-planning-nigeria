import sys,difflib,re,json
sys.path.insert(0,'scripts')
import task58_passages as P
from task58_measure import load,classify,words
S=''+__import__('os').environ.get('T101_SCRATCH','/tmp/t101')+'/'
old=open(S+'ch3_old.md',encoding='utf-8').read().split('\n')
new=load('docs/chapters/CHAPTER_3_assembled.md')
sm=difflib.SequenceMatcher(None,old,new,autojunk=False)
m={}  # old idx -> new idx for equal lines
chg={}
for tag,i1,i2,j1,j2 in sm.get_opcodes():
    if tag=='equal':
        for k in range(i2-i1): m[i1+k]=j1+k
    else:
        for k in range(i1,i2): chg[k]=(tag,j1,j2)
cls=classify(new)
out=[]
for tier,rows in (('A',P.A),('B',P.B)):
    for id_,sec,rng,q,note in rows:
        newr=[];changed=0;oldw=0
        for a,b in rng:
            oldw+=sum(words(old[i]) for i in range(a-1,b))
            idx=[i for i in range(a-1,b)]
            ch=[i for i in idx if i not in m]
            changed+=len(ch)
            mm=[m[i] for i in idx if i in m]
            # expand to changed blocks
            lo=[m[i] if i in m else chg[i][1] for i in idx]
            hi=[m[i] if i in m else chg[i][2]-1 for i in idx]
            newr.append((min(lo)+1,max(hi)+1))
        # measure
        p=t=e=0
        for a,b in newr:
            for i in range(a-1,b):
                k=cls[i][0];w=words(new[i])
                if k=='table':t+=w
                elif k=='eq':e+=w
                elif k in('prose','code'):p+=w
        # section of first line
        out.append(dict(id=id_,tier=tier,sec_old=sec,sec_new=cls[newr[0][0]-1][1],old=rng,new=newr,changed_lines=changed,oldw=oldw,prose=p,table=t,eq=e,q=q))
json.dump(out,open(S+'remap.json','w'))
for o in out: print(o['id'],o['sec_old'],o['sec_new'],o['old'],'->',o['new'],'chg',o['changed_lines'],'oldw',o['oldw'],'new p/t',o['prose'],o['table'])
