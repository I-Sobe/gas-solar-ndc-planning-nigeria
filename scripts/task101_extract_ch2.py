import docx,sys
d=docx.Document('docs/chapters/Chapter_2_thesis_submission.docx')
t=[p.text for p in d.paragraphs]
tab=[]
for tb in d.tables:
    for r in tb.rows:
        tab.append(' | '.join(c.text for c in r.cells))
open(sys.argv[1],'w').write('\n\n'.join(t)+'\n\n'+'\n'.join(tab))
print(len(' '.join(t).split()), len(' '.join(tab).split()), len(d.tables))
