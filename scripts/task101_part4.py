import sys
sys.path.insert(0,'scripts')
import task101_overlap as T
L=T.FILES['ch3'].read_text().split('\n')
def w(rs): return sum(len(L[i-1].split()) for a,b in rs for i in range(a,b+1))
items=[
('3.4.2','timing-correction measurement (25.86→24.99 bn, 3.39%, 875,593,215/218; VERIFY-marked)',[(1782,1796)]),
('3.4.4','B* 10.27 bn from a blended no-cap sweep against the vintage requirement 5.52 bn',[(1931,1944)]),
('3.4.5','six-runner coverage-margin re-measurement (⟦+0.0014%⟧, ⟦40⟧ cells, ⟦14.3%⟧, ⟦23.1%⟧), measured 2026-10-04',[(2008,2029)]),
('3.5.4','total-against-annual reliability comparison (⟦38%⟧, ⟦49⟧, ⟦three⟧ years, ⟦20.2%⟧ in 2054)',[(2420,2435)]),
('3.5.5','carbon-price sweep self-generation ⟦2.3%⟧, ⟦49.8⟧→⟦48.7⟧ TWh',[(2453,2465)]),
('3.5.6','out-of-boundary surface: the "favours the finding" paragraph (⟦0.68%⟧, eleven of fifteen)',[(2493,2499)]),
('3.5.6','out-of-boundary surface: intro, 13-row table, count (⟦Seven⟧ of fifteen) and the two readings',[(2516,2550)]),
('3.6.4','correction-epoch percentages ⟦3.39%⟧, ⟦5.11%⟧, ⟦6.92%⟧, ⟦70.9%⟧ beside the vintage anchor',[(2681,2694)]),
('3.6.7','Sobol shares ⟦87.3⟧, ⟦90.3⟧ and bootstrap interval ⟦0.718⟧ to ⟦1.014⟧',[(2776,2785)]),
('3.6.9','"2 of 20 → 5 of 20" and "0.05 → 0.15" (original-resolution figures superseded; extended-grid figures vintage)',[(2882,2888)]),
('3.7.7','item 5 capital-frontier counts (⟦11, 7, 1⟧; ⟦9, 9, 1⟧)',[(3456,3460)]),
('3.7.7','item 8 carbon-price sweep (⟦1.8⟧ per cent; solar passes half of 2035 generation between 20 and 30)',[(3469,3475)]),
('3.3.6','regression-configuration storage ⟦43,540⟧→⟦74,435⟧ (⟦70.9%⟧) and the identity "under all three capital timing conventions"',[(1528,1539)]),
('3.6.4','regression anchor ⟦19.853⟧ bn and ⟦74,435.22⟧ MWh',[(2663,2665)]),
]
for s,d,rs in items: print(f"| {s} | {', '.join(f'{a}-{b}' for a,b in rs)} | {w(rs)} | {d} |")
