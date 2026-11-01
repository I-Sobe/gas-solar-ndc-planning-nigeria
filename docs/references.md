§3.6 reference block

ε-constraint method — Haimes, Y.Y., Lasdon, L.S. and Wismer, D.A. (1971) "On the bicriterion formulation of the problems of integrated system identification and system optimization", IEEE Transactions on Systems, Man, and Cybernetics, SMC-1, pp. 296–297. DOI 10.1109/TSMC.1971.4308298. 
springer
scholarhub

Two notes. That original is a two-page correspondence item, which is thin as the sole citation for the method your whole bi-objective analysis rests on; the standard practice is to cite it alongside the fuller treatments in Cohon (1978) and Chankong and Haimes (1983). And an optimal solution to the plain ε-constraint formulation is not guaranteed to be efficient — it may be weakly efficient — which is why Ehrgott and Ruzika (2008) and Mavrotas (2009) propose modified formulations. Your per-point non-dominance verification is the practical answer to exactly that weakness, so citing one of those two at that sentence in §3.6.6 turns a procedural detail into a methodologically motivated choice. 
arxiv
arxiv

Weighted-sum non-convexity — Das, I. and Dennis, J.E. (1997) "A closer look at drawbacks of minimizing weighted sums of objectives for Pareto set generation in multicriteria optimization problems", Structural Optimization, 14(1), pp. 63–69. Its abstract states directly that the weighted-sum method reaches all parts of the Pareto set only when the Pareto curve is convex, and gives the geometric argument for why — which is precisely the sentence §3.6.6 needs. 
unibo
unibo

Sobol indices — Sobol', I.M. (2001) "Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates", Mathematics and Computers in Simulation, 55, pp. 271–280; the issue is 55(1–3). 
knowledgeminer
theoj

Saltelli estimator — Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M. and Tarantola, S. (2010) "Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index", Computer Physics Communications, 181(2), pp. 259–270. 
rdocumentation

If the implementation uses SALib, cite it too — SALib's own paper names Sobol' (2001), Saltelli (2002) and Saltelli et al. (2010) as the estimators it implements, so the chain from your code to the method is then explicit. Have Code confirm which library the Sobol runner actually uses. 
theoj

Conditional Value at Risk — Rockafellar, R.T. and Uryasev, S. (2000) "Optimization of conditional value-at-risk", Journal of Risk, 2(3), pp. 21–42.

Check the issue number. Uryasev's own faculty page gives Journal of Risk, Vol. 2, No. 3, 2000, while a bibliographic record gives Volume 2, Number 1, pages 21–42. The pagination agrees; the issue does not. 2(3) is the form in general use and is on the author's own page, but verify against the paper itself before it goes in. 
stonybrook
casact

The wait-and-see bound — Madansky, A. (1960) "Inequalities for Stochastic Linear Programming Problems", Management Science, 6(2), pp. 197–204, DOI 10.1287/mnsc.6.2.197. It is the right citation and worth reading rather than just citing: it treats exactly your construction — replacing a random right-hand side by its expected value and solving the resulting program, with bounds on the gap, and the relation to the problem where one observes the realisation first and then solves. That second problem is your wait-and-see leg. 
repec
repec

Birge and Louveaux — I could not verify this one. It is the textbook for the $WS \le RP \le EEV$ chain and the VSS and EVPI definitions, and I believe it to be Introduction to Stochastic Programming, 2nd edition, Springer, 2011. Confirm the edition and year from a copy before citing. Everything else above came from a source I read; this did not.