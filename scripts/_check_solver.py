from pyomo.environ import SolverFactory

s = SolverFactory("cbc")
print("cbc available:", s.available())
print("cbc executable:", s.executable())

s2 = SolverFactory("glpk")
print("glpk available:", s2.available())
print("glpk executable:", s2.executable())

s3 = SolverFactory("highs")
print("highs available:", s3.available())
print("highs executable:", s3.executable())
