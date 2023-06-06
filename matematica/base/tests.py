# from django.test import TestCase

# Create your tests here.
from icecream import ic
ic.configureOutput(prefix="> ", includeContext=True)

# from mez import calcolo,
from mez.calcolo import *
import time

a =  ( ( Factor(InfComp_num(1.0,1),5.0).solve(InfComp_num(0.0,1,False)) + Factor(InfComp_num(3.0,1),1.0).solve(InfComp_num(0.0,1,False)) + InfComp_num(2.0,1) ) ) / \
     ( (Factor(InfComp_num(1.0,1),2.0).solve(InfComp_num(0.0,1,False)) ) )
ic(a)

a = ( (Factor(InfComp_num(1.0,1)).solve(InfComp_num(0.0,1,False)) ) )
ic(a)




# def A():
#     while True:
#
#         for i in range(8):
#             time.sleep(0.3)
#             print(i)
#             if i == 3:
#                 return
#
# A()




a = ( ( Factor(InfComp_num(1.0,1),1.0) + InfComp_num(1.0,1) ) * ( Factor(InfComp_num(1.0,1),1.0) + InfComp_num(-2.0,1) ) ) / \
    ( ( Factor(InfComp_num(-2.0,1),1.0) + InfComp_num(1.0,1) ) * ( Factor(InfComp_num(1.0,1),1.0) + InfComp_num(-2.0,1) ) )
ic(a)

# a = ( ( Factor(InfComp_num(1.0,1),1.0).solve(InfComp_num(-1.0,1,False)) + InfComp_num(-5.0,1) ) )
# a = ( ( Factor(InfComp_num(1.0,1),1.0).solve(InfComp_num(-1.0,1, False)) + InfComp_num(1.0,1) ) )
a =  InfComp_num(1.0,1) * InfComp_num(-1.0,1, False)
ic(a)

a = {"a": 1, "b": 2, "c": 3}
b = {k: v for k, v in a.items()}
ic(b)

a = InfComp_num(2, 1, True)
ic(a)

a = ( ( InfComp_num(1.0,1) ) * ( Factor(InfComp_num(-1.0,1),5.0).solve(InfComp_num(0.0,1)) + Factor(InfComp_num(6.0,1),1.0).solve(InfComp_num(0.0,1)) + InfComp_num(-30.0,1) ) ) / ( (
                       InfComp_num(1.0,1) ) * ( Factor(InfComp_num(-1.0,1),5.0).solve(InfComp_num(0.0,1)) + Factor(InfComp_num(6.0,1),1.0).solve(InfComp_num(0.0,1)) + InfComp_num(-30.0,1) ) )

ic(a)

a = Expression(list_repr=[
    [[InfComp_num(5)], [InfComp_num(4)]],
    [[InfComp_num(1), InfComp_num(1)]]
])

a = a.eval()
ic(a)


1/0
a = -8
b = a**(1/3)
ic(b)



a = "( ( Factor(InfComp_num(1.0,1),1.0).solve(InfComp_num(1.0,1)) + InfComp_num(1.0,1) ) * ( Factor(InfComp_num(1.0,1),1.0).solve(InfComp_num(1.0,1)) + InfComp_num(-1.0,1) ) ) / ( ( Factor(InfComp_num(1.0,1),4.0).solve(InfComp_num(1.0,1)) + InfComp_num(-1.0,1) ) )"
a = eval(a)
ic(a)
ic(a.to_str())




a = [1,2,3]
b = [13,32,32]
ic(a + b)

a = {1,2,3,4,5}
b = {4,5,6,7,8,9}
c = a ^ b
ic(c)
a.add(1)
ic(a)



a = calcolo.Expression(list_repr=[
    [[calcolo.Factor(1)]],
    [[calcolo.Factor(1)], ['(Factor(InfComp_num(1.0,1),2.0) / Factor(InfComp_num(1.0,1),2.0))', '(Factor(InfComp_num(-1.0,1),1.0) / Factor(InfComp_num(1.0,1),2.0))']]
]
)
ic(a.list_repr)

c = []
a = 1
c.append(a)
a = 2
c.append(a)
ic(c)


a = 1.0e-18
ic(a+1)

1/0

a = "(x^2 - 1) / (10x^2 + 30x - 40)"
steps, result = calcolo.calc_lim(a, lim="1")
ic(result)




# a = "3x^2 - 7x - 6"
# a = "x^2 + 8x + 15"
# a = "3x^2 + 8x - 3"
# a = "x^2 + 5x + 6"
a = "10x^3 + 30x^2 - 40x"
b = calcolo.create_evaluable_exp(a)
c = calcolo.Expression(b)
ic(c.list_repr)

d = c.to_factorized()
ic(d.to_str())

1/0

a = 36
b = a**0.5
ic(b)

a = [[1], [2]]
a.insert(0, "a")
a.insert(0, "")
ic(a)

a = - (calcolo.Inf(positivity=True))**4
ic(a.to_html())


a = calcolo.InfComp_num(0,1,True)
b = 1
c = a*1
ic(str(c))

# a = [1,2,3]
# a += 3
# ic(a)

a = 1
if 1 in a:
    print("SI")

str_ = "10x / (x^4 - x^3 - 30)"
expr = calcolo.Expression(calcolo.create_evaluable_exp(str_))
expr = expr.add_solve(-1)
print(expr.to_html())

1/0

a, _, _, _ = calcolo.calc_lim("10x / (x^4 - x^3 - 2x^2)", -2)
ic(a)



# a = "x**3 + 4"
# b = lambda x: eval(a)
# c = b(2)
# print(c)

# a = -1 * 3**2
# ic(a)


1/0

# a, _, base_f, a = calcolo.calc_lim("(2/3x)", 1)
# ic(a)
# a, _, base_f, a = calcolo.calc_lim("(3x^2 + x + 30) / (x^2 - 4)", 1)
# ic(a)
a = calcolo.create_evaluable_exp("(3x^2 + x + 30) / (x^2 - 4)")
ic(a)


# l = base_f.to_str("lambda")
ic(l)
f_lambda = lambda x: eval(l)
a = f_lambda(1)
ic(a)

a = calcolo.InfComp_num(4)
ic(a)