import copy
from math import sqrt

from .Mat import Factor, InfComp_num, Inf, alphabet, expand_parenthesis, Expression, ic

eval_classes_dict = {"Factor": Factor, "InfComp_num": InfComp_num, "Inf": Inf}


# FUNZIONI

def lim_to_html(lim):
    math_tag = '<math xmlns = "http://www.w3.org/1998/Math/MathML">'
    return f"{math_tag}" \
               "<munder>" \
                   "<mrow><mi>lim</mi></mrow>" \
                   f"<mrow><mi>x</mi><mo>&rarr;</mo>{lim.to_html()}</mrow>" \
               "</munder>" \
           "</math>"


def str_to_infcomp(str_):

    def str_to_int(string):
        string = "".join([i for i in string if i.isdigit() or i in ["-", "+"]])
        if not string[-1].isdigit():
            string = string[:-1]
        return string

    str_ = str(str_)
    str_ = str_.split("/")

    nd_epsilons = [None, None]
    eps_signs = {"+": True, "-": False}
    last_char = [str_[0][-1], str_[1][-1] if len(str_) > 1 else 1]

    if last_char[0] in eps_signs:
        nd_epsilons[0] = eps_signs[last_char[0]]
    if last_char[1] in eps_signs:
        nd_epsilons[1] = not eps_signs[last_char[1]]  # essendo al denumeratore viene invertita: 1/1- = 1+

    replacements = {None: 0, False: -1, True: 1}
    nd_epsilons = [replacements[i] for i in nd_epsilons]
    epsilon_num = nd_epsilons[0] + nd_epsilons[1]
    epsilon = InfComp_num.num_as_state(epsilon_num)

    num = str_to_int(str_[0])
    den = 1 if len(str_) == 1 else str_to_int(str_[1])

    return InfComp_num(num, den, epsilon)


def str_to_factor(x_str):
    for chr in alphabet:
        x_str = x_str.replace(chr, "x")  # fa in modo che si può usare qualsiasi lettera al posto di x

    if x_str[0] == "x":
        x_str = x_str.replace("x", "1x")
    x_str = x_str.replace("-x", "-1x")
    x_split = x_str.split("x")

    coefficient = str_to_infcomp(x_split[0])
    exponent = float(x_split[1][1:] if x_split[1] != "" else 1)

    return Factor(coefficient, exponent)


def lim_to_obj(lim_str):
    inf_sym = "∞"

    lim_str = lim_str.replace("inf", inf_sym)

    if inf_sym in lim_str:
        return Inf(False) if "-" in lim_str else Inf(True)
    else:
        return str_to_infcomp(lim_str)



# CALC LIM

def create_evaluable_exp(base_expression, x=None, solve=False):
    # STR -> EVAL EXPRESSION VALIDA
    base_expression = " " + base_expression
    base_expression = expand_parenthesis(base_expression)
    expression_split = base_expression.split(" ")

    valid_expression_split = []
    for i in expression_split:
        if "x" in i:
            factor = str_to_factor(i)
            valid_expression_split.append(f"{factor}.solve({x})" if solve else f"{factor}")
        elif any([j.isdigit() for j in i]):  # se non ha la x ma contiene un numero
            icn = str_to_infcomp(i)
            valid_expression_split.append(f"{icn}")
        else:
            valid_expression_split.append(i)


    return " ".join(valid_expression_split)


def calc_lim(expression, lim):
    print(f"Solving expression: {expression}, lim: {lim}")
    steps = []
    # str_exp = expand_parenthesis(expression)
    eval_expr = Expression(create_evaluable_exp(expression))
    lim = lim_to_obj(lim)

    steps.append(HTML_step(desc="Scrittura dell'espressione", expr=eval_expr, lim=lim))

    base_form_expr = eval_expr.to_base_form()

    steps.append(HTML_step(desc="Riscrittura in forma normale", expr=base_form_expr, lim=lim))
    # base_form_expr_str = base_form_expr.to_str()


    def substitution(expression, x, append_steps=True, return_expr=False):
        solve_expr = expression.add_solve(x)

        # solve_expr = add_solve(expression, x)

        fs_steps = []
        if append_steps:
            fs_steps.append(HTML_step(desc="sostituzione delle variabili con il limite", expr=solve_expr))

        return solve_expr.eval(return_expr=return_expr), fs_steps

    # primo step, arriveremo a una forma di indecisione, inf, o il risultato finale (in caso non ci siano forme di indecisione o inf)
    substitution_result, fs_steps = substitution(base_form_expr, lim, return_expr=False)
    steps.append(fs_steps[0])

    steps.append(HTML_step(desc="Calcolo del risultato della sostituzione", expr=substitution_result))

    result_IF_type = substitution_result.indeterm_type if type(substitution_result) == Expression else ""

    factorized_expr = None

    # RESULT = INF, DETERMINARE IL SEGNO
    if type(substitution_result) == Inf and substitution_result.positivity == None:

        results_list = [[],[]]
        for i in range(len(results_list)):
            # results_list[i].append(steps[-1])
            lim = InfComp_num(lim.num, lim.den, False if i == 0 else True)
            results_list[i].append(HTML_step(desc="Riscrittura dell'espressione", expr=base_form_expr, lim=lim))
            result, fs_steps = substitution(base_form_expr, lim)
            results_list[i].append(fs_steps[0])
            results_list[i].append(HTML_step(desc="Calcoli", expr=result))

        flattened_res_list = [i for j in results_list for i in j]
        steps = steps + flattened_res_list

        result = None

    # RESULT = INF / INF, INF - INF
    elif result_IF_type in ["Inf_div_Inf", "Inf_sub_Inf"]:
        steps.append(HTML_step(desc="riscrittura dell'espressione", expr=base_form_expr) )

        def get_max_factor(terms):
            factors = [i for i in terms if type(i) == Factor]
            max_expr = max([i.exponent for i in factors])
            return [i for i in factors if i.exponent == max_expr][0]


        max_fact_x = get_max_factor([i for i in base_form_expr.list_repr[0][0]])
        f_args = [max_fact_x]

        if result_IF_type == "Inf_div_Inf":  # viene scomposto anche il ddenominatore in caso abbiamo inf div inf
            max_fact_y = get_max_factor([i for i in base_form_expr.list_repr[1][0]])
            f_args.append(max_fact_y)

        fact_steps, maxF_factorized_expr = Expression.factorize(list_repr=base_form_expr.list_repr, mode="x * (.../x)", f_args=f_args)
        for step in fact_steps:
            steps.append(HTML_step(desc=step["desc"], expr=step["expr"], lim=lim if step["lim"] else None))

        solve_factorized_expr = maxF_factorized_expr.add_solve(lim, sel_i_tg=[1])
        steps.append(HTML_step("Sostituzione all'interno delle parentesi", expr=solve_factorized_expr))

        result = solve_factorized_expr.eval()
        steps.append(HTML_step("Calcoli", expr=result))

        if len(result.list_repr) > 1 or type(result.list_repr[0][0][0]) == Factor:  # se finisce per essere (n / factor) o (factor / 1)
            solve2_fact_expr = result.add_solve(lim)
            steps.append(HTML_step("Sostituzione dei termini rimanenti con il limite", expr=solve2_fact_expr))

            result = solve2_fact_expr.eval()
            steps.append(HTML_step("Calcoli", expr=result))


    # 0 / 0
    elif result_IF_type == "0_div_0":

        steps.append(HTML_step(desc="riscrizione dell'espressione", expr=base_form_expr))

        factorized_expr = base_form_expr.to_factorized()
        steps.append(HTML_step(desc="scomposizione di numeratore e denominatore", expr=factorized_expr))

        simp_expr = factorized_expr.simplify_fraction()
        steps.append(HTML_step(desc="semplificazione tra numeratore e denominatore", expr=simp_expr))

        solve_expr = simp_expr.add_solve(lim)
        steps.append(HTML_step(desc="sostituzione usando il limite", expr=solve_expr))
        result = solve_expr.eval()
        steps.append(HTML_step(desc="Calcoli", expr=result))


    # nessuna forma di indecisione
    else:
        result = substitution_result


    # DISCONTINUITà IN CASO DI ESPRESSIONE FRATTA

    x_disconts = {"specie2": [], "specie3": []}  # punto in cui c'è una discontinuità: se abbiamo una frazione con lim al num e al den allora quando la lim è 0 al den abbiamo la discontinuità

    den_types = [type(i) for i in base_form_expr.list_repr[1][0]] if len(base_form_expr.list_repr) == 2 else []  # otteniamo tutti i tipi di ogni termine, ci basta un solo termine
        # che sia un fattore per procedere

    if Factor in den_types:  # se al den c'è una x

        # la calcoliamo solo se non abbiamo già scomposto il denominatore (succede solo nella indecisione 0/0)
        sep_factorized_expr = base_form_expr.to_factorized(separation=True)

        y_0s = [[], []]  # conterrà tutti i numeri che rendono y = 0, idx 0 num, idx 1 den

        for i_nd, nd in enumerate(sep_factorized_expr.list_repr):  # selezioniamo il denominatore
            for i_tg, tg in enumerate(nd):
                y_0s[i_nd].append([])

                # 1 TERMINE
                if len(tg) == 1:
                    if type(tg[0]) == Factor:
                        y_0s[i_nd][i_tg].append(InfComp_num(0))  # traduzione: InfComp_num(0) fa in modo che i_nd, i_tg sia = a 0

                # 2 TERMINI, X = N
                elif len(tg) == 2:
                    expo = tg[0].exponent
                    coeff = tg[0].coefficient

                    res = (-tg[1]/coeff).root(expo)
                    if res:
                        if type(res) == list:
                            y_0s[i_nd][i_tg].append(res[0])
                            y_0s[i_nd][i_tg].append(res[1])
                        else:
                            y_0s[i_nd][i_tg].append(res)

                # 3 TERMINI, X1,2 = ...
                elif len(tg) == 3:
                    if tg[0].exponent == 2 and tg[1].exponent == 1 and type(tg[2]) == InfComp_num:
                        a = tg[0].coefficient
                        b = tg[1].coefficient
                        c = tg[2]

                        delta = (b**2 - a*c*4).root(2)

                        if delta:  # solo se il delta non è negativo
                            if type(delta) == list:
                                delta = delta[0]

                            x1 = (-b + delta) / (a*2)
                            x2 = (-b - delta) / (a*2)

                            y_0s[i_nd][i_tg].append(x1)
                            y_0s[i_nd][i_tg].append(x2)


        # numeratori e denominatori, ci serve per sapere se degli elementi del numeratore sono anche nel denominatore
        nd_strIcns = [[str(icn) for tg in y_0s[0] for icn in tg], [str(icn) for tg in y_0s[1] for icn in tg]]

        all_strIcns = set([j for i in nd_strIcns for j in i])

        disconts_2 = set()
        disconts_3 = set()

        for strIcn in all_strIcns:
            num_count = nd_strIcns[0].count(strIcn)
            den_count = nd_strIcns[1].count(strIcn)

            # if strIcn == str(InfComp_num(0)):
            #     if num_count > 0 and den_count > 0:
            #         disconts_3.add(strIcn)
            #     elif den_count > num_count:
            #         disconts_2.add(strIcn)

            # else:
            if num_count == den_count:
                disconts_3.add(strIcn)
            elif den_count > num_count:
                disconts_2.add(strIcn)

        x_disconts["specie2"] = [eval(x) for x in disconts_2]

        x_disconts["specie3"] = []
        factorized_expr = factorized_expr if factorized_expr != None else base_form_expr.to_factorized()
        simplified_expr = factorized_expr.simplify_fraction()
        for x in disconts_3:
            eval_x = eval(x)
            solve_expr = simplified_expr.add_solve(eval_x)
            y_value = solve_expr.eval(return_expr=False)

            x_disconts["specie3"].append([eval_x, y_value])

    return result, steps, base_form_expr, x_disconts, lim


class HTML_step():
    def __init__(self, desc="", lim=None, expr=None):
        math_tag = '<math xmlns = "http://www.w3.org/1998/Math/MathML">'

        self.desc = desc.capitalize()

        self.lim = lim
        if lim:
            self.lim = lim_to_html(lim)
        else:
            self.lim = ""

        # momentaneo
        if type(expr) == Expression:
            self.expr = expr.to_html()
        elif type(expr) in [Factor, InfComp_num, Inf]:
            self.expr = f"{math_tag}{expr.to_html()}</math>"
        else:
            self.expr = expr

