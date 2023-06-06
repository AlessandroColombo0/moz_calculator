import copy
from math import sqrt
import re
from icecream import ic
from icecream import ic
wrap_width = 195
def to_string(obj):
    """ obj: contenuto di quello che viene printato, es: ic("aaa"), "aaa" è obj, ma non è per forza una stringa, può essere anhce un numero, lista, ecc. """
    def element_wrap(wrap_width, text, key=None):
        # se ha una key viene trattato come un dizionario
        string_split = text.split(" ")
        str_split_lengths = [len(i) + 1 for i in string_split]
        newline_idxs = []
        new_str = ""
        newline_i = 0
        n_line_padding = 0
        last_split = 0

        for i_split, split_ in enumerate(string_split):
            if sum(str_split_lengths[i] for i in range(last_split, i_split + 1)) + n_line_padding > wrap_width:
                last_split = i_split
                newline_idxs.append(last_split)
                n_line_padding = 1

        more_newlines = True if len(newline_idxs) > 0 else False

        for i, split_ in enumerate(string_split):
            if more_newlines and i == newline_idxs[newline_i]:
                if not key:  # non dizionario
                    new_str += "\n" + " " + split_ + " "
                else:  # dizionario
                    new_str += "\n" + " " * (len(key) + 6) + split_ + " "

                newline_i += 1
                if newline_i >= len(newline_idxs):
                    more_newlines = False

            else:
                new_str += split_ + " "

        return new_str

    if type(obj) == dict:
        new_str = "{"
        for key, value in obj.items():
            dict_wrap_width = wrap_width - (len(key) + 5)
            new_value = element_wrap(dict_wrap_width, str(value), key=key)

            new_str += f'"{key}": {new_value[:-1]},\n '
        new_str = new_str[:-3] + "}"

    elif type(obj) in [str, list, set, tuple, int, float]:
        new_str = element_wrap(wrap_width, str(obj))

    else:  # ad esempio in caso di numpy ecc
        new_str = str(obj)

    return new_str
ic.configureOutput(prefix="> ", includeContext=True, argToStringFunction=to_string)








blank_char = "⠀"

def is_square(term):
    def is_square_only_nums(num):
        # ic(num)
        num_sqrt = int(num**0.5)  # **0.5: square root
        return num_sqrt * num_sqrt == num

    if type(term) in [int, float, InfComp_num]:
        term = num_to_InfComp(term)

        if term.den == 1:
            return is_square_only_nums(term.num)
        else:
            return False

    elif type(term) == Factor:
        if term.coefficient.den == 1:
            return is_square_only_nums(term.coefficient.num) and term.exponent % 2 == 0  # se il coeff è un quadrato e se l'esponente è divisibile per 2
        else:
            return False


def find_divisors(num, include_one=True):
    if num != 0:
        num = abs(num)
        divisors = []
        if include_one:
            divisors.append(1)

        # quando è pari
        while num % 2 == 0:
            num /= 2
            divisors.append(2)

        # quando non è più pari
        for i in range(3, int(sqrt(num)) + 1, 2):  # la logica è che se un numero non è stato diviso per un numero che c'era prima di esso nell'iterazione allora è impossibile
                # che quel numero non sia un numero primo, lo step è di 2 perchè con 1 si includono tutti i numeri pari. cosa che ovviamente non accade con 2
            if num % i == 0:
                num /= i
                divisors.append(int(i))

        # passo finale
        if num > 2:  # l'idea è che arrivati a questo punto è impossibile avere numeri divisibili per 2, e che il numero rimanente è per forza un numero primo
            divisors.append(int(num))

        return divisors

    else:
        return 1


def mul_list(list_):
    """ [2, 2, 3] -> 12 """

    mul = 1
    for num in list_:
        mul *= num
    return mul


def find_gcd(terms_list, output_type=None):
    """ greatest common divisor """
    nofract = True

    terms_nums = []
    for i in terms_list:
        if type(i) == InfComp_num:
            if i.den == 1:
                terms_nums.append(i.num)
            else:
                nofract = False
        elif type(i) == Factor:
            if i.coefficient.den == 1:
                terms_nums.append(i.coefficient.num)
            else:
                nofract = False
        else:
            terms_nums.append(i)

    numeric_gcd = 1
    if nofract:
        terms_divisors_list = [find_divisors(num, include_one=False) for num in terms_nums]
        common_divisors = terms_divisors_list[0]
        p_common_divisors = [i for i in common_divisors]  # flat common_divs
        for i, divs_list in enumerate(terms_divisors_list[1:]):
            for div in divs_list:
                if div in p_common_divisors:
                    p_common_divisors.remove(div)
                else:
                    pass
            if p_common_divisors == []:
                p_common_divisors = copy.deepcopy(common_divisors)
            else:
                for i in p_common_divisors:  # rimuoviamo quelli che sono rimasti in quanto visto che sono rimasti non sono parte dell'mcm, non essendo condivisi da entrambi i numeri
                    common_divisors.remove(i)
                p_common_divisors = copy.deepcopy(common_divisors)

        common_divisors = copy.deepcopy(p_common_divisors)

        numeric_gcd = mul_list(common_divisors) if common_divisors else 1  # l'gcd è = a 1 se non ci sono divisori in comune

    factor_gcd = 1  # serve per fare in modo che è come se non ci fosse in caso l'if statement sotto non si avvere
    if output_type != "numeric":
        # factor_gcd = Factor(coefficient=1)  # serve per fare in modo che è come se non ci fosse in caso l'if statement sotto non si avvere
        factors = [f for f in terms_list if type(f) == Factor]

        if len(factors) == len(terms_list):
            min_degree = min([f.exponent for f in factors])
            factor_gcd = Factor(exponent=min_degree, coefficient=1)


    if output_type == "numeric":
        return numeric_gcd
    if output_type == "separated":
        return InfComp_num(numeric_gcd), factor_gcd
    else:
        return InfComp_num(numeric_gcd) * factor_gcd


def find_lcm(nums_couple):
    mul = nums_couple[0] * nums_couple[1]
    gcd = find_gcd(nums_couple, output_type="numeric")
    return mul/gcd


def num_to_InfComp(y):
    """ compatibilità con int e float, es: rende possibile x^2 * 2 """
    if type(y) in [int, float]:
        return InfComp_num(y)
    else:
        return y

alphabet = [chr(i+97) for i in range(26)] + [chr(i+65) for i in range(26)]  # alfabeto lowercase + alfabeto uppercase


def add_solve(exp, x):
    """
    input:  exp: ( InfComp_num(1,None) / Factor("0.3333333333333333x^3") ) + InfComp_num(1.0,None)
            x:   Inf(True)
    output: ( InfComp_num(1,None) / Factor("0.3333333333333333x^3").solve(Inf(True)) ) + InfComp_num(1.0,None)
    """
    exp = f" {exp} "
    exp_split = exp.split(" ")
    # ic(exp_split)
    factors = [i for i in exp_split if "Factor(" in i]
    for factor in factors:
        exp = exp.replace(f" {factor} ", f" {factor}.solve({x}) ")

    return exp


def separate_ops_and_terms(exp):
    # ic(exp)
    exp = str(exp).strip()
    # ic(exp)
    # ic(exp)
    exp = retract_parenthesis(exp)  # compattare tutte le parentesi per poi isolare i singoli elementi

    step_split = exp.split(" ")
    # ic(step_split)
    objs = [exp for exp in step_split if any([obj in exp for obj in str_classes_list])]
    p_step_split = [cleanse_exp(i, opposite=True) if i in objs else i for i in step_split]
    # ic(p_step_split)
    p_step_split = [i for i in p_step_split if i != ""]
    # ic(p_step_split)
    # ic(exp)

    if len(p_step_split) > 0:
        # idx_counter = 0 if p_step_split[0] in "*/" else 1  # se p_step_split inizia con * o / allora il primo elemento deve essere un termine
        # ic(idx_counter)

        masked_step_list = []  # solo i segni, dove c'è la virgola vuol dire che lì ci va un comp
        skip = False
        for i, char in enumerate(p_step_split[:-1]):

            # if p_step_split[i+1] != "":
            if i != 0 and p_step_split[i-1][-1] == ")" and p_step_split[i+1][0] == "(":
                del masked_step_list[-1]  # lo avevamo aggiunto prima quindi lo togliamo
                masked_step_list.append(f"{p_step_split[i-1]} {p_step_split[i]} {p_step_split[i+1]}")  # unione delle stringhe
                skip = True

            elif p_step_split[i+1][0] == "(":
                masked_step_list.append(f"{p_step_split[i]} {p_step_split[i+1]}")  # unione delle stringhe
                skip = True

            elif i != 0 and p_step_split[i-1][-1] == ")" and skip == False:
                masked_step_list.append(f"{masked_step_list[-1]} {p_step_split[i]}")  # unione delle stringhe
                del masked_step_list[-2]

            elif skip == False:
                masked_step_list.append(p_step_split[i])

            else:
                skip = False

        if skip == False:  # se prima era True vuol dire che ha già preso l'ultimo elemento
            masked_step_list.append(p_step_split[-1])
        """ p_step_list = ['( ', '-', ' )', '/', '( ', '+', ' )']   maked_step_list = ['( ', '-', ' ) / ( ', '+', ' )']  """
    else:
        masked_step_list = []

    return masked_step_list, objs


def concatenate_lists(list1, list2):
    # ic(list2)
    # ic(list1)
    """  list1 = ['f', 'o', 'o'], list2 = ['hello', 'world']
    out: ['f', 'hello', 'o', 'world', 'o'] """

    result = [None for _ in range(len(list1) + len(list2))]
    result[::2] = list1
    result[1::2] = list2

    result = [i for i in result if i != None]
    return result


def exp_to_sorted_comps(expression, return_signs=True):
    # ic(expression)
    """ "( ( InfComp_num(1, None) - Factor(10x^3) ) - ( Factor(5x^1) ) )"  ->  [InfComp_num(1, None), '+', Factor("-10x^3"), '+', Factor("-5x^1")] """

    ops, terms = separate_ops_and_terms(expression)
    # ic(terms)
    # ic(ops)

    if len(ops) > len(terms):
        expression_list = concatenate_lists(ops, terms)
    else:
        expression_list = concatenate_lists(terms, ops)
    # ic(expression_list)
    """ expression_list: InfComp_num(1, None), '-', Factor("10x^3"), '-', Factor("5x^1") """


    expression_list = [eval(cleanse_exp(i)) if any(j in i for j in str_classes_list) else i for i in expression_list]
    # ic([str(i) for i in expression_list])
    # expression_list = [i.replace("(", "").replace(")", "") if type(i) == str else i for i in expression_list]
    expression_list = [i for i in expression_list if type(i) in classes_list or i.strip() != ""]


    in_par_idxs = []
    opening_par_idxs = []
    closing_par_idxs = []
    double_pars_idxs = []
    open = False
    for i, comp in enumerate(expression_list):
        # ic(comp)
        if type(comp) == str and "( " in comp:
            open = True
            opening_par_idxs.append(i)
            if ")" in comp:
                double_pars_idxs.append(i)

        elif type(comp) != str and open:
            in_par_idxs.append([opening_par_idxs[-1], i])

        elif type(comp) == str and " )" in comp:
            open = False
            closing_par_idxs.append(i)

    deletion_idxs = []
    append_plus_at_idx = []

    for opening_idx in opening_par_idxs:
        if "-" in expression_list[opening_idx]:
            for par_idx, in_par_idx in in_par_idxs:
                if par_idx == opening_idx:
                    expression_list[in_par_idx] = expression_list[in_par_idx] * -1

        deletion_idxs.append(opening_idx)
        if opening_idx not in double_pars_idxs:
            deletion_idxs.append(closing_par_idxs[0])
            del closing_par_idxs[0]
        else:
            append_plus_at_idx.append(par_idx-1)

    expression_list = [comp for i, comp in enumerate(expression_list) if i not in deletion_idxs]
    # ic(expression_list)
    for i in append_plus_at_idx:
        expression_list.insert(i, "+")

    # se ci sono 2 oggetti vicini che nons ono stringhe deve esserci un + in mezzo
    # for i in range(len(expression_list)):
    #     if type(expression_list[i]) in classes_list and type(expression_list[i+1]) in classes_list:

    # ic([str(i) for i in expression_list])

    # riscrittura dell'espressione usando solo i + e mettendo i - ai coefficienti, es: 1 - Factor(2x) -> 1 + Factor(-2x), in questo modo è tutto uniforme
    terms = []
    for i, elem in enumerate(expression_list):
        if i != len(expression_list) - 1 or len(expression_list) == 1:  # ultimo elemento
            if elem in ["-", "(-"]:
                terms.append(expression_list[i+1] * -1)
            elif elem in ["+", "(+"]:
                terms.append(expression_list[i+1])

            elif i == 0:  # primo elemento
                terms.append(elem)
                # ic(elem)
    """ [InfComp_num(1, None), Factor("-10x^3"), Factor("-5x^1")] """


    nums = [i for i in terms if type(i) != Factor]
    factors = [i for i in terms if type(i) == Factor]

    sorting_dict = {idx: fact.exponent for idx, fact in enumerate(factors)}
    sorting_dict = dict(sorted(sorting_dict.items(), key=lambda x: x[1], reverse=True))

    exp_list = []
    for idx in list(sorting_dict.keys()):
        exp_list.append(factors[idx])
        if return_signs:
            exp_list.append("+")

    for i in nums:
        exp_list.append(i)
        if return_signs:
            exp_list.append("+")

    if return_signs:
        del exp_list[-1]  # ultimo +

    return exp_list


def expand_parenthesis(string):
    string = string.replace("(", "( ").replace(")", " )")
    string = re.sub("\s+", " ", string)  # sostituisce i whitespace multipli con uno solo, è utile in caso la linea sopra aggiunga sequenze di whitespaces
    return string

def retract_parenthesis(string):
    string = re.sub("\s+", " ", string)  # sostituisce i whitespace multipli con uno solo
    string = string.replace("( ", "(").replace(" )", ")")
    return string



def cleanse_exp(exp, opposite=False):
    """ opposite=False  ( ( 1 - ) ) ) ) ) -> ( ( 1 - ) )
        opposite=True   ( ( 1 - ) ) ) ) ) -> ) ) ) """

    exp = expand_parenthesis(exp)

    opening_p = exp.count("( ")
    closing_p = exp.count(" )")
    opposite_parenthesis = []

    if opening_p != closing_p:
        p = "( " if opening_p > closing_p else " )"

        exp = exp if p == "( " else "".join(list(reversed(exp)))

        exp_list = [i for i in exp]  # in caso la p è ) dobbiamo fare le cose al contrario, visto che le parentesi extra sono a
        d = abs(closing_p - opening_p)

        for _ in range(d):
            exp = "".join(exp_list)  # viene ricreata perchè sono stati eliminati dei pezzi da exp_list e quindi si crea uan discrepanza tra le due
            for i, char in enumerate(exp[:-1]):
                if exp[i:i+2] == p:
                    if opposite:
                        opposite_parenthesis.append("".join(exp_list[i:i+2]))
                    del exp_list[i:i+2]
                    break

        exp_list = exp_list if p == "( " else list(reversed(exp_list))
        exp = "".join(exp_list)

    if opposite:
        return "".join(opposite_parenthesis)
    return exp


# INFCOMPNUM

class InfComp_num():
    """ un numero normale che però può essere diviso per 0 ecc. + compatibilitàc on epsilon"""

    def __init__(self, num, den=1., epsilon=None):
        self.num = float(num)
        self.den = float(den)
        self.epsilon = epsilon  # True: 0+, None: 0, False: 0-

        epsilon_as_num = {True: 1, None: 0, False: -1}
        self.epsilon_as_num = epsilon_as_num[epsilon]
        self.num_representation = self.epsilon_as_num if num == 0 else (1 if self.as_decimal() > 0 else -1)

    @staticmethod
    def get_opposite_epsilon(epsilon):
        opp = {True: False, None: None, False: True}
        return opp[epsilon]

    @staticmethod
    def get_epsilon_as_num(epsilon):
        epsilon_as_num = {True: 1, None: 0, False: -1}
        return epsilon_as_num[epsilon]

    @staticmethod
    def num_as_state(num):
        epsilon = True if num > 0 else ""
        epsilon = None if num == 0 else epsilon
        epsilon = False if num < 0 else epsilon
        return epsilon

    @staticmethod
    def adapt_fraction(x, y):
        """ es: input:  x: 2/3, y: 1/2
                output: x: 4/6 + 6/6 """
        lcm = find_lcm([x.den, y.den])
        x_mul = lcm / x.den
        y_mul = lcm / y.den

        return [InfComp_num(num=x.num*x_mul, den=lcm, epsilon=x.epsilon), InfComp_num(num=y.num*y_mul, den=lcm, epsilon=y.epsilon)]


    def __str__(self):
        den = f",{self.den}" if self.den != 1 else ",1"
        epsilon = f",{self.epsilon}" if self.epsilon != None else ""
        return f"InfComp_num({self.num}{den}{epsilon})"

    def fancy_str(self):
        num = int(self.num) if int(self.num) == self.num else round(self.num, 3)
        den = int(self.den) if int(self.den) == self.den else round(self.den, 3)

        den = "" if den == 1 else f"/{den}"
        sign = "⁺" if self.epsilon == True else ""
        sign = "⁻" if self.epsilon == False else sign
        p1 = "(" if sign != "" or self.den != 1 else ""
        p2 = ")" if sign != "" or self.den != 1 else ""
        return f"{p1}{num}{den}{p2}{sign}"

    def to_html(self):
        den = int(self.den) if int(self.den) == self.den else self.den
        num = int(self.num) if int(self.num) == self.num else self.num

        epsilon_pre = "<mn>"
        epsilon_after = f"</mn>"
        if self.epsilon != None:
            epsilon_pre = "<msup><mi>"
            epsilon_after = f"</mi><mn>{'+' if self.epsilon == True else '-'}</mn></msup>"

        if self.den != 1:

            return f"<mfrac>" \
                   f"<mrow>{epsilon_pre}{num}{epsilon_after}</mrow>" \
                   f"<mrow><mn>{den}</mn></mrow>" \
                   f"</mfrac>"
        else:
            return f"{epsilon_pre}{num}{epsilon_after}"


    def to_lambda(self):
        return f"{self.as_decimal()}"


    def simplify(self):
        """ -6/-12 -> 1/2 """

        if self.num != 0:
            gcd = find_gcd([self.num, self.den], output_type="numeric")
            if gcd != 1:
                resulting_icn = InfComp_num(self.num/gcd, self.den/gcd, self.epsilon)
            else:
                resulting_icn = self

            if resulting_icn.num < 0 and resulting_icn.den < 0:
                resulting_icn = InfComp_num(resulting_icn.num*-1, resulting_icn.den*-1, self.epsilon)
            elif resulting_icn.den == -1:
                resulting_icn = InfComp_num(resulting_icn.num*-1, 1, self.epsilon)

            return resulting_icn

        else:
            return InfComp_num(0, epsilon=self.epsilon)

    def as_decimal(self):
        return self.num / self.den


    def __abs__(self):
        return InfComp_num(abs(self.num), abs(self.den), self.epsilon)

    def __neg__(self):
        return InfComp_num(self.num*-1, self.den, self.get_opposite_epsilon(self.epsilon))

    def __add__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            x, y = self.adapt_fraction(self, y)
            epsilon_num = x.epsilon_as_num + y.epsilon_as_num
            epsilon = self.num_as_state(epsilon_num)

            return InfComp_num(x.num + y.num, x.den, epsilon).simplify()

        elif type(y) == Factor:
            return ExpressionString(f"{self} + {y}")

        elif type(y) == Inf:
            return Inf(positivity=y.positivity)

        elif type(y) == ExpressionString:
            return ExpressionString(f"{self} + {y}")

    def __sub__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            x, y = self.adapt_fraction(self, y)
            epsilon_num = x.epsilon_as_num - y.epsilon_as_num
            epsilon = self.num_as_state(epsilon_num)

            return InfComp_num(self.num - y.num, x.den, epsilon).simplify()

        elif type(y) == Factor:
            return ExpressionString(f"{self} - {y}")

        elif type(y) == Inf:
            return Inf(positivity=self.num_as_state(y.num_representation * -1))  # opposto di add, per farlo ci basta moltiplicare per -1 (cambiando il segno)

        elif type(y) == ExpressionString:
            return ExpressionString(f"{self} - {y}")

    def __mul__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            mul_res = self.num * y.num

            if self.num_representation > 0 and y.num_representation > 0:  # 1 1
                epsilon = self.num_as_state(self.epsilon_as_num + y.epsilon_as_num)
            elif self.num_representation < 0 and y.num_representation < 0:  # -1 -1
                epsilon = self.get_opposite_epsilon(self.num_as_state(self.epsilon_as_num + y.epsilon_as_num))
            elif self.num_representation < 0 and y.num_representation > 0:  # -1 1
                epsilon = self.num_as_state(self.epsilon_as_num - y.epsilon_as_num)
                ic(epsilon)
            elif self.num_representation > 0 and y.num_representation < 0:  # 1 -1
                epsilon = self.num_as_state(y.epsilon_as_num - self.epsilon_as_num)
            else:  # 0
                epsilon = None

            return InfComp_num(mul_res, self.den * y.den, epsilon).simplify()

        elif type(y) == Factor:
            coefficient = (self * y.coefficient).simplify()
            return Factor(InfComp_num(coefficient.num, coefficient.den), y.exponent)

        elif type(y) == Inf:
            positivity = True if y.num_representation * self.num_representation > 0 else False
            return Inf(positivity=positivity)

        elif type(y) == ExpressionString:
            return ExpressionString(f"{self} * ( {y} )")


    def __truediv__(self, y):
        # ic(str(y))
        # ic(str(self))
        y = num_to_InfComp(y)

        # INFCOMP
        if type(y) == InfComp_num:

            # n/0 / n
            if y.num != 0:
                # todo potrebbe essere sbagliato
                if self.num_representation > 0 and y.num_representation > 0:  # 1 1
                    epsilon = self.num_as_state(self.epsilon_as_num - y.epsilon_as_num)
                elif self.num_representation < 0 and y.num_representation < 0:  # -1 -1
                    epsilon = self.get_opposite_epsilon(self.num_as_state(self.epsilon_as_num - y.epsilon_as_num))
                elif self.num_representation < 0 and y.num_representation > 0:  # -1 1
                    epsilon = self.num_as_state(self.epsilon_as_num + y.epsilon_as_num)
                    ic(epsilon)
                elif self.num_representation > 0 and y.num_representation < 0:  # 1 -1
                    epsilon = self.num_as_state(y.epsilon_as_num + self.epsilon_as_num)
                else:  # 0
                    epsilon = None

                return InfComp_num(self.num * y.den, self.den * y.num, epsilon).simplify()

            # 0 / 0
            elif self.num == 0 and y.num == 0:
                return Expression(list_repr=[[[InfComp_num(0)]], [[InfComp_num(0)]]], indeterm_type="0_div_0")

            # n / 0
            else:
                rep_result = self.num_representation * y.num_representation
                positivity = self.num_as_state(rep_result)

                return Inf(positivity=positivity)

        # FACTOR
        elif type(y) == Factor:
            div = (self / y.coefficient).simplify()
            return ExpressionString(f"( {InfComp_num(div.num)} / {Factor(div.den, exponent=y.exponent)} )")

        # INF
        elif type(y) == Inf:
            epsilon = True if y.num_representation * self.num > 0 else False
            ic(epsilon  )
            return InfComp_num(0, epsilon=epsilon)

        elif type(y) == ExpressionString:
            return ExpressionString(f"{self} / {y}")


    def __pow__(self, y):
        if self.num_representation < 0:
            if y % 2 == 0:
                self.epsilon = self.get_opposite_epsilon(self.epsilon)


        return InfComp_num(self.num**y, self.den**y, self.epsilon).simplify()

    def root(self, root_num):
        ic(str(self))
        ic(root_num)
        icn = abs(self)
        icn = InfComp_num(num=icn.num**(1/root_num), den=icn.den**(1/root_num))

        # radice pari
        if root_num % 2 == 0:
            # numero pos
            ic(str(icn))
            ic(self.num_representation)
            if self.num_representation > 0:
                return [icn, -icn]
            # numero neg
            else:
                return False

        # radice dispari
        else:
            # numero pos
            if self.num_representation > 0:
                return icn
            # numero neg
            else:
                return -icn


    # COMPARAZIONE

    def __lt__(self, y):  # minore di
        y = num_to_InfComp(y)
        return self.as_decimal() < y.as_decimal()

    def __gt__(self, y):  # maggiore di
        y = num_to_InfComp(y)
        return self.as_decimal() > y.as_decimal()



# FACTOR

class Factor():
    def __init__(self, coefficient=InfComp_num(1), exponent=1.):
        self.exponent = float(exponent)
        self.coefficient = num_to_InfComp(coefficient)  # InfComp_num
        self.coefficient.epsilon = None  # epsilon disattivata per i coefficienti di factors

        self.exponent_digits = [dig for dig in str(exponent)]
        self.num_to_exponent = {"0":"⁰", "1":"¹", "2":"²", "3":"³", "4":"⁴", "5":"⁵", "6":"⁶", "7":"⁷", "8":"⁸", "9":"⁹", ".":"˙"}


    def __str__(self):
        return(f"Factor({self.coefficient},{self.exponent})")

    def fancy_str(self):
        sign = "" if self.coefficient > 0 else "-"
        coeff = self.coefficient.fancy_str() if abs(self.coefficient).as_decimal() != 1.0 else f"{sign}"

        if int(self.exponent) == self.exponent:
            self.exponent = int(self.exponent)
            self.exponent_digits = [dig for dig in str(self.exponent)]

        exponent_digits = ''.join([self.num_to_exponent[dig] for dig in self.exponent_digits]) if self.exponent != 1 else ""
        return f"{coeff}x{exponent_digits}"

    def to_html(self):
        coefficient = self.coefficient.to_html() if self.coefficient.as_decimal() != 1 else ""

        exponent = int(self.exponent) if int(self.exponent) == self.exponent else self.exponent
        exponent = "" if exponent == 1 else exponent

        return f"{coefficient}<msup><mi>x</mi><mn>{exponent}</mn></msup>"


    def to_lambda(self):
        coeff = self.coefficient.to_lambda()
        coeff = f"{coeff} * " if coeff not in ["1", "1.0"] else ""

        expo = f" ** {self.exponent}" if self.exponent != 1 else ""

        return f"{coeff}x{expo}"


    def __abs__(self):
        return Factor(abs(self.coefficient), self.exponent)

    def __neg__(self):
        return Factor(-self.coefficient, self.exponent)


    def __add__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            return ExpressionString(f"{self} + {y}")

        elif type(y) == Factor:
            if y.exponent == self.exponent:
                coefficient = (self.coefficient + y.coefficient).simplify()
                return Factor(coefficient, self.exponent)

            else:
                return ExpressionString(f"{self} + {y}")

    def __sub__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            return ExpressionString(f"{self} - {y}")

        elif type(y) == Factor:
            if y.exponent == self.exponent:
                coefficient = (self.coefficient - y.coefficient).simplify()
                return Factor(coefficient, self.exponent)

            else:
                return ExpressionString(f"{self} - {y}")


    def __mul__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            coefficient = (self.coefficient * y).simplify()
            return Factor(coefficient, self.exponent)

        elif type(y) == Factor:
            coefficient = (self.coefficient * y.coefficient).simplify()
            exponent = self.exponent + y.exponent
            return Factor(coefficient, exponent)

        elif type(y) == ExpressionString:
            return ExpressionString(f"{self} * ( {y} )")


    def __truediv__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            coefficient = (self.coefficient / y).simplify()
            return Factor(coefficient, self.exponent)

        elif type(y) == Factor:
            # x^2 / x^2
            if self.exponent == y.exponent:
                return (self.coefficient / y.coefficient).simplify()

            # x^2 / x^3
            elif self.exponent < y.exponent:
                div = (self.coefficient / y.coefficient).simplify()
                return ExpressionString(f"( {InfComp_num(div.num)} / {Factor(div.den, y.exponent - self.exponent)} )")

            # x^3 / x^2
            elif self.exponent > y.exponent:
                div = (self.coefficient / y.coefficient).simplify()
                return Factor(div, self.exponent - y.exponent)


    def __pow__(self, power):
        coefficient = self.coefficient ** power
        exponent = self.exponent * power
        return Factor(coefficient, exponent)


    def solve(self, value):
        if type(value) == InfComp_num:
            return self.coefficient * (value**self.exponent)

        elif type(value) == Inf:  # inf
            return self.coefficient * (value**self.exponent)

        else:  # int, float, else
            return self.coefficient * (value**self.exponent)


    def root(self, root_num):
        coefficient = self.coefficient.root(root_num)
        exponent = self.exponent / root_num

        if type(coefficient) == list:
            return [Factor(coefficient[0], exponent), Factor(coefficient[1], exponent)]

        elif type(coefficient) == InfComp_num:
            return Factor(coefficient, exponent)

        else:
            return False



def get_max_exponent_factors(factors=None):
    max_exp = max(f.exponent for f in factors)
    return [f for f in factors if f.exponent == max_exp]


# INF

class Inf():
    """
    Inf_div_Inf
    Inf_sub_add_Inf
    Inf_mul_0

    """
    def __init__(self, positivity=None):
        self.positivity = positivity

        pos_to_numRep = {True: 1, None: 0, False: -1}
        self.num_representation = pos_to_numRep[positivity]


    @staticmethod
    def numRep_to_positivity(num_rep):
        positivity = True if num_rep > 0 else ""
        positivity = None if num_rep == 0 else positivity
        positivity = False if num_rep < 0 else positivity
        return positivity

    @staticmethod
    def y_to_num(y):
        yType_to_num = {int: y, float: y, InfComp_num: y.num, Inf: y.num_representation}
        return yType_to_num[type(y)]


    def __str__(self):
        return f"Inf({self.positivity})"

    def fancy_str(self):
        # "" se non è nè positivo nè negativo, "+" pos, "-" neg
        sign = "+" if self.positivity else ""
        sign = "-" if self.positivity == False else sign
        return f"{sign}∞"

    def to_lambda(self):
        return self

    def to_html(self, spaced=False):

        if spaced:
            sign = "<mo> +</mo>" if self.positivity else ""
            sign = "<mo> -</mo>" if self.positivity == False else sign
        else:
            sign = "<mo>+</mo>" if self.positivity else ""
            sign = "<mo>-</mo>" if self.positivity == False else sign

        return f"<mrow>{sign}<mi>∞</mi></mrow>"


    def __neg__(self):
        return Inf(positivity=self.numRep_to_positivity(self.num_representation*-1))


    def __add__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            return self

        elif type(y) == Inf:
            result_num_rep = self.num_representation + y.num_representation
            if result_num_rep == 0:
                return Expression(list_repr=[[[self, y]]], indeterm_type="Inf_sub_Inf")
            else:
                return Inf(self.numRep_to_positivity(result_num_rep))

    def __sub__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            return self

        elif type(y) == Inf:
            result_num_rep = self.num_representation - y.num_representation
            if result_num_rep == 0:
                return Expression(list_repr=[[[self, y]]], indeterm_type="Inf_sub_Inf")
                # return Indeterminate_Form("Inf_sub_Inf", f"{self} - {y}")
            else:
                return Inf(self.numRep_to_positivity(result_num_rep * -1))  # inversione visto che è una sottrazione

    def __mul__(self, y):
        y = num_to_InfComp(y)

        # INFCOMP = 0
        if type(y) == InfComp_num and y.num == 0:
            return Expression(list_repr=[[[y],[0]]], indeterm_type="Inf_mul_0")
        # INCOMP e INF
        else:
            positivity = True if y.num_representation * self.num_representation > 0 else False
            return Inf(positivity)

    def __truediv__(self, y):
        y = num_to_InfComp(y)

        if type(y) == InfComp_num:
            self.positivity = True if self.y_to_num(y) * self.num_representation > 0 else False
            return self

        elif type(y) == Inf:
            return Expression(list_repr=[[[self]],[[y]]], indeterm_type="Inf_div_Inf")

        elif type(y) == Expression:
            resulting_expr = Expression(list_repr=[[[self]]]) / y
            # ic(resulting_expr.list_repr)
            resulting_expr.indeterm_type = "Inf_div_Inf"
            return resulting_expr

    def __pow__(self, y):
        positivity = self.numRep_to_positivity(self.num_representation ** y)
        return Inf(positivity)


# ALTRE CLASSI

class ExpressionString(str):
    """ questa classe ha l'obiettivo di retudrnare una stringa quando vengono fatte delle operazioni tra stringhe,
    ad esepmio: "Factor(2)" / "Factor(2)" = "Factor(2) / Factor(2)". in questo modo non dà un errore e si possono continuare a fare calcoli dalla stringa stessa """
    # todo: questa classe è obsoleta e Expression dovrà ricprire il suo utilizzo

    def __add__(self, y):
        return ExpressionString(f"{self} + {y}")

    def __sub__(self, y):
        return ExpressionString(f"{self} - {y}")

    def __mul__(self, y):
        if y == 1:
            return self
        else:
            return ExpressionString(f"{self} * {y}")

    def __truediv__(self, y):
        # return ExpressionString(f"( {self} ) / ( {y} )")
        return ExpressionString(f"( {self} ) / ( {y} )")

    def __pow__(self, y):
        return ExpressionString(f"{self} ** {y}")


    def to_html(self):
        # ic(self)
        a = Expression(str_exp=self)
        # ic(a.list_repr)
        # ic(a.to_str())
        return a.to_html(nested=True)



# EXPRESSION

str_classes_list = ["Inf(", "InfComp_num(", "Factor("]
classes_list = [Factor, Inf, InfComp_num]

class Expression():
    """ epsressione: (x + y) / (z * (a + b)  ->  list repr: [[[x, y]], [[z], [a, b]]] """

    def __init__(self, str_exp=None, list_repr=None, unsolved_elem=None, indeterm_type=None):
        self.unsolved_elem = unsolved_elem  # es: 'Factor(InfComp_num(1.0,1),4.0).solve(-1)', '+', 'InfComp_num(-30.0, 1)' è unsolved
        self.indeterm_type = indeterm_type

        # creazione con stringa
        if list_repr == None:
            num_denum = str_exp.split(" / ")
            self.list_repr = []
            self.str_exp = str_exp

            for i_nd, num_den in enumerate(num_denum):
                self.list_repr.append([])

                for term_group in num_den.split(" * "):
                    # ic(term_group)

                    self.list_repr[i_nd].append(exp_to_sorted_comps(cleanse_exp(term_group), return_signs=False))
                    # ic(self.list_repr[i_nd])

        else:
            self.list_repr = list_repr
            self.str_exp = self.to_str()

    # da fare (non entrano in uso per ora)
    def __add__(self, y):
        if self.indeterm_type:
            return self

    def __sub__(self, y):
        if self.indeterm_type:
            return self

    def __mul__(self, y):
        ...

    def __truediv__(self, y):

        if len(self.list_repr) == 1 and type(y) == Inf:
            return Expression(list_repr=[self.list_repr[0], [[y]]], indeterm_type="Inf_div_Inf")

        elif type(y) == Expression and len(self.list_repr) == 1 and len(y.list_repr) == 1:
            return Expression(list_repr=[self.list_repr[0], y.list_repr[0]], indeterm_type="Inf_div_Inf")
        else:
            ...

    @staticmethod
    def list_algebraic_sum(terms):
        del_idxs = []
        summed_terms = []

        def sum_and_update(summable_idxs):
            summed_term = eval(" + ".join([str(terms[i]) for i in summable_idxs]))

            # aggiutna degli idx dei termini che dovranno sparire
            [del_idxs.append(i) for i in summable_idxs]
            summed_term_num = summed_term.coefficient.num if type(summed_term) == Factor else summed_term.num

            if summed_term_num != 0:  # se il risultato finale ha un coefficiente non uguale a 0 viene aggiunto ai termini finali
                summed_terms.append(summed_term)

        # FACTORS
        exponents = [i.exponent if type(i) == Factor else None for i in terms]
        summable_exponents = set([i for i in exponents if i != None and exponents.count(i) > 1])

        for summable_expo in summable_exponents:
            summable_idxs = [i for i, term in enumerate(terms) if type(term) == Factor and term.exponent == summable_expo]
            sum_and_update(summable_idxs)

        # ICNS
        types = [str(type(term)) for term in terms]
        summable_idxs = [i for i, type_ in enumerate(types) if type_ == str(InfComp_num)]

        if len(summable_idxs) > 1:
            sum_and_update(summable_idxs)

        # PARTE COMUNE
        return [t for i, t in enumerate(terms) if i not in del_idxs] + summed_terms


    def get_all_terms(self):
        terms = []
        for nd in self.list_repr:
            for tg in nd:
                terms.append([i for i in tg])

        flattened_terms = [i for j in terms for i in j]
        return flattened_terms


    def to_base_form(self):
        # ic(self.list_repr)
        list_repr = copy.deepcopy(self.list_repr)
        # ic(list_repr)

        for i_nd, num_denum in enumerate(self.list_repr):
            new_terms = [i for i in num_denum[0]]
            # ic(new_terms)

            if len(num_denum) > 1:
                for i_tg, term_group in enumerate(num_denum[1:]):

                    next_terms = [i for i in num_denum[i_tg+1]]
                    mul_terms = []

                    for new_t in new_terms:
                        for next_t in next_terms:
                            mul_terms.append(new_t*next_t)
                    # ic(mul_terms)

                    new_terms = self.list_algebraic_sum(mul_terms)
                    # ic(new_terms)
            else:
                new_terms = self.list_algebraic_sum(new_terms)

            list_repr[i_nd] = [new_terms]

        return Expression(list_repr=list_repr)



    def expr_algebraic_sum(self):
        list_repr = copy.deepcopy(self.list_repr)
        # ic(list_repr)

        for i_nd, num_denum in enumerate(self.list_repr):
            for i_tg, term_group in enumerate(num_denum):

                terms = [i for i in term_group]

                final_terms = self.list_algebraic_sum(terms)


                list_repr[i_nd][i_tg] = []
                for t in final_terms:
                    list_repr[i_nd][i_tg].append(t)

        return Expression(list_repr=list_repr)


    def apply_function(self, func):
        list_repr = copy.deepcopy(self.list_repr)

        for i_nd, nd in enumerate(self.list_repr):
            for i_tg, tg in enumerate(nd):
                list_repr[i_nd][i_tg] = [func(t) for t in tg]

        return Expression(list_repr=list_repr)


    def simplify_fraction(self):

        def term2maskNum(term):
            term_num = term.coefficient if type(term) == Factor else term
            return 1 if term_num.as_decimal() > 0 else -1

        simplified_expr = self.expr_algebraic_sum()
        simplify_further = True

        while simplify_further:
            abs_expr = simplified_expr.apply_function(abs)
            ic(abs_expr.to_str())

            abs_str_num = [[str(t) for t in tg] for tg in abs_expr.list_repr[0]]
            abs_str_den = [[str(t) for t in tg] for tg in abs_expr.list_repr[1]]

            possible_tgs = set([tuple(i) for i in abs_str_num]) | set([tuple(i) for i in abs_str_den])

            skip_n_simplification = False
            same_abs_tgs_idxs = []
            for tg in possible_tgs:
                # per i termini come: (x) (x^2) (-2x):
                if len(tg) == 1:
                    single_tg_num = [i for i in abs_str_num if len(i) == 1 if "Factor" in i[0]]
                    single_tg_den = [i for i in abs_str_den if len(i) == 1 if "Factor" in i[0]]

                    if len(single_tg_num) > 0 and len(single_tg_den) > 0:
                        sE_list_repr = copy.deepcopy(simplified_expr.list_repr)

                        sTg_num_idx = abs_str_num.index(single_tg_num[0])
                        sTg_den_idx = abs_str_den.index(single_tg_den[0])

                        # vengono presi i termini singoli, facenti parte di un tg che ha solo un termine (x)(...)
                        num_term = simplified_expr.list_repr[0][sTg_num_idx][0]
                        den_term = simplified_expr.list_repr[1][sTg_den_idx][0]

                        exp_diff = num_term.exponent - den_term.exponent

                        if exp_diff > 0:  # exp num > exp den
                            sE_list_repr[0][sTg_num_idx][0] = num_term.coefficient * Factor(InfComp_num(1), exp_diff)
                            sE_list_repr[1][sTg_den_idx][0] = den_term.coefficient

                        elif exp_diff < 0:  # exp den > exp num
                            sE_list_repr[0][sTg_num_idx][0] = num_term.coefficient
                            sE_list_repr[1][sTg_den_idx][0] = den_term.coefficient * Factor(InfComp_num(1), -exp_diff)

                        else:  # exp den = exp num (diff=0)
                            sE_list_repr[0][sTg_num_idx][0] = num_term.coefficient
                            sE_list_repr[1][sTg_den_idx][0] = den_term.coefficient

                        simplified_expr = Expression(list_repr=sE_list_repr)
                        skip_n_simplification = True
                        break

                # per tutti gli altri termini
                elif list(tg) in abs_str_num and list(tg) in abs_str_den:  # se è al num e al den
                    same_abs_tgs_idxs.append([abs_str_num.index(list(tg)), abs_str_den.index(list(tg))])

            if not skip_n_simplification:
                simplified_num_tg_idx = None  # idx di tg del numeratore che vengono semplificati
                simplified_den_tg_idx = None
                negate_nd = None

                for idx_couple in same_abs_tgs_idxs:
                    num_tg = simplified_expr.list_repr[0][idx_couple[0]]
                    ic([str(i) for i in num_tg])
                    den_tg = simplified_expr.list_repr[1][idx_couple[1]]
                    ic([str(i) for i in den_tg])

                    mask_numTg = [term2maskNum(i) for i in num_tg]
                    ic(mask_numTg)
                    mask_denTg = [term2maskNum(i) for i in den_tg]
                    ic(mask_denTg)
                    mask_neg_denTg = [i * -1 for i in mask_denTg]
                    ic(mask_neg_denTg)

                    count_num_tgs = len(simplified_expr.list_repr[0])
                    count_den_tgs = len(simplified_expr.list_repr[1])

                    if mask_denTg == mask_numTg:
                        ic("\n\nB")
                        simplified_num_tg_idx = idx_couple[0]
                        simplified_den_tg_idx = idx_couple[1]
                        break
                    elif mask_neg_denTg == mask_numTg and sum([count_den_tgs, count_num_tgs]) > 2:  # se è inverso ma solo in caso ha almeno un
                            # altro tg su cui scaricare la negatività
                        ic("\n\nA")
                        simplified_num_tg_idx = idx_couple[0]
                        simplified_den_tg_idx = idx_couple[1]
                        negate_nd = 0 if count_num_tgs > count_den_tgs else 1  # 0 se bisogna negativizzare un tg al num, 1 se al den
                        break

                if [simplified_den_tg_idx, simplified_num_tg_idx] == [None, None]:  # caso in cui non ci sono più tg compatibili
                    break

                # rimozione dei tg semplificabili
                simplified_expr_lR = [[], []]
                simplified_expr_lR[0] = [tg for i_tg, tg in enumerate(simplified_expr.list_repr[0]) if i_tg != simplified_num_tg_idx]
                simplified_expr_lR[1] = [tg for i_tg, tg in enumerate(simplified_expr.list_repr[1]) if i_tg != simplified_den_tg_idx]
                ic(simplified_expr_lR)

                if negate_nd == 0:
                    ic("\n\n\nAA")
                    simplified_expr_lR[0][0] = [-term for term in simplified_expr_lR[0][0]]  # viene preso il primo index e vengono negativizzati tutti i termini
                elif negate_nd == 1:
                    ic("\n\n\nAA")
                    simplified_expr_lR[1][0] = [-term for term in simplified_expr_lR[1][0]]  # viene preso il primo index e vengono negativizzati tutti i termini

                # in caso vengono eliminati tutti i tg al num o al den
                simplified_expr_lR[0] = simplified_expr_lR[0] if simplified_expr_lR[0] else [[InfComp_num(1)]]
                simplified_expr_lR[1] = simplified_expr_lR[1] if simplified_expr_lR[1] else [[InfComp_num(1)]]

                simplified_expr = Expression(list_repr=simplified_expr_lR)
            ic(simplified_expr.to_str())

        return simplified_expr


    def replace_terms(self, new_terms, idxs):
        list_repr = copy.deepcopy(self.list_repr)
        i_nd, i_tg = idxs

        terms_idxs = [i for i, comp in enumerate(self.list_repr[i_nd][i_tg]) if type(comp) in [Inf, InfComp_num, Factor] or any([j in comp for j in str_classes_list])]

        if len(terms_idxs) != len(new_terms):
            raise ValueError("I termini trovati non coincidono coi termini dati come attributi")

        for i, term_idx in enumerate(terms_idxs):
            list_repr[i_nd][i_tg][term_idx] = new_terms[i]

        return Expression(list_repr=list_repr)



    def add_solve(self, x, sel_i_nd=None, sel_i_tg=None):
        """
        sel_i_nd/tg sono gli idx selezionati, devono essere liste
        """
        sel_i_nd = [0,1] if sel_i_nd == None else sel_i_nd
        x = num_to_InfComp(x)

        updated_expr = copy.deepcopy(self)

        for i_nd, nd in enumerate(self.list_repr):
            if i_nd in sel_i_nd:

                for i_tg, tg in enumerate(nd):
                    if sel_i_tg == None or i_tg in sel_i_tg:

                        terms = [i for i in tg]
                        new_terms = []

                        for term in terms:
                            # factor
                            if type(term) == Factor:
                                new_terms.append(f"{term}.solve({x})")

                            # divisione stringa
                            elif type(term) in [ExpressionString, str] and " / " in term:
                                split = term.split(" / ")
                                split_terms = []
                                for i_s, s in enumerate(split):
                                    term = eval(cleanse_exp(s))
                                    if type(term) == Factor:
                                        split_terms.append(f"{term}.solve({x})")
                                    else:
                                        split_terms.append(f"{term}")

                                new_terms.append(" / ".join(split_terms))

                            # infcomp / altro
                            else:
                                new_terms.append(f"{term}")


                        updated_expr = updated_expr.replace_terms(new_terms, [i_nd, i_tg])

        return Expression(list_repr=updated_expr.list_repr, unsolved_elem=x)


    def eval(self, return_expr=True):
        # ic(self.to_str())
        if return_expr:
            evalued_obj = eval(self.to_str())
            if type(evalued_obj) == Expression:
                return evalued_obj
            else:
                return Expression(str_exp=str(evalued_obj))
        else:
            return eval(self.to_str())

    @staticmethod
    def factorize(list_repr=None, mode=None, f_args=None):

        if list_repr:
            list_repr = copy.deepcopy(list_repr)

        # list_repr = list_repr if i_nd == [0,1] else list_repr[i_nd]  # selezioniamo solo la parte a cui siamo interessati

        steps = []
        # 30x^5 + 6x^2 - 30x / 4x^3 - x^2 - x

        if mode in ["x * (.../x)", "x*y * (.../x*y)"]:
            # f_args: mode = "x * (.../x)": [termine_num, termine_den]
            # f_args: mode = "x*y * (.../x*y)": [[termine1_num, termine2_num], [termine1_den, termine2_den]]
            go_on = [False, False]

            # sottoforma di stringa, senza eval
            for i_nd, nd in enumerate(list_repr):
                if len(nd[0]) > 1:  # se c'è solo un termine nel num o den è inutile
                    go_on[i_nd] = True  # in questo modo il prossimo passaggio verrà fatto solo se questo è stato fatto

                    if mode == "x * (.../x)":
                        # x *
                        list_repr[i_nd].insert(0, [f_args[i_nd]])

                        # (.../x)
                        terms = [i for i in list_repr[i_nd][-1]]  # sono i termini che adesso devono essere divisi per x
                        list_repr[i_nd][1] = [f"({t} / {f_args[i_nd]})" for t in terms]

                    else:
                        # x*y *
                        list_repr[i_nd].insert(0, [f_args[i_nd][0]])
                        list_repr[i_nd].insert(1, [f_args[i_nd][1]])

                        # (.../x*y)
                        terms = [i for i in list_repr[i_nd][-1]]
                        list_repr[i_nd][-1] = [f"({t} / {f_args[i_nd][0] * f_args[i_nd][1]})" for t in terms]
                        a = [f"({t} / {f_args[i_nd][0] * f_args[i_nd][1]})" for t in terms]
                        ic(a)


            # ic(list_repr)
            steps.append({"expr": Expression(list_repr=list_repr), "lim": True, "desc": "Scomposizione in base al termine con il grado massimo"})

            eval_list_repr = copy.deepcopy(list_repr)
            for i_nd, nd in enumerate(list_repr):
                if go_on[i_nd]:
                    # viene fatta l'operazione: (.../x*y)  /  (.../x)
                    ic([i for i in list_repr[i_nd][-1]])
                    eval_list_repr[i_nd][-1] = [eval(i) for i in list_repr[i_nd][-1]]
                    ic(eval_list_repr)

            steps.append({"expr": Expression(list_repr=eval_list_repr), "lim": True, "desc": "Calcolo delle divisioni"})

            ic(Expression(list_repr=eval_list_repr).to_str())

            return steps, Expression(list_repr=eval_list_repr)


        elif "..." not in mode:  # in questo caso deve sostituire
            # per riuscire a fare " {term_name} ":
            mode = f" {mode} "  # mode="a + b"  mode=" a + b "
            mode = expand_parenthesis(mode)  # (a + ... -> ( a + ...

            for term_name, term in f_args.items():  # in questo caso f_args è un dizionario

                mode = mode.replace(f" {term_name} ", f" {str(term)} ")
                mode = mode.replace(f" {term_name}.", f" {str(term)}.")

            # ic(mode)

            return Expression(str_exp=mode)



    def to_factorized(self, separation=False):
        copy_expr = copy.deepcopy(self)

        check_idxs = [[i_nd, i_tg] for i_nd, nd in enumerate(copy_expr.list_repr) for i_tg in range(len(nd))]  # all'inizio bisogna controllare tutti gli idxs della list repr
        # ic(check_idxs)

        while check_idxs:  # finchè check_idxs non sarà vuoto
            # ic(copy_expr.list_repr)

            new_check_idxs = []  # vengono svuotati i check idx in modo da avere dentro solo quelli di questo iter

            for i_nd, nd in enumerate(copy_expr.list_repr):
                ic(nd)
                ic(i_nd)
                for i_tg, tg in enumerate(nd):
                    ic(i_tg)
                    if [i_nd, i_tg] in check_idxs:  # viene fatto solo se è nei check idxs

                        terms = [i for i in tg]
                        ic(tg)
                        ic([str(i) for i in terms])
                        # ic(terms)

                        if len(terms) > 1:  # se è 1 salta direttamente alla fine senza aggiungere niente

                            term_nums = [i if type(i) == InfComp_num else i.coefficient for i in terms]

                            # QUALSIASI POLINOMIO
                            # SCOMPOSIZIONE PER GCD(...)

                            num_gcd, fact_gcd = find_gcd(terms, output_type="separated")
                            if fact_gcd != 1 or num_gcd.num != 1:
                                ic("\nGCD")
                                if separation:
                                    ic(separation)
                                    steps, expr = self.factorize(list_repr=[[copy_expr.list_repr[i_nd][i_tg]]], mode="x*y * (.../x*y)", f_args=[[num_gcd, fact_gcd]])

                                    # eliminzione del tg che c'era prima
                                    del copy_expr.list_repr[i_nd][i_tg]

                                    gcd_exponent = fact_gcd.exponent
                                    ic(gcd_exponent)

                                    copy_expr.list_repr[i_nd].insert(i_tg, expr.list_repr[0][0])  # N *
                                    for i in range(int(gcd_exponent)):
                                        copy_expr.list_repr[i_nd].insert(i_tg + i+1, [Factor(coefficient=InfComp_num(1))])  # X * X * ...

                                    copy_expr.list_repr[i_nd].insert(i_tg + i+2, expr.list_repr[0][2])  # (X + N ...)

                                else:
                                    steps, expr = self.factorize(list_repr=[[copy_expr.list_repr[i_nd][i_tg]]], mode="x * (.../x)", f_args=[num_gcd * fact_gcd])

                                    del copy_expr.list_repr[i_nd][i_tg]
                                    copy_expr.list_repr[i_nd].insert(i_tg, expr.list_repr[0][0])
                                    copy_expr.list_repr[i_nd].insert(i_tg+1, expr.list_repr[0][1])
                                # ic(expr.list_repr)
                                # eliminazione del vecchio term group e sostituzione con i 2 nuovi della scomposizione gcd

                                new_check_idxs.append([i_nd, i_tg+1])  # controlliamo solo il secondo tg perchè il primo ha un solo termine

                            # BINOMIO
                            elif len(terms) == 2:
                                # SOMMA PER DIFFERENZA
                                mul = term_nums[0] * term_nums[1]
                                if mul.den == 1:
                                    if mul < 0 and is_square(abs(terms[0])) and is_square(abs(terms[1])) and terms[0].exponent == 2:
                                        ic("\nSOMMA * DIFF")

                                        # a e b diventano positivi in qualsiasi caso (visto che il meno è in str_factorize) e viene applicata la radice essendo quadrati
                                        terms_dict = dict(a = abs(terms[0]).root(2)[0],  # 0: prendiamo il primo elemento della lista, che è la radice positiva,
                                                              # il secondo è la radice negativa
                                                          b = abs(terms[1]).root(2)[0])

                                        del copy_expr.list_repr[i_nd][i_tg]
                                        if term_nums[0] < 0:
                                            expr = self.factorize(mode="(a + b) * (- a + b)", f_args=terms_dict)
                                        else:
                                            expr = self.factorize(mode="(a + b) * (a - b)", f_args=terms_dict)

                                        copy_expr.list_repr[i_nd].insert(i_tg, expr.list_repr[0][1])
                                        copy_expr.list_repr[i_nd].insert(i_tg, expr.list_repr[0][0])

                                        new_check_idxs.append([i_nd, i_tg])
                                        new_check_idxs.append([i_nd, i_tg+1])

                            # TRINOMIO
                            elif len(terms) == 3:
                                ic("\nTRINOMIO")

                                # DA TRINOMIO A QUADRINOMIO (non so come si chiama)

                                ac_mul = term_nums[0] * term_nums[2]

                                if ac_mul.den == 1:
                                    ac_mul = ac_mul.num

                                    # scomposizione in fattori primi della moltiplicazione tra a e c
                                    ac_divs = find_divisors(ac_mul)

                                    # trovare tutte le possibili comvinazionoi di moltiplicazione
                                    possible_combinations = []
                                    for i_sel_div, sel_div in enumerate(ac_divs):
                                        possible_combinations.append([sel_div, mul_list(ac_divs[i] for i in [div_i for div_i in range(len(ac_divs)) if div_i != i_sel_div])])
                                    """ ac_divs: [1, 2, 3, 5] -> possible_combinations: [[1, 30], [2, 15], [3, 10], [5, 6]] """

                                    # aggiunta del - nelle coppie

                                    if term_nums[2] < 0:  # se c è minore di 0 allora uno dei due numeri delle coppie è negativo
                                        pc = []
                                        for couple in possible_combinations:
                                            pc.append([-couple[0], couple[1]])
                                            pc.append([couple[0], -couple[1]])
                                        possible_combinations = pc

                                    elif term_nums[1] < 0:  # se invece solo term_nums["b è minore di 0 vuol dire che sono entrambi numeri negativi
                                        pc = []
                                        for couple in possible_combinations:
                                            pc.append([-couple[0], -couple[1]])
                                        possible_combinations = pc

                                    # cercare la coppia giusta tra tutte le possibili in base alla sua somma
                                    fitting_couple = None
                                    for couple in possible_combinations:
                                        if sum(couple) == term_nums[1].num:
                                            fitting_couple = couple
                                            break

                                    if fitting_couple:
                                        terms_dict = dict(a = terms[0],
                                                          c = terms[2],
                                                          e = Factor(coefficient=InfComp_num(1), exponent=1) * InfComp_num(fitting_couple[1]),
                                                          d = Factor(coefficient=InfComp_num(1), exponent=1) * InfComp_num(fitting_couple[0]))

                                        expr = self.factorize(mode="a + e + d + c", f_args=terms_dict)
                                        copy_expr.list_repr[i_nd][i_tg] = expr.list_repr[0][0]

                                        new_check_idxs.append([i_nd, i_tg])  # subirà al 100% la scomposizione parziale

                            # QUADRINOMIO
                            elif len(terms) == 4:
                                ic("\nRAGGRUPPAMENTO PARZIALE")
                                # RAGGRUPPAMENTO PARZIALE

                                list_idxs = [i for i in range(4)]
                                for i in range(3):  # in un quadrinomio ci sono solo 3 possibilità per scomporre, ognuna con 2 coppie dentro.
                                        # [0,1]: rimane [2,3], [0,2]: rimane [1,3], [0,3]: rimane [1,2]. queste sono tutte le ocppie possibili
                                    first_idx = 0
                                    secon_idx = i+1
                                    gcds = []
                                    couple_1 = [terms[0], terms[i+1]]
                                    ic([str(i) for i in couple_1])
                                    gcds.append(find_gcd(couple_1))
                                    ic(str(gcds[0]))

                                    remaining_list_idxs = [i for i in list_idxs if i not in [first_idx, secon_idx]]  # idxs rimanenti, nel commento sopra soono [2,3], [1,3], [1,2]
                                    couple_2 = [terms[remaining_list_idxs[0]], terms[remaining_list_idxs[1]]]
                                    ic([str(i) for i in couple_2])
                                    gcds.append(find_gcd(couple_2))
                                    ic(str(gcds[1]))

                                    factorized_couples = []
                                    factorized_couples.append([i/gcds[0] for i in couple_1])
                                    factorized_couples.append([i/gcds[1] for i in couple_2])
                                    ic([str(i) for i in factorized_couples[0]])
                                    ic([str(i) for i in factorized_couples[1]])

                                    neg_terms_mask = [[], []]  # es: primo termine e terzo termine positivi, gli altri neg: neg_terms_mask = [[0, 1],[0, 1]]
                                    for i, f_couple_list in enumerate([factorized_couples[0], factorized_couples[1]]):
                                        for f_couple in f_couple_list:
                                            fc_num = f_couple.coefficient if type(f_couple) == Factor else f_couple.num
                                            if fc_num < 0:
                                                neg_terms_mask[i].append(1)
                                            else:
                                                neg_terms_mask[i].append(0)

                                    abs_f_c1 = {str(abs(i)) for i in factorized_couples[0]}
                                    ic(abs_f_c1)
                                    abs_f_c2 = {str(abs(i)) for i in factorized_couples[1]}
                                    ic(abs_f_c2)

                                    def is_valid_neg(neg_terms_masks):
                                        """ False: coppia sbagliata inutilizzabile
                                            True: coppia che va bene così com'è
                                            []: indexes a cui cambiare il segno"""
                                        flattened_neg_terms_mask = [j for i in neg_terms_masks for j in i]
                                        c1, c2 = neg_terms_masks

                                        if sum(flattened_neg_terms_mask) in [3, 1]:
                                            return False
                                        elif c1 == c2 and c1 != [1,1]:
                                            return True
                                        elif c1 == [abs(i-1) for i in c2]:  # abs(i-1): inversione da 0 a 1 e vicev.. è come controllare se cambiando il segno sono uguali
                                            if c1 == [1,1]:
                                                return [0]
                                            elif c2 == [1,1]:
                                                return [1]
                                            else:
                                                return [0]  # in questo caso decidiamo di cambiare segno al primo arbitriariamente, visto che non c'è differenza
                                        elif c1 == [1,1] and c2 == [1,1]:
                                            return [0,1]

                                    if abs_f_c1 & abs_f_c2 == abs_f_c1:  # se hanno gli stessi elementi assoluti, es: 3(x - 1) + 1(-x + 1) ha gli stessi elementi assoluti
                                        neg_terms_mask_validity = is_valid_neg(neg_terms_mask)

                                        if neg_terms_mask_validity == False:
                                            break
                                        elif neg_terms_mask_validity == True:
                                            pass
                                        else:
                                            for i in neg_terms_mask_validity:  # in questo caso sono degli idxs (o uno solo)
                                                factorized_couples[i] = [-term for term in factorized_couples[i]]  # cambio del segno
                                                gcds[i] = -gcds[i]

                                        terms_dict = dict(a = gcds[0],
                                                          b = gcds[1],
                                                          c = factorized_couples[0][0],
                                                          d = factorized_couples[0][1])
                                        ic(terms_dict)
                                        ic([str(i) for i in terms_dict.values()])

                                        del copy_expr.list_repr[i_nd][i_tg]
                                        expr = self.factorize(mode="( a + b ) * ( c + d )", f_args=terms_dict)

                                        copy_expr.list_repr[i_nd].insert(i_tg, expr.list_repr[0][1])
                                        copy_expr.list_repr[i_nd].insert(i_tg, expr.list_repr[0][0])

                                        # controlliamo entrambi perchè potremmo avere una somma * differenza in questi binomi
                                        new_check_idxs.append([i_nd, i_tg])
                                        new_check_idxs.append([i_nd, i_tg+1])

                                        break

                        else:
                            term = terms[0]
                            if separation and type(term) == Factor:
                                del copy_expr.list_repr[i_nd][i_tg]

                                copy_expr.list_repr[i_nd].insert(i_tg, [term.coefficient])  # N *
                                for i in range(int(term.exponent)):
                                    copy_expr.list_repr[i_nd].insert(i_tg + i+1, [Factor(coefficient=InfComp_num(1))])  # X * X * ...




            # una volta finito il den vengono copiati i check idx
            check_idxs = copy.deepcopy(new_check_idxs)

        return copy_expr


    def fully_factorize(self):
        """ si applica alle expression già scomposte, e fa in modo che abbiamo tutte le x al primo grado se possibile """

        numDen_tgs = [[], []]

        for i_nd, nd in enumerate(self.list_repr):  # selezioniamo il denominatore
            for tg in nd:

                # 1 TERMINE  (X)
                if len(tg) == 1:
                    # if type(tg[0]) == Factor:
                    numDen_tgs[i_nd].append(tg)

                # 2 TERMINI, (X + N)
                elif len(tg) == 2:
                    expo = tg[0].exponent
                    coeff = tg[0].coefficient

                    res = (-tg[1]/coeff).root(expo)

                    if res:
                        if type(res) == list:
                            numDen_tgs[i_nd].append(res[0])
                            numDen_tgs[i_nd].append(res[1])
                        else:
                            numDen_tgs[i_nd].append(res)
                    else:
                        numDen_tgs[i_nd].append(tg)

                # 3 TERMINI, (X^2 + X + N)
                elif len(tg) == 3:
                    ic(tg)
                    if tg[0].exponent == 2 and tg[1].exponent == 1 and type(tg[2]) == InfComp_num:
                        a = tg[0].coefficient
                        b = tg[1].coefficient
                        c = tg[2]

                        delta = (b**2 - a*c*4).root(2)
                        ic(delta)

                        if delta:  # solo se il delta non è negativo
                            if type(delta) == list:
                                delta = delta[0]

                            x1 = (-b + delta) / a*2
                            x2 = (-b - delta) / a*2

                            numDen_tgs[i_nd].append(x1)
                            numDen_tgs[i_nd].append(x2)


    def to_str(self, type="str"):
        str_ = ""
        # ic(self.list_repr)

        for i_nd, num_denum in enumerate(self.list_repr):
            str_ += "( "
            for i_tg, term_group in enumerate(num_denum):
                str_ += "( " if len(num_denum) > 1 or len(self.list_repr) > 1 else ""  # se ci sono più polinomi o se è una frazione si mette la parentesi altirmenti no

                terms = [i for i in term_group]
                for i_t, term in enumerate(terms):
                    if type == "str":
                        str_ += str(term)
                    elif type == "fancy":
                        str_ += term.fancy_str()
                    elif type == "lambda":
                        str_ += term.to_lambda()

                    str_ += " + " if i_t != len(terms)-1 else ""

                if len(num_denum) > 1 or len(self.list_repr) > 1:
                    str_ += " ) * " if i_tg != len(num_denum)-1 else " )"

            str_ += " ) / " if i_nd != len(self.list_repr)-1 else " )"

        if type == "lambda":
            split = str_.split(" / ")
            if len(split) > 1:
                # epsilon = 1.0e-18  # per evitare / 0
                epsilon = 0
                str_ = f"{split[0]} / ({split[1]} + {epsilon})"

        return str_


    def to_html(self, nested=False):
        math = '<math xmlns = "http://www.w3.org/1998/Math/MathML">'

        if not nested:
            html_str, html_str_end = (math, '</math>') if len(self.list_repr) == 1 else (f'<div class="expr-div">{math}<mfrac>', '</mfrac></math></div>')
        else:
            html_str, html_str_end = ("", "") if len(self.list_repr) == 1 else ("<mfrac>", "</mfrac>")

        list_repr = copy.deepcopy(self.list_repr)
        # ic(list_repr)
        unsolved_idxs = []

        # facciamo in modo che è come se solve non esistesse (quindi tutti i terms con eval), ma lo sostituiremo dopo alle x
        if self.unsolved_elem:
            for i_nd, num_den in enumerate(self.list_repr):
                for i_tg, tg in enumerate(num_den):
                    for i_t, term in enumerate(tg):
                        if type(term) in [str, ExpressionString]:
                            if ".solve(" in term:
                                term = eval(term.split(".solve(")[0])
                                unsolved_idxs.append([i_nd, i_tg, i_t])
                            else:
                                term = eval(term)

                            list_repr[i_nd][i_tg][i_t] = term
        # ic(list_repr)

        for i_nd, num_denum in enumerate(list_repr):
            html_str += "<mrow>"

            for i_tg, term_group in enumerate(num_denum):
                html_str += "<mo>(</mo>" if len(num_denum) > 1 else ""

                terms = [i for i in term_group]

                for i_t, term in enumerate(terms):
                    # ic(term)

                    if type(term) == Factor:
                        term_num = term.coefficient.as_decimal()
                    elif type(term) == InfComp_num:
                        term_num = term.as_decimal()
                    elif type(term) == Inf:
                        term_num = 0  # Inf ha già il segno giusto incorporato, quindi bisogna agire diversamente
                    elif type(term) in [str, ExpressionString]:
                        term = ExpressionString(term)
                        term_num = 1  # moltiplicato per 1 è uguale a prima

                    # 1 maggiore, 0 INF, -1 minore
                    to_pos = 1 if term_num > 0 else term_num
                    to_pos = -1 if term_num < 0 else to_pos

                    if to_pos != 0:
                        if i_t != 0:  # se non è il primo termine
                            html_str += f"<mo>+</mo>" if to_pos == 1 else "<mo>-</mo>"
                        else:
                            html_str += "" if to_pos == 1 else "<mo>-</mo>"

                        if [i_nd, i_tg, i_t] not in unsolved_idxs:  # se il term non aveva .solve
                            html_str += (term*to_pos).to_html()
                        else:
                            html_str += (term*to_pos).to_html().replace("<mi>x</mi>", f"<mi>({self.unsolved_elem.fancy_str()})</mi>")

                    else:
                        if i_t == 0:
                            html_str += term.to_html()
                        else:
                            html_str += term.to_html(spaced=True)

                if len(num_denum) > 1:
                    # html_str += "<mo>)</mo><mo>*</mo>" if i_tg != len(num_denum)-1 else "<mo>)</mo>"
                    html_str += "<mo>)</mo>"

            html_str += "</mrow>"

        html_str += html_str_end

        if self.indeterm_type:
            html_str += f"{blank_char*1}{math}<mrow><mi>F.I.</mi></mrow></math>"

        return html_str






