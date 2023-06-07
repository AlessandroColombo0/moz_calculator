from django.shortcuts import render
# no
# no

# from .models import Parte, Commento, Utente   #importiamo dal database
# from .forms import Form_Reg_Utente
# from .utils import generate_token

# from django.conf.urls.static import static

# from PIL import Image
import re

from .mez import calcolo, grafici, Mat


from icecream import ic
wrap_width = 195
ic.configureOutput(prefix="> ", includeContext=True)

from django.http import JsonResponse
from django.shortcuts import render
from plotly.offline import plot
import plotly.graph_objs as go

lower_alphabet = [chr(i+97) for i in range(26)]
lower_alphabet_no_x = [i for i in lower_alphabet if i != "x"]
esc_replace_notX_w_x = {re.escape(i): "x" for i in lower_alphabet_no_x}
pattern = re.compile("|".join(esc_replace_notX_w_x.keys()))
# escaped_r_nX_w_x = {re.escape(k): v for k, v in replace_notX_with_x.items()}

# HOME
def home(request):
    context = {}
    return render(request, "home.html", context)


def come_funziona(request):
    context = {}
    return render(request, "come_funziona.html", context)


def calcolo_limiti(request):
    # todo: Expression non contempla "- ( - (x^2 + 1))"
    # todo: tipo di indeterminazione inf/inf e inf-inf non è completo a livello di html
    # todo: non capisce: x + y -z, ma capisce: x + y - z

    context = {"expr_placeholder": "", "lim_placeholder": ""}

    if request.method == "POST":
        ic(request)
        ic(request.POST)
        expr = request.POST.get("expr")
        lim = request.POST.get("lim")

        # inf_sym = "∞"

        result, steps, base_form_expr, x_discont, lim_obj = calcolo.calc_lim(expr, lim)

        if "Expression" in str(type(result)):
            result = result.eval(return_expr=False)
        ic(result)

        base_form_lambda = base_form_expr.to_str("lambda")

        math_tag = '<div class="math-expr"><math xmlns = "http://www.w3.org/1998/Math/MathML">'
        base_form_expr_html = base_form_expr.to_html()
        y_eq = f'{math_tag}<mrow><mi>y</mi><mo>=</mo></mrow></math></div>'
        func_expr = y_eq+base_form_expr_html

        plot = grafici.grafico_funzione(base_form_lambda, lim_obj, x_discont, result)

        context = {"expr": expr, "lim": lim, "steps": steps, "func_expr": func_expr, "plot": plot}

    return render(request, "calcolo_limiti.html", context)


def test(request):
    context = {"a": 2}
    return render(request, "test.html", context)


def ajax_text2math(request):
    text = request.POST["text"]
    text = text.lower()

    if request.POST["type"] == "expr":
        text_list = [i for i in text]

        try:
            if any([i in lower_alphabet_no_x for i in text_list]):  # controllo se ci sono lettere che non sono x nell'expr, serve per non fare replace in caso non ce ne sia
                # bisogno
                text = pattern.sub(lambda m: esc_replace_notX_w_x[re.escape(m.group(0))], text)  # replace multiplo per tutte le lettere

            eval_exp = calcolo.create_evaluable_exp(text)
            expr = Mat.Expression(str_exp=eval_exp)
            text = expr.to_html()
        except Exception as exc:
            ic(exc)
            text = "error"

    elif request.POST["type"] == "lim":
        try:
            if text == "":
                pass
            else:
                lim = calcolo.lim_to_obj(text)
                text = calcolo.lim_to_html(lim)

        except Exception as exc:
            ic(exc)
            text = "error"

    json_data = {"text": text}

    return JsonResponse(json_data)



