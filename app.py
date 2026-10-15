"""
app.py --- Interface Streamlit do projeto de controladores pelo lugar das raízes.

Fluxo da aplicação
------------------
1. Escolha do tipo de controlador (P, PI, PD, PID ou Genérico).
2. Definição do polo desejado por três caminhos alternativos:
   M_p/tau, w_n/xi ou coordenada complexa.
3. Definição da planta G(s) e H(s) por coeficientes.
4. Propriedades da planta em malha aberta (zeros, polos, ganho) e tabelas do LGR.
5. Síntese do controlador (condições de ângulo e módulo).
6. Análise de malha fechada (polos resultantes).
7. Discretização do sistema (cap. 5 do SisContr).

Toda a matemática vive em `lib.py`; aqui apenas exibimos os resultados.
"""

import matplotlib.pyplot as plt
import sympy as sp
import streamlit as st

import lib
from lib import (
    DESCRICAO_METODOS,
    coeficientes_ft,
    calcular_controlador,
    calcular_zc,
    calcular_controlador_gen,
    discretizar_sistema,
    ganho_planta,
    get_ft,
    grandezas_polo,
    pole_from_complex,
    pole_from_freq,
    pole_from_Mp,
    polos_malha_fechada,
    s,
    tabela_lgr,
    z,
)
from sympy.physics.control import pole_zero_numerical_data

st.set_page_config(page_title="Sintonização de controladores", layout="wide")

TIPOS = ["P", "PI", "PD", "PID", "Genérico"]


# --- helpers ---

st.html(
    """
<style>
/* O Streamlit forca .katex-display{display:inline;margin:0} e isso impede
   qualquer centralizacao, entao a regra precisa de !important. Vale so para a
   FT (container com key="ft"), para nao mexer nas outras equacoes. */
.st-key-ft .katex-display {
    display: block !important;
    text-align: center !important;
    margin: .4rem 0 !important;
    white-space: normal !important;
}
.st-key-ft .katex { font-size: 1.45em !important; }

[data-testid="stMetricValue"] { font-size: 1.15rem !important; }
[data-testid="stMetricLabel"] { font-size: .78rem !important; }
</style>
"""
)

CASAS = 4   # casas decimais exibidas

# substituição s = ... de cada método (SisContr.pdf, cap. 5, sec. 5.2)
FORMULA_DISCRETIZACAO = {
    "euler": r"s \;\approx\; \frac{z - 1}{T_s}",
    "backward": r"s \;\approx\; \frac{z - 1}{z\,T_s}",
    "trapezoidal": r"s \;\approx\; \frac{2}{T_s}\,\frac{z - 1}{z + 1}",
}


def num(v, casas=CASAS):
    """Número sem zeros à direita: 0.50000 -> 0.5, 3.0000 -> 3."""
    q = round(float(v), casas)
    if q == 0:
        return "0"
    if q == int(q) and abs(q) < 1e15:
        return str(int(q))
    return f"{q:.{casas}f}".rstrip("0").rstrip(".")


def latex_ft(expr, var=s):
    """LaTeX de N(s)/D(s) com o líder do denominador puxado para fora.

    1/(10000*s**2 - 11772) -> 1 / {10000 (s^2 - 1.1772)}
    """
    def _arredonda(e):
        subs = {}
        for a in e.atoms(sp.Float, sp.Rational):
            v = round(float(a), CASAS)
            if v == 0 and float(a) != 0:        # 1e-5 nao pode virar 0
                subs[a] = sp.Float(float(f"{float(a):.3g}"))
            else:
                subs[a] = sp.Integer(v) if v == int(v) else sp.Float(v)
        return e.xreplace(subs)

    n, d = sp.fraction(sp.cancel(sp.simplify(expr)))
    if not n.free_symbols and not d.free_symbols:
        return sp.latex(_arredonda(n / d))
    if sp.simplify(d) == 1:                      # polinômio puro, sem "/1"
        return sp.latex(_arredonda(sp.expand(n)))
    lc = sp.LC(sp.Poly(sp.expand(d), var))
    interno = _arredonda(sp.expand(sp.expand(d) / lc))
    # em SymPy Float(1.0) == 1 e falso, entao compara por valor
    lc1 = not lc.free_symbols and float(lc) == 1
    int1 = not interno.free_symbols and float(interno) == 1
    if int1:
        den_latex = sp.latex(_arredonda(lc))
    elif lc1:
        den_latex = sp.latex(interno)
    else:
        den_latex = sp.latex(_arredonda(lc)) + r"\left(" + sp.latex(interno) + r"\right)"
    return r"\frac{" + sp.latex(_arredonda(n)) + "}{" + den_latex + "}"


def ft_para_latex(expr, var=s):
    """ft_para_latex com retorno seguro se o SymPy recusar a expressão."""
    try:
        return latex_ft(expr, var)
    except (sp.PolynomialError, TypeError, ValueError, ZeroDivisionError):
        return sp.latex(sp.simplify(expr))





def coeficientes_texto(coef, casas=CASAS):
    """Coeficientes (numerador, denominador) como texto."""
    return tuple(" ".join(num(v, casas) for v in coef[k])
                 for k in ("numerador", "denominador"))


def complexo_para_latex(v, casas=CASAS):
    """Complexo como a ± jb; raiz real sai sem parte imaginária."""
    v = complex(v)
    if abs(v.imag) < 10 ** (-casas):
        return num(v.real, casas)
    return f"{num(v.real, casas)} \\pm j\\,{num(abs(v.imag), casas)}"


def zero_txt(posicoes, casas=CASAS):
    """Zeros como o PDF escreve: `z = 8.66` (o parametro de `Gc = Kc(s+z)`).

    O zero fica em `s = -z`; mostramos os dois para nao confundir.
    Zeros iguais aparecem uma vez, com a multiplicidade.
    """
    vistos = {}
    for p in posicoes:
        vistos[round(float(p), casas)] = vistos.get(round(float(p), casas), 0) + 1
    partes = []
    for p, mult in vistos.items():
        txt = f"z = {num(p, casas)} \\ (s = {num(-p, casas)})"
        partes.append(f"{txt} \\times {mult}" if mult > 1 else txt)
    return ", ".join(partes)


def lista_para_latex(valores, casas=CASAS):
    """Lista de zeros/polos; par conjugado aparece uma vez só."""
    vistos = set()
    partes = []
    for v in valores:
        v = complex(v)
        chave = (round(v.real, casas), round(abs(v.imag), casas))
        if abs(v.imag) > 10 ** (-casas):
            if chave in vistos:
                continue
            vistos.add(chave)
        partes.append(complexo_para_latex(v, casas))
    return ", ".join(partes)


def tentar(acao, contexto=""):
    """(True, resultado) ou (False, mensagem de erro)."""
    try:
        return True, acao()
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        return False, f"{contexto}{exc}" if contexto else str(exc)


# --- 1. Controlador ---

st.title("Sintonia de controladores pelo método do lugar das raízes")

st.caption(
    "Método do LGR (SisContr.pdf, cap. 2 e 4) com discretização opcional "
    "(SisContr.pdf, cap. 5)."
)

control_type = st.selectbox("Tipo de controlador", TIPOS, key="control_type")

# nomes curto para as fórmulas padrão
MAPA_TIPO = {"P": "P", "PI": "PI", "PD": "PD", "PID": "PID", "Genérico": "GEN"}


# --- 2. Polo desejado ---

with st.expander("1. Definir parâmetros de desempenho", expanded=True):
    tab_mp, tab_wn, tab_cx = st.tabs(
        [r"$M_P$ [%] e $\tau$ [s]", r"$\omega_n$ e $\xi$", r"$s = \sigma \pm j\omega_d$"]
    )

    pole = st.session_state.get("pole")

    with tab_mp:
        with st.form("form_mp"):
            c1, c2 = st.columns(2)
            M_p = c1.number_input("Sobressinal máximo [%]:", min_value=0.0, max_value=100.0,
                                  value=10.0, step=0.5, format="%.2f")
            tol = c2.radio("Tolerância [%]:", [2, 5], horizontal=True)
            tau = st.number_input("Tempo de acomodação [s]:", min_value=0.0,
                                  value=1.0, step=0.1, format="%.3f")

            st.latex(
                r"\xi=\sqrt{\frac{\ln^2(M_p/100)}{\pi^2+\ln^2(M_p/100)}}"
                r"\qquad\omega_n=\frac{4}{\tau\xi}\ (2\%)\ \text{ou}\ "
                r"\omega_n=\frac{3}{\tau\xi}\ (5\%)"
            )
            if st.form_submit_button("Confirmar"):
                ok, res = tentar(lambda: pole_from_Mp(M_p, tol, tau), "Polo inválido: ")
                if ok:
                    st.session_state["pole"] = res
                    st.success(f"Polo desejado: $s_i = {complexo_para_latex(res)}$")
                else:
                    st.error(res)

    with tab_wn:
        with st.form("form_wn"):
            c1, c2 = st.columns(2)
            omega_n = c1.number_input(r"Frequência natural $\omega_n$ [rad/s]:",
                                      min_value=0.0, value=4.0, step=0.1, format="%.3f")
            xi = c2.number_input(r"Amortecimento $\xi$:", min_value=0.0, max_value=0.999,
                                 value=0.5, step=0.01, format="%.3f")
            st.latex(r"\omega_d=\omega_n\sqrt{1-\xi^2},\quad \sigma=\xi\omega_n")
            if st.form_submit_button("Confirmar"):
                ok, res = tentar(lambda: pole_from_freq(omega_n, xi), "Polo inválido: ")
                if ok:
                    st.session_state["pole"] = res
                    st.success(f"Polo desejado: $s_i = {complexo_para_latex(res)}$")
                else:
                    st.error(res)

    with tab_cx:
        with st.form("form_cx"):
            c1, c2 = st.columns(2)
            real = c1.number_input("Parte real $\\sigma$ (posicione $\\sigma < 0$):",
                                   value=-2.0, step=0.1, format="%.3f")
            imag = c2.number_input("Parte imaginária $\\omega_d$:",
                                   value=3.0, step=0.1, format="%.3f")
            st.latex(r"s_i=\sigma\pm j\omega_d,\qquad \sigma=-\operatorname{Re}(s_i)")
            if st.form_submit_button("Confirmar"):
                ok, res = tentar(lambda: pole_from_complex(real, imag), "Polo inválido: ")
                if ok:
                    st.session_state["pole"] = res
                    st.success(f"Polo desejado: $s_i = {complexo_para_latex(res)}$")
                else:
                    st.error(res)

pole = st.session_state.get("pole")

if pole is None:
    st.error("**Defina os parâmetros de desempenho para continuar.**")
    st.stop()

sigma, w_d = grandezas_polo(pole)
st.markdown(
    f"**Polo desejado:** $s_i = {complexo_para_latex(pole)}$ &nbsp;&nbsp; "
    f"$\\sigma = {num(sigma)}$, $\\omega_d = {num(w_d)}$"
)


# --- 3. Planta ---

with st.expander("2. Definir a planta", expanded=True):
    if "TF" not in st.session_state:
        st.session_state["TF"] = {"g_num": "1", "g_den": "1, 8, 32, 0",
                                 "h_num": "1", "h_den": "1, 4"}

    with st.form("form_planta"):
        col_g, col_h = st.columns(2)
        with col_g:
            st.subheader("$G(s)$")
            g_num = st.text_input("Coeficientes do numerador de $G(s)$ (maior → menor grau):",
                                  key="tf_g_num", value=st.session_state["TF"]["g_num"])
            g_den = st.text_input("Coeficientes do denominador de $G(s)$ (maior → menor grau):",
                                  key="tf_g_den", value=st.session_state["TF"]["g_den"])
        with col_h:
            st.subheader("$H(s)$")
            h_num = st.text_input("Coeficientes do numerador de $H(s)$ (maior → menor grau):",
                                  key="tf_h_num", value=st.session_state["TF"]["h_num"])
            h_den = st.text_input("Coeficientes do denominador de $H(s)$ (maior → menor grau):",
                                  key="tf_h_den", value=st.session_state["TF"]["h_den"])

        st.form_submit_button("Confirmar planta")

    st.session_state["TF"] = {"g_num": g_num, "g_den": g_den,
                              "h_num": h_num, "h_den": h_den}

    if not all(str(v).strip() for v in (g_num, g_den, h_num, h_den)):
        st.error("Preencha todos os coeficientes da planta.")
        st.stop()

    ok, (ft, G, H) = tentar(lambda: get_ft(g_num, g_den, h_num, h_den), "Planta inválida: ")
    if not ok:
        st.error(ft)
        st.stop()

st.markdown('**Funções de transferência da planta:**')
col1, col2 = st.columns([1, 1])
with col1:
    st.latex(f'G(s)={ft_para_latex(G.to_expr())}')
with col2:
    st.latex(f'H(s)={ft_para_latex(H.to_expr())}')

# --- 4. Propriedades de G(s)H(s) ---

zeros, polos = pole_zero_numerical_data(ft)
ganho = ganho_planta(ft)
tbl = tabela_lgr(pole, polos, zeros)

col1, col2 = st.columns([1, 1])

with col1:
    with st.expander("3. Propriedades de $G(s)H(s)$", expanded=True):
        st.latex(f"G(s)H(s)={ft_para_latex(ft.doit().to_expr())}")
        st.markdown("**Zeros**")
        st.markdown(f"${lista_para_latex(zeros)}$" if zeros else "Não há zeros.")
        st.divider()
        st.markdown("**Polos**")
        st.markdown(f"${lista_para_latex(polos)}$" if polos else "Não há polos.")
        st.divider()
        st.markdown(f"**Ganho** (razão dos coeficientes líderes): $K = {num(ganho)}$")

        # plano-s
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.axhline(0, color="grey", lw=0.7)
        ax.axvline(0, color="grey", lw=0.7)
        if zeros:
            ax.plot([complex(z).real for z in zeros], [complex(z).imag for z in zeros],
                    "o", color="tab:green", label="zeros")
        if polos:
            ax.plot([complex(p).real for p in polos], [complex(p).imag for p in polos],
                    "x", color="tab:red", ms=9, mew=2, label="polos")
        ax.plot(pole.real, pole.imag, "*", color="black", ms=16, label=r"$s_i$")
        ax.set_xlabel("Re(s)")
        ax.set_ylabel("Im(s)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with col2:
    with st.expander("4. Passos 11 e 12 do LGR", expanded=True):
        st.subheader("**Passo 11 — distâncias e produtos**")
        st.markdown(r"(condição de módulo: $1+K\prod_j(s_i-z_j)/\prod_k(s_i-p_k)=0$).")
        # zeros: índice j, polos: índice k
        for rotulo, simbolo, indice, linhas in (("zeros", "z", "j", tbl["zeros"]),
                                                ("polos", "p", "k", tbl["polos"])):
            if not linhas:
                st.markdown(f"**Não há {rotulo}.**")
                continue
            st.markdown(f"**{rotulo.capitalize()}**")
            for i, linha in enumerate(linhas):
                pos = complex(linha["posicao"])
                sinal = "-" if pos.imag < 0 else "+"
                local = (f"{num(pos.real)}" if abs(pos.imag) < 1e-12
                         else f"{num(pos.real)} {sinal} j\\,{num(abs(pos.imag))}")
                st.markdown(
                    f"- ${simbolo}_{{{i}}}$ = ${local}$: "
                    f"$|s_i - {simbolo}_{indice}| = {num(linha['modulo'])}$, "
                    f"$\\angle(s_i - {simbolo}_{indice}) = {num(linha['fase'])}^\\circ$"
                )
            st.divider()

        st.latex(rf"\prod_j |s_i - z_j| = {num(tbl['produto_zeros'])}")
        st.latex(rf"\prod_k |s_i - p_k| = {num(tbl['produto_polos'])}")

        st.subheader("Passo 12 — soma das fases")
        st.latex(r"\sum_j \angle(s_i - z_j) - \sum_k \angle(s_i - p_k)")
        st.latex(f"= {num(tbl['angulo_ponto'])}^\\circ")
        st.markdown("**Déficit de ângulo do controlador:**")
        st.latex(
            rf"\varphi = 180^\circ - \left({num(tbl['angulo_ponto'])}^\circ\right)"
            rf" = {num(tbl['phi'])}^\circ"
        )


# --- 5. Síntese do controlador ---

with st.expander("5. Síntese do controlador", expanded=True):
    tipo = MAPA_TIPO[control_type]

    opcoes_generico = {}
    if control_type == "PID":
        opcoes_generico["relacao_zeros"] = st.number_input(
            "Razão entre os dois zeros ($z_2 = r\\,z_1$); $r=1$ mantém os zeros coincidentes:",
            min_value=0.0, value=1.0, step=0.1, format="%.3f",
        )
    if control_type == "Genérico":
        gz = st.columns(3)
        n_zeros = gz[0].number_input("Zeros informados (z > 0):", min_value=0, value=1, step=1)
        n_polos = gz[1].number_input("Polos a calcular:", min_value=1, value=1, step=1)
        n_fixos = gz[2].number_input("Polos fixos:", min_value=0, value=0, step=1)

        c1, c2, c3 = st.columns(3)
        zeros_ctrl = [
            c1.number_input(f"Zero {i + 1} (em $s=-z$):", min_value=0.0,
                            value=1.0 + i, step=0.5, format="%.3f", key=f"gz{i}")
            for i in range(int(n_zeros))
        ]
        relacoes_polos = [
            c2.number_input(f"Razão polo {i + 1}/polo 1:", min_value=0.0,
                            value=1.0, step=0.1, format="%.3f", key=f"gr{i}")
            for i in range(int(n_polos))
        ]
        polos_fixos = [
            c3.number_input(f"Polo fixo {i + 1} (0 = integrador):", value=0.0,
                            step=0.5, format="%.3f", key=f"gp{i}")
            for i in range(int(n_fixos))
        ]

    st.container(border=True).markdown(f"**Controlador: {control_type}**")

    # --- cálculo ---
    def _calcular():
        """Reproduz as células PI/PD/PID/GEN do notebook `pid.ipynb`."""
        avisos = []

        # integrador do PI/PID
        polos_int = list(polos)
        if tipo in ("PI", "PID"):
            polos_int = polos_int + [0j]

        if tipo == "P":
            # P: sem zero, só a condição de módulo
            return calcular_controlador(
                ft, pole, polos, zeros, sigma, w_d,
                n_zeros=0, relacoes=[], polos_denominador=(),
                avisos=avisos,
            ), avisos

        if tipo == "GEN":
            return calcular_controlador_gen(
                ft, pole, polos, zeros, sigma, w_d,
                zeros_controlador=zeros_ctrl,
                n_polos=int(n_polos),
                relacoes_polos=relacoes_polos,
                polos_fixos=polos_fixos,
                avisos=avisos,
            ), avisos

        if tipo == "PID":
            # PID: dois zeros com razão configurável
            r = float(opcoes_generico["relacao_zeros"])
            return calcular_controlador(
                ft, pole, polos_int, zeros, sigma, w_d,
                n_zeros=2, relacoes=[1.0, r],
                polos_denominador=[0.0],
                avisos=avisos,
            ), avisos

        # PI e PD via calcular_zc
        pid, pd_, pi_ = tipo == "PID", tipo == "PD", tipo == "PI"
        zc, Kc = calcular_zc(ft, pole, polos_int, zeros, ganho, w_d, sigma,
                             pid, pd_, pi_)
        return calcular_controlador(
            ft, pole, polos_int, zeros, sigma, w_d,
            n_zeros=1, relacoes=[1.0],
            polos_denominador=[0.0] if (pid or pi_) else [],
            avisos=avisos,
        ), avisos

    ok, saida = tentar(_calcular, "Falha no projeto: ")
    if not ok:
        st.error(saida)
        st.stop()
    resultado, avisos = saida

    Gc = resultado["Gc"]
    Kc = resultado["ganho"]
    phi_ctrl = resultado["phi"]

for aviso in avisos:
    if aviso.startswith("Aviso:"):
        st.warning(aviso)

# --- linha de resultados ---
Kp = Ki = Kd = None
if tipo != "GEN":
    Kp, Ki, Kd = lib.parametros_padrao(Gc, tipo)
elif sp.simplify(sp.fraction(sp.together(lib._expressao_s(Gc)))[1]) in (1, s):
    num_gc, den_gc = sp.fraction(sp.together(lib._expressao_s(Gc)))
    pol = sp.Poly(sp.expand(num_gc), s)
    c0 = float(pol.coeff_monomial(1))
    c1 = float(pol.coeff_monomial(s))
    c2 = float(pol.coeff_monomial(s ** 2))
    if sp.simplify(den_gc) == s:
        Kp, Ki, Kd = c1, c0, c2
    else:
        Kp, Ki, Kd = c0, 0.0, c1

itens = [("Ganho $K_c$", num(Kc)), (r"Ângulo $\varphi$", f"{num(phi_ctrl)}°")]
itens += [(n, num(v)) for n, v in (("$K_p$", Kp), ("$K_i$", Ki), ("$K_d$", Kd))
          if v is not None]
st.markdown('**Parametros do controlador:**')
st.table(itens, border='horizontal')

# --- singularidades ---
zeros_ctrl = zero_txt(resultado["zeros"])
if tipo == "GEN":
    calc = ", ".join(f"$s = {num(-p)}$" for p in resultado["polos_calculados"])
    fixos = resultado["polos"][len(resultado["polos_calculados"]):]
    polos_ctrl = f"calculados: {calc}"
    if fixos:
        polos_ctrl += " | fixos: " + ", ".join(f"$s = {num(p)}$" for p in fixos)
elif resultado["polos"]:
    polos_ctrl = ", ".join(f"$s = {num(p)}$" for p in resultado["polos"])
else:
    polos_ctrl = "nenhum"

st.markdown(f"**Zeros:** ${zeros_ctrl or 'nenhum'}$")
st.markdown(f"**Polos:** {polos_ctrl}")

if tipo == "P":
    st.warning(
        "O controlador **P** não possui zero nem polo: ele **não altera a fase**. "
        "Apenas a condição de módulo é satisfeita — confirme nos passos 11/12 se "
        "$s_i$ já pertence ao LGR da planta pura."
    )

# --- função de transferência ---
cabecalho = st.columns([6, 1])
cabecalho[0].markdown("**Função de transferência**")
with cabecalho[1]:
    with st.popover("Coeficientes"):
        coef = resultado["coeficientes"]
        cn, cd = coeficientes_texto(coef)
        st.markdown(f"Numerador: ${cn}$")
        st.markdown(f"Denominador: ${cd}$")

with st.container(key="ft"):
    st.markdown(f"$$G_c(s) = {ft_para_latex(Gc)}$$")


# --- 6. Malha fechada ---

with st.expander("6. Polos de malha fechada ($1 + C(s)G(s)H(s) = 0$)"):
    try:
        fechados = polos_malha_fechada(ft, Gc)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Polos de malha fechada**")
            for i, r in enumerate(fechados):
                v = r["valor"]
                sinal = "-" if v.imag < 0 else "+"
                local = (f"{num(v.real)}" if abs(v.imag) < 1e-12
                         else f"{num(v.real)} {sinal} j\\,{num(abs(v.imag))}")
                st.markdown(f"- $p_{{{i}}} = {local}$")
        with c2:
            estavel = all(r["valor"].real < 0 for r in fechados)
            if estavel:
                st.success("Sistema **estável** (todos os polos no semiplano esquerdo).")
            else:
                st.error("Sistema **instável**: há polo no semiplano direito.")
            st.markdown(
                f"Polo desejado presente: "
                f"{'sim' if any(abs(r['valor'] - pole) < 1e-6 for r in fechados) else 'não'}"
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Não foi possível calcular os polos de malha fechada: {exc}")


# --- 7. Discretização ---

with st.expander("7. Discretização (cap. 5 — SisContr.pdf)"):
    st.caption(
        "SisContr.pdf, cap. 5, sec. 5.2 (PDF p. 55-57): Euler progressivo, "
        "Euler regressivo e trapezoidal (Tustin)."
    )

    c1, c2 = st.columns(2)
    metodo = c1.selectbox(
        "Método",
        list(DESCRICAO_METODOS),
        format_func=lambda m: DESCRICAO_METODOS[m],
    )
    Ts = c2.number_input("Período de amostragem $T_s$ [s]:",
                         min_value=0.0, value=0.1, step=0.01, format="%.4f")

    st.markdown("Substituição aplicada:")
    st.latex(FORMULA_DISCRETIZACAO[metodo] + rf"\qquad\text{{com }}\; T_s = {Ts}")

    if Ts <= 0:
        st.error("O período de amostragem deve ser positivo.")
    else:
        ok, disc = tentar(
            lambda: discretizar_sistema(ft, Gc, Ts, metodo), "Falha na discretização: "
        )
        if not ok:
            st.error(disc)
        else:
            cols = st.columns(3)
            for coluna, titulo, expr in zip(
                cols,
                (r"Planta $G_d(z)$", r"Controlador $C_d(z)$",
                 r"Malha aberta $T_d(z)$"),
                (disc["planta_discreta"], disc["controlador_discreto"],
                 disc["sistema_aberto_discreto"]),
            ):
                with coluna:
                    st.markdown(f"**{titulo}**")
                    st.latex(ft_para_latex(expr, var=z))

            with st.expander("Coeficientes em $z$"):
                for nome, expr in (
                    ("G_d(z)", disc["planta_discreta"]),
                    ("C_d(z)", disc["controlador_discreto"]),
                    ("T_d(z)", disc["sistema_aberto_discreto"]),
                ):
                    c = coeficientes_ft(expr, var=z)
                    cn, cd = coeficientes_texto(c)
                    st.markdown(f"**{nome}**")
                    st.markdown(f"- Numerador: ${cn}$")
                    st.markdown(f"- Denominador: ${cd}$")


st.divider()
st.caption(
    "Teoria: *SisContr.pdf* (cap. 2, 4 e 5) e Ogata, *Engenharia de Controle "
    "Moderno* (cap. 5, 6 e 8)."
)