import streamlit as st
from sympy.physics.control import *
import sympy as sp

import numpy as np

s, k = sp.symbols('s k')

def pole_from_Mp(M_p, tol, tau):
    """Calcula polo desejado a partir de sobressinal máximo, tolerância e tempo de acomodação"""
    xi = np.sqrt(np.log(M_p / 100) ** 2 / (np.pi ** 2 + np.log(M_p / 100) ** 2))
    if tol == 2:
        omega_n = 4 / (tau * xi)
    elif tol == 5:
        omega_n = 3 / (tau * xi)
    else:
        raise ValueError(f'Tolerância deve ser 2 ou 5, não {tol}')

    w_d = omega_n * np.sqrt(1 - xi ** 2)
    sigma = xi * omega_n

    return -sigma + w_d * 1j

def pole_from_freq(w_n, xi):
    """Calcula polo desejado a partir de freq. natural e amortecimento"""
    w_d = w_n * np.sqrt(1 - xi ** 2)
    sigma = xi * w_n
    return -sigma + w_d * 1j

def pole_from_complex(real, imag):
    return complex(real, imag)

def get_ft(g_num_str, g_den_str, h_num_str, h_den_str):
    numerador = sp.Poly.from_list(g_num_str.split(','), gens=s).as_expr()
    denominador = sp.Poly.from_list(g_den_str.split(','), gens=s).as_expr()

    G = TransferFunction(numerador, denominador, s)

    numerador = sp.Poly.from_list(h_num_str.split(','), gens=s).as_expr()
    denominador = sp.Poly.from_list(h_den_str.split(','), gens=s).as_expr()
    H = TransferFunction(numerador, denominador, s)
    # return Series(G, H).doit()  # fixme: era para ser em feedback
    return Feedback(G, H, sign=-1).doit()


def calcular_zc(ft, ponto_si, polos, zeros, ganho, controller):
    w_d = np.real(ponto_si)
    sigma = np.imag(ponto_si)

    diferenca_polos = np.empty((0, 2))
    diferenca_zeros = np.empty((0, 2))

    txt_z_dist = ''
    txt_p_dist = ''
    txt_z_fase = ''
    txt_p_fase = ''
    for i, zero in enumerate(zeros):
        diferenca = ponto_si - zero
        modulo = abs(diferenca)
        fase = np.degrees(np.angle(diferenca))
        nova_linha = [[round(modulo, 4), fase]]
        diferenca_zeros = np.append(diferenca_zeros, nova_linha, axis=0)
        txt_z_dist += f'- $z_{i}$ = (${zero:.3f}$): ${nova_linha[0][0]}°$\n'
        txt_z_fase += f'- $z_{i}$ = (${zero:.3f}$): ${nova_linha[0][1]}°$\n'

    for i, polo in enumerate(polos):
        diferenca = ponto_si - polo
        modulo = abs(diferenca)
        fase = np.degrees(np.angle(diferenca))
        nova_linha = [[round(modulo, 4), fase]]
        diferenca_polos = np.append(diferenca_polos, nova_linha, axis=0)
        txt_p_dist += f'- $p_{i}$ = (${polo:.3f}$): ${nova_linha[0][0]}°$\n'
        txt_p_fase += f'- $p_{i}$ = (${polo:.3f}$): ${nova_linha[0][1]}°$\n'


    produto_polos = 1
    produto_zeros = 1
    angulo_ponto = 0

    for linha in diferenca_polos:
        produto_polos = produto_polos * linha[0]
        angulo_ponto = angulo_ponto - linha[1]

    for linha in diferenca_zeros:
        produto_zeros = produto_zeros * linha[0]
        angulo_ponto = angulo_ponto + linha[1]

    with st.expander('**Passo 11:**'):
        st.subheader('Distância entre o ponto e os zeros:')
        if zeros:
            st.markdown(txt_z_dist)
            st.markdown('#### Produto das distâncias:')
            st.latex(rf'\prod_j |z_j - p_i| = {produto_zeros:.3f}')
        else:
            st.markdown('Não há zeros.')

        st.subheader('Distância entre o ponto e os polos:')
        if polos:
            st.markdown(txt_p_dist)
            st.markdown('#### Produto das distâncias:')
            st.latex(rf'\prod_j |p_j - p_i| = {produto_polos:.3f}')
        else:
            st.markdown('Não há polos.')

    with st.expander('**Passo 12:**'):
        st.subheader('Fase entre o ponto e os zeros:')
        if zeros:
            st.markdown(txt_z_dist)
        else:
            st.markdown('Não há zeros.')

        st.subheader('Fase entre o ponto e os polos:')
        if polos:
            st.markdown(txt_p_dist)
        else:
            st.markdown('Não há polos.')

        st.subheader('Soma das fases:')
        st.latex(rf'\sum_j \angle(z_j - p_i) - \sum_j \angle(p_j - p_i)=')
        st.latex(f'{angulo_ponto:.3f}°')

        phi = 180 - angulo_ponto

        while phi < 0:
            phi += 360

        phi = phi % 360

        if controller == 'PID':
            phi = phi / 2

        st.subheader('Ângulo do ponto:')
        st.latex(rf'\phi = {phi:.3f}°')

    if phi > 90:
        z_c = abs(sigma) - (w_d / np.tan(np.radians(180 - phi)))
    else:
        z_c = abs(sigma) + (w_d / np.tan(np.radians(phi)))

    # print(f"Zero do controlador: {z_c:.2f} + 0*j")

    if controller == 'PID':
        Gc = TransferFunction((s + z_c) ** 2, s, s)
    elif controller == 'PD':
        Gc = TransferFunction(s + z_c, 1, s)
    elif controller == 'PI':
        Gc = TransferFunction(s + z_c, s, s)
    else:
        raise ValueError('Tipo de controlador não suportado')

    ft_c = Series(ft, Gc).doit()

    freq = ft_c.eval_frequency(ponto_si)
    M = abs(complex(freq))
    Kc = 1 / M
    # print(f"Kc = {Kc}\n")

    return z_c, Kc