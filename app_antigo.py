from lib import *

st.title('Sintonização de controladores PID.')

control_type = st.selectbox('Selecionar tipo de controlador:', ['P', 'PI', 'PD', 'PID'],
                            key='control_type')

tab1, tab2, tab3 = st.expander('**Definir parâmetros de desempenho**').tabs([
    r'$M_P$ [%] e $\tau$ [s]',
    r'$\omega_n$ e $\xi$',
    r'$s=\sigma \pm j\omega_d$'
], on_change='rerun', key='input_mode')


pole = st.session_state['pole'] if 'pole' in st.session_state else None

with tab1.form('Sobressinal máximo e tempo de acomodação'):
    M_p = st.number_input('Sobressinal máximo [%]:', min_value=0.0, key='M_p', required=True)
    tol = st.radio('Tolerância [%]:', [2, 5], key='tol')
    tau = st.number_input('Tempo de acomodação [s]:', min_value=0.0, key='tau', required=True)

    # texto:
    st.latex(r'\xi = \sqrt\frac{\ln(Mp/100)^2}{\pi^2 + \ln(Mp/100)^2}')
    if tol == 2:
        st.latex(r'\omega_n = \frac{4}{\tau \xi}')
    elif tol == 5:
        st.latex(r'\omega_n = \frac{3}{\tau \xi}')

    st.latex(r'\omega_d = \omega_n \sqrt{1 - \xi^2},\; \sigma = \xi \omega_n')

    st.latex(r'p_i = \sigma \pm j \omega_d')

    if st.form_submit_button('Confirmar'):
        pole = pole_from_Mp(M_p, tol, tau)
        st.session_state['pole'] = pole

with tab2.form('Frequência natural e amortecimento'):
    omega_n = st.number_input(r'Frequência natural $\omega_n$:', key='omega_n', required=True)
    xi = st.number_input(r'Amortecimento $\xi$:', key='xi', required=True)

    # texto:
    st.latex(r'\omega_d = \omega_n \sqrt{1 - \xi^2},\; \sigma = \xi \omega_n')

    st.latex(r'p_i = \sigma \pm j \omega_d')

    if st.form_submit_button('Confirmar'):
        pole = pole_from_freq(omega_n, xi)
        st.session_state['pole'] = pole

with tab3.form('Polo complexo'):
    sigma = st.number_input(r'Parte real $\sigma$:', key='sigma', required=True)
    omega_d = st.number_input(r'Parte imaginária $\omega_d$:', key='omega_d', required=True)

    # texto:
    st.latex(r'p_i = \sigma \pm j \omega_d')

    if st.form_submit_button('Confirmar'):
        pole = pole_from_complex(sigma, omega_d)
        st.session_state['pole'] = pole

if pole is None:
    st.error('**Por favor defina parâmetros de desempenho!**')
    st.stop()
assert isinstance(pole, complex)

st.markdown(fr'Polo desejado = ${pole.real:.3f} \pm j{pole.imag:.3f}$')


if 'TF' not in st.session_state:
    st.session_state['TF'] = {
        'g_num_str': None,
        'g_den_str': None,
        'h_num_str': None,
        'h_den_str': None,
    }

with st.expander('**Definir planta**').form('Listas de coeficientes do maior para o menor grau'):
    col_g, col_h = st.columns(2)
    with col_g:
        st.subheader('$G(s)=$')
        _g_num_str = st.text_input('Coeficientes do numerador de  $G(s)$:', '1', required=True, key='g_num')
        _g_den_str = st.text_input('Coeficientes do denominador de $G(s)$:', '1, 8, 32, 0', required=True, key='g_den')
    with col_h:
        st.subheader('$H(s)=$')
        _h_num_str = st.text_input('Coeficientes do numerador de $H(s)$:', '1', required=True, key='h_num')
        _h_den_str = st.text_input('Coeficientes do denominador $H(s)$:', '1, 4', required=True, key='h_den')

    if st.form_submit_button('Confirmar'):
        st.session_state['TF']['g_num_str'] = _g_num_str
        st.session_state['TF']['g_den_str'] = _g_den_str
        st.session_state['TF']['h_num_str'] = _h_num_str
        st.session_state['TF']['h_den_str'] = _h_den_str

g_num_str = st.session_state['TF']['g_num_str']
g_den_str = st.session_state['TF']['g_den_str']
h_num_str = st.session_state['TF']['h_num_str']
h_den_str = st.session_state['TF']['h_den_str']

if not g_den_str:
    st.error('**Por favor defina funções de transferência da planta!**')
    st.stop()

st.markdown('Funções de transferência:')
col_g, col_h = st.columns(2)
with col_g:
    st.latex(fr'G(s)={sp.latex(
        sp.Poly(g_num_str.split(','), gens=s) / sp.Poly(g_den_str.split(','), gens=s) 
    )}')
with col_h:
    st.latex(fr'H(s)={sp.latex(
        sp.Poly(h_num_str.split(','), gens=s) / sp.Poly(h_den_str.split(','), gens=s) 
    )}')

with st.expander('**Propriedades da função de transferência da planta**'):
    st.subheader('Função de transferência:')
    ft = get_ft(g_num_str, g_den_str, h_num_str, h_den_str)
    zeroes, poles = pole_zero_numerical_data(ft)
    st.latex(f'G_{{mf}}(s) = {sp.latex(ft.simplify())}')

    st.subheader('Zeros:')
    txt = ''
    for i, z in enumerate(zeroes):
        txt += f'- $z_{i}$ = ${z:.3f}$\n'
    if txt:
        st.markdown(txt)
    else:
        st.markdown('Não há zeros.')

    st.subheader('Polos:')
    txt = ''
    for i, p in enumerate(poles):
        txt += f'- $p_{i}$ = ${p:.3f}$\n'
    if txt:
        st.markdown(txt)
    else:
        st.markdown('Não há polos.')

    num = ft.num
    den = ft.den
    ganho = sp.Poly(num, s).LC() / sp.Poly(den, s).LC()
    st.subheader(f'Ganho: ${ganho:.3f}$')

if control_type == 'PI' or control_type == 'PID':
    poles.append(complex(0))

z_c, Kc = calcular_zc(ft, pole, poles, zeroes, ganho, control_type)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader('Zero do controlador:')
    st.latex(f'{z_c:.3f}')
with col2:
    st.subheader('Ganho do controlador:')
    st.latex(f'K_c = {Kc:.3f}')

st.header(f'Projeto do controlador {control_type}')
if control_type == 'PD':
    Kd = Kc
    Kp = z_c * Kc

    Gc = sp.together(Kp + Kd * s)

    st.markdown(f'''
    - $K_p = {Kp:.5e}$
    - $K_d = {Kd:.5e}$
    ''')
    st.markdown('Função de Transferência do Controlador PD:')
    st.latex(sp.latex(Gc))

elif control_type == 'PI':
    Kp = Kc
    Ki = z_c * Kc

    Gc = sp.together(Kp + Ki / s)

    st.markdown(f'''
    - $K_p = {Kp:.5e}$
    - $K_i = {Ki:.5e}$
    ''')
    st.markdown('Função de Transferência do Controlador PI:')
    st.latex(sp.latex(Gc))

elif control_type == 'PID':
    Kp = 2*Kc*z_c
    Ki = Kc * z_c**2
    Kd = Kc

    Gc = sp.together(Kp + Ki/s + Kd*s)

    st.markdown(f'''
    - $K_p = {Kp:.5e}$
    - $K_i = {Ki:.5e}$
    - $K_d = {Kd:.5e}$
    ''')
    st.markdown('Função de Transferência do Controlador PI:')
    st.latex(sp.latex(Gc))