import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import pandas as pd

st.set_page_config(page_title="Метод Демукрона", layout="wide", initial_sidebar_state="expanded")
st.title("Метод Демукрона")

# Для мобильной адаптации: уменьшить размеры элементов, сделать горизонтальный скролл для таблиц
st.markdown("""
<style>
/* Меньше отступы для мобильных */
section.main > div {padding-top: 0.5rem;}
/* Горизонтальный скролл для таблиц */
div[data-testid="stTable"] {overflow-x: auto;}
/* Меньше кнопки и инпуты */
button, input, .stButton>button {font-size: 1rem; padding: 0.3rem 0.7rem;}
</style>
""", unsafe_allow_html=True)

def default_labels(n):
    return [chr(ord('A')+i) for i in range(n)]

def get_default_example():
    labels = list("ABCDEFGH")
    n = len(labels)
    adj = np.zeros((n, n), dtype=int)
    names = {c: i for i, c in enumerate(labels)}
    adj[names['A'], names['C']] = 1
    for i, c in enumerate(labels):
        if c != 'B':
            adj[names['B'], i] = 1
    adj[names['C'], names['D']] = 1
    adj[names['C'], names['E']] = 1
    adj[names['D'], names['G']] = 1
    adj[names['E'], names['H']] = 1
    adj[names['F'], names['A']] = 1
    adj[names['F'], names['C']] = 1
    return labels, adj

# --- Сайдбар: параметры графа ---
st.sidebar.header("Параметры графа")
default_labels8 = ','.join(default_labels(8))
num_vertices = st.sidebar.number_input("Количество вершин", min_value=2, max_value=15, value=8)
labels_input = st.sidebar.text_input("Имена вершин (через запятую)", value=default_labels8)

# --- Session state init ---
if 'adj_matrix' not in st.session_state or 'labels' not in st.session_state:
    st.session_state['labels'] = default_labels(num_vertices)
    st.session_state['adj_matrix'] = np.zeros((num_vertices, num_vertices), dtype=int).tolist()

# --- Авто-пример и Очистка ---
col_btns = st.columns(2)
if col_btns[0].button("Авто-пример (с семинара)"):
    labels, adj = get_default_example()
    st.session_state['labels'] = labels
    st.session_state['adj_matrix'] = adj.tolist()
if col_btns[1].button("Очистить матрицу"):
    st.session_state['adj_matrix'] = np.zeros((num_vertices, num_vertices), dtype=int).tolist()

# --- Обновить имена вершин при ручном вводе ---
if labels_input and labels_input != ','.join(st.session_state['labels']):
    st.session_state['labels'] = [x.strip() for x in labels_input.split(',')][:num_vertices]
    if len(st.session_state['adj_matrix']) != num_vertices:
        st.session_state['adj_matrix'] = np.zeros((num_vertices, num_vertices), dtype=int).tolist()

labels = st.session_state['labels']
num_vertices = len(labels)
adj_matrix = st.session_state['adj_matrix']

st.write("### 1. Введите или скорректируйте матрицу смежности")
st.write("_Отметьте ячейки, если есть связь из строки в столбец_:")

cols = st.columns([1]+[1]*num_vertices)
cols[0].markdown("**→/↓**")
for j in range(num_vertices):
    cols[j+1].markdown(f"**{labels[j]}**")

for i in range(num_vertices):
    row = st.columns([1]+[1]*num_vertices)
    row[0].markdown(f"**{labels[i]}**")
    for j in range(num_vertices):
        if i == j:
            row[j+1].write("-")
            st.session_state['adj_matrix'][i][j] = 0
        else:
            st.session_state['adj_matrix'][i][j] = row[j+1].checkbox(
                " ", value=bool(st.session_state['adj_matrix'][i][j]), key=f"cell_{i}_{j}"
            )
adj_matrix_np = np.array(st.session_state['adj_matrix'])

run_algo = st.button("Запустить алгоритм Демукрона")

def demukron_lambda_table(adj_matrix, labels):
    matrix = adj_matrix.copy()
    n = matrix.shape[0]
    removed = np.zeros(n, dtype=bool)
    lambda_rows = []
    cross_rows = []
    order = []
    matrix_work = matrix.copy()
    while True:
        incoming = matrix_work.sum(axis=0)
        row = [incoming[i] if not removed[i] else '×' for i in range(n)]
        lambda_rows.append(row)
        zero_nodes = [i for i in range(n) if incoming[i] == 0 and not removed[i]]
        cross_rows.append(zero_nodes)
        if not zero_nodes:
            break
        for idx in zero_nodes:
            removed[idx] = True
            order.append(idx)
            for j in range(n):
                if matrix_work[idx, j] > 0:
                    matrix_work[idx, j] -= 1
    return lambda_rows, cross_rows, order

def demukron_levels(adj_matrix):
    matrix = adj_matrix.copy()
    n = matrix.shape[0]
    levels = []
    used = set()
    while len(used) < n:
        incoming = matrix.sum(axis=0)
        level = [i for i in range(n) if incoming[i] == 0 and i not in used]
        if not level:
            return None
        levels.append(level)
        used.update(level)
        for idx in level:
            matrix[idx, :] = 0
    return levels

if run_algo:
    levels = demukron_levels(adj_matrix_np)
    st.write("### 2. Множества уровней (Ni):")
    if levels is None:
        st.error("Граф содержит цикл! Топологическая сортировка невозможна.")
    else:
        for i, level in enumerate(levels):
            st.write(f"N{i}: {{ " + ', '.join(labels[j] for j in level) + " }}")
        st.write("### 3. Таблица λ (количество входящих связей по шагам)")
        lambda_rows, cross_rows, order = demukron_lambda_table(adj_matrix_np, labels)
        table_data = []
        for step, row in enumerate(lambda_rows):
            row_marked = []
            for i, val in enumerate(row):
                if val == '×':
                    row_marked.append('×')
                elif i in cross_rows[step]:
                    row_marked.append(f"**{val}**")
                else:
                    row_marked.append(str(val))
            table_data.append(row_marked)
        st.markdown("#### Шаги λi (строки — шаги, столбцы — вершины)")
        st.table(pd.DataFrame(table_data, columns=labels))
        st.write("### 4. Визуализация графа по уровням")
        G = nx.DiGraph()
        for i in range(len(labels)):
            G.add_node(labels[i])
        for i in range(len(labels)):
            for j in range(len(labels)):
                if adj_matrix_np[i, j]:
                    G.add_edge(labels[i], labels[j])
        pos = {}
        y_gap = 2
        x_gap = 2
        for lvl, nodes in enumerate(levels):
            for k, idx in enumerate(nodes):
                pos[labels[idx]] = (lvl * x_gap, -k * y_gap)
        fig, ax = plt.subplots(figsize=(min(10, 2+2*len(levels)), min(6, 2+len(labels)//2)))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color="#d0e6fa", arrows=True, ax=ax, font_size=12)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=20, width=2)
        plt.axis('off')
        st.pyplot(fig)
        st.success("Готово!")
else:
    st.info("Введите матрицу и нажмите 'Запустить алгоритм Демукрона' или воспользуйтесь авто-примером.")