import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from config import Config

def draw_graph_ascii(adj, labels=None, node_names=None, features=None):
    Config.print_subsection("VISUALIZACIÓN DEL GRAFO")
    
    num_nodes = adj.shape[0]
    
    if node_names is None:
        node_names = [f"N{i}" for i in range(num_nodes)]
    
    print("  CONEXIONES DEL GRAFO:")
    print(f"  NODOS:")
    for i in range(num_nodes):
        label_info = f" (clase: {labels[i]})" if labels is not None else ""
        feature_info = ""
        if features is not None:
            feature_preview = features[i][:3] if len(features[i]) > 3 else features[i]
            feature_str = [f"{x:.2f}" for x in feature_preview]
            suffix = "..." if len(features[i]) > 3 else ""
            feature_info = f" (features: [{', '.join(feature_str)}{suffix}])"
        print(f"    {node_names[i]}{label_info}{feature_info}")
    
    print(f"\n  ARISTAS:")
    edges_found = False
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # Solo mostrar una vez cada arista
            if adj[i, j] > 0:
                weight = f" (peso: {adj[i,j]:.2f})" if adj[i,j] != 1.0 else ""
                print(f"    {node_names[i]} ←→ {node_names[j]}{weight}")
                edges_found = True
    
    if not edges_found:
        print("    (No hay aristas en el grafo)")
    
    self_loops = []
    for i in range(num_nodes):
        if adj[i, i] > 0:
            self_loops.append(node_names[i])
    
    if self_loops:
        print(f"\n  AUTO-BUCLES: {', '.join(self_loops)}")
    
    if features is not None:
        print(f"\n  VECTORES DE CARACTERÍSTICAS COMPLETOS:")
        for i in range(num_nodes):
            feature_str = [f"{x:6.3f}" for x in features[i]]
            print(f"    {node_names[i]}: [{', '.join(feature_str)}]")
    
    print(f"\n   MATRIZ DE ADYACENCIA VISUAL:")
    print("     ", end="")
    for j in range(num_nodes):
        print(f"{node_names[j]:>4}", end="")
    print()
    
    for i in range(num_nodes):
        print(f"  {node_names[i]:>2} ", end="")
        for j in range(num_nodes):
            symbol = "●" if adj[i,j] > 0 else "○"
            print(f"{symbol:>4}", end="")
        print()

def draw_graph_matplotlib(adj, labels=None, node_names=None, features=None, show_features=True, legend_fontsize=18):
    Config.print_subsection("VISUALIZACIÓN GRÁFICA DEL GRAFO")
    
    G = nx.Graph()
    num_nodes = adj.shape[0]
    
    if node_names is None:
        node_names = [f"N{i}" for i in range(num_nodes)]
    
    for i in range(num_nodes):
        G.add_node(i, name=node_names[i], label=labels[i] if labels is not None else None)
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  
            if adj[i, j] > 0:
                G.add_edge(i, j, weight=adj[i, j])
    
    fig_width = 16 if show_features and features is not None else 10
    fig_height = 12 if show_features and features is not None else 8
    
    plt.figure(figsize=(fig_width, fig_height))
    
    if num_nodes <= 10:
        pos = nx.spring_layout(G, seed=42, k=3, iterations=50)  # Más espacio entre nodos
    else:
        pos = nx.spring_layout(G, seed=42, k=2)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        node_colors = [colors[np.where(unique_labels == labels[i])[0][0]] for i in range(num_nodes)]
    else:
        node_colors = 'lightblue'
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=1200,
                          alpha=0.8)
    
    nx.draw_networkx_edges(G, pos, 
                          alpha=0.6,
                          width=2)
    
    node_labels = {i: f"{node_names[i]}" for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, node_labels, font_size=15, font_weight='bold')
    
    if show_features and features is not None:
        ax = plt.gca()
        for i in range(num_nodes):
            x, y = pos[i]
            
            feature_str = [f"{val:.2f}" for val in features[i]]
            feature_text = f"[{', '.join(feature_str)}]"
            
            class_info = f"C{labels[i]}" if labels is not None else ""
            if class_info:
                full_text = f"{class_info}\n{feature_text}"
            else:
                full_text = feature_text
            
            offset_x = 0.15
            offset_y = -0.08
            
            ax.text(x + offset_x, y + offset_y, full_text, 
                   fontsize=18, 
                   fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor="white", 
                            edgecolor="gray",
                            alpha=0.9),
                   verticalalignment='center',
                   horizontalalignment='left')
    else:
        if labels is not None:
            for i in range(num_nodes):
                x, y = pos[i]
                ax = plt.gca()
                ax.text(x, y - 0.12, f"C{labels[i]}", 
                       fontsize=10, 
                       fontweight='bold',
                       horizontalalignment='center',
                       verticalalignment='center')
    
    edge_labels = {}
    for i, j in G.edges():
        weight = adj[i, j]
        if weight != 1.0:  # Solo mostrar si no es peso estándar
            edge_labels[(i, j)] = f"{weight:.2f}"
    
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=14)
        
    if labels is not None:
        legend_elements = []
        for i, label in enumerate(unique_labels):
            legend_elements.append(plt.scatter([], [], c=[colors[i]], s=400, label=f'Clase {label}'))
        plt.legend(handles=legend_elements, loc='upper left', fontsize=legend_fontsize)
    
    plt.axis('off')
    
    plt.margins(0.2)
    plt.tight_layout()    
    plt.show()
    
    return G

def draw_attention_graph(adj, attention_matrix, labels=None, node_names=None, features=None, title="Attention Weights", legend_fontsize=18):
    Config.print_subsection("VISUALIZACIÓN DE PESOS DE ATENCIÓN")
    
    G = nx.DiGraph() 
    num_nodes = adj.shape[0]
    
    if node_names is None:
        node_names = [f"N{i}" for i in range(num_nodes)]
    
    for i in range(num_nodes):
        G.add_node(i, name=node_names[i])
    
    max_attention = np.max(attention_matrix)
    min_attention = np.min(attention_matrix[attention_matrix > 0])
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j] > 0 and attention_matrix[i, j] > 0:
                norm_weight = (attention_matrix[i, j] - min_attention) / (max_attention - min_attention)
                G.add_edge(i, j, weight=attention_matrix[i, j], norm_weight=norm_weight)
    
    fig_width = 16 if features is not None else 12
    plt.figure(figsize=(fig_width, 10))
    
    pos = nx.spring_layout(G, seed=42, k=3, iterations=50)  # Más espacio para características
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        node_colors = [colors[np.where(unique_labels == labels[i])[0][0]] for i in range(num_nodes)]
    else:
        node_colors = 'lightblue'
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=1200,
                          alpha=0.8)
    
    edges = G.edges()
    weights = [G[u][v]['norm_weight'] * 5 + 0.5 for u, v in edges]  # Escalar grosor
    
    nx.draw_networkx_edges(G, pos, 
                          width=weights,
                          alpha=0.7,
                          edge_color='red',
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->')
    
    node_labels = {i: f"{node_names[i]}" for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, node_labels, font_size=12, font_weight='bold')
    
    edge_labels = {}
    for u, v in edges:
        weight = G[u][v]['weight']
        if weight > 0.1:  # Solo mostrar pesos significativos
            edge_labels[(u, v)] = f"{weight:.3f}"
    
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    if features is not None:
        ax = plt.gca()
        for i in range(num_nodes):
            x, y = pos[i]
            
            # Formatear el vector de características
            feature_str = [f"{val:.2f}" for val in features[i]]
            feature_text = f"[{', '.join(feature_str)}]"
            
            # Información adicional del nodo
            class_info = f"C{labels[i]}" if labels is not None else ""
            if class_info:
                full_text = f"{class_info}\n{feature_text}"
            else:
                full_text = feature_text
            
            # Posicionar el texto al lado del nodo
            offset_x = 0.15
            offset_y = -0.08
            
            ax.text(x + offset_x, y + offset_y, full_text, 
                   fontsize=8, 
                   fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor="lightyellow", 
                            edgecolor="orange",
                            alpha=0.9),
                   verticalalignment='center',
                   horizontalalignment='left')
    
    if labels is not None:
        unique_labels = np.unique(labels)
        legend_elements = []
        for i, label in enumerate(unique_labels):
            legend_elements.append(plt.scatter([], [], c=[colors[i]], s=400, label=f'Clase {label}'))
        plt.legend(handles=legend_elements, loc='upper left', fontsize=legend_fontsize)
    
    plt.title(f"{title}\n(Grosor de arista = peso de atención)", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.margins(0.2)  # Más margen para ver las características
    plt.tight_layout()
    plt.show()
    
    print(f"  TOP 5 PESOS DE ATENCIÓN:")
    flat_attention = attention_matrix.flatten()
    top_indices = np.argsort(flat_attention)[-5:][::-1]
    
    for idx in top_indices:
        i, j = np.unravel_index(idx, attention_matrix.shape)
        if flat_attention[idx] > 0:
            print(f"    {node_names[i]} → {node_names[j]}: {flat_attention[idx]:.4f}")
    
    if features is not None:
        print(f"\n  CARACTERÍSTICAS DE NODOS MÁS IMPORTANTES:")
        node_importance = np.sum(attention_matrix, axis=1)  # Suma de atención recibida
        top_nodes = np.argsort(node_importance)[-3:][::-1]  # Top 3 nodos
        
        for node_idx in top_nodes:
            if node_importance[node_idx] > 0:
                feature_str = [f"{x:6.3f}" for x in features[node_idx]]
                print(f"    {node_names[node_idx]} (importancia: {node_importance[node_idx]:.3f}): [{', '.join(feature_str)}]")