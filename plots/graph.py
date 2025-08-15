import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from config import Config

def draw_graph_ascii(adj, labels=None, node_names=None, features=None, directed=True):
    Config.print_subsection("VISUALIZACIÓN DEL GRAFO")
    
    num_nodes = adj.shape[0]
    
    if node_names is None:
        node_names = [f"N{i}" for i in range(num_nodes)]
    
    print("  CONEXIONES DEL GRAFO:")
    print(f"  TIPO: {'Dirigido' if directed else 'No dirigido'}")
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
    
    if directed:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and adj[i, j] > 0:
                    weight = f" (peso: {adj[i,j]:.2f})" if adj[i,j] != 1.0 else ""
                    reciprocal = " ↔" if adj[j, i] > 0 else ""
                    print(f"    {node_names[i]} → {node_names[j]}{weight}{reciprocal}")
                    edges_found = True
    else:
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj[i, j] > 0:
                    weight = f" (peso: {adj[i,j]:.2f})" if adj[i,j] != 1.0 else ""
                    print(f"    {node_names[i]} ↔ {node_names[j]}{weight}")
                    edges_found = True
    
    if not edges_found:
        print("    (No hay aristas en el grafo)")
    
    self_loops = []
    for i in range(num_nodes):
        if adj[i, i] > 0:
            self_loops.append(node_names[i])
    
    if self_loops:
        print(f"\n  AUTO-BUCLES: {', '.join(self_loops)}")
    
    if directed:
        print(f"\n  ESTADÍSTICAS DE CONECTIVIDAD:")
        in_degrees = np.sum(adj, axis=0)
        out_degrees = np.sum(adj, axis=1)
        
        for i in range(num_nodes):
            print(f"    {node_names[i]}: in-degree={int(in_degrees[i])}, out-degree={int(out_degrees[i])}")
        
        # Contar aristas recíprocas
        reciprocal_edges = np.sum((adj > 0) & (adj.T > 0)) // 2
        total_edges = np.sum(adj > 0)
        print(f"    Aristas recíprocas: {reciprocal_edges}/{total_edges}")
    
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
            if directed:
                if adj[i,j] > 0 and adj[j,i] > 0 and i != j:
                    symbol = "⟷"  # Recíproco
                elif adj[i,j] > 0:
                    symbol = "→"  # Solo i→j
                elif i == j and adj[i,j] > 0:
                    symbol = "⟲"  # Auto-bucle
                else:
                    symbol = "○"  # Sin conexión
            else:
                symbol = "●" if adj[i,j] > 0 else "○"
            print(f"{symbol:>4}", end="")
        print()

def draw_graph_matplotlib(adj, labels=None, node_names=None, features=None, show_features=True, directed=True, legend_fontsize=18):
    Config.print_subsection("VISUALIZACIÓN GRÁFICA DEL GRAFO")
    
    if directed:
        G = nx.DiGraph()
        print(f"  Creando grafo dirigido con {adj.shape[0]} nodos...")
    else:
        G = nx.Graph()
        print(f"  Creando grafo no dirigido con {adj.shape[0]} nodos...")
    
    num_nodes = adj.shape[0]
    
    if node_names is None:
        node_names = [f"N{i}" for i in range(num_nodes)]
    
    for i in range(num_nodes):
        G.add_node(i, name=node_names[i], label=labels[i] if labels is not None else None)
    
    total_edges = 0
    if directed:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i, j] > 0:
                    G.add_edge(i, j, weight=adj[i, j])
                    total_edges += 1
    else:
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):  
                if adj[i, j] > 0:
                    G.add_edge(i, j, weight=adj[i, j])
                    total_edges += 1
    
    print(f"  Agregadas {total_edges} aristas")
    
    fig_width = 20 if show_features and features is not None else 14
    fig_height = 14 if show_features and features is not None else 10
    
    plt.figure(figsize=(fig_width, fig_height))
    
    if num_nodes <= 10:
        pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
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
                          node_size=1500,  # Nodos más grandes para grafos dirigidos
                          alpha=0.8)
    
    if directed:
        unidirectional_edges = []
        bidirectional_edges = []
        
        for i, j in G.edges():
            if G.has_edge(j, i):  # Es recíproca
                if i < j: 
                    bidirectional_edges.append((i, j))
            else:  # Es unidireccional
                unidirectional_edges.append((i, j))
        
        if unidirectional_edges:
            nx.draw_networkx_edges(G, pos, 
                                  edgelist=unidirectional_edges,
                                  alpha=0.7,
                                  width=2,
                                  edge_color='black',
                                  arrows=True,
                                  arrowsize=40,
                                  arrowstyle='->')
        
        if bidirectional_edges:
            nx.draw_networkx_edges(G, pos,
                                  edgelist=bidirectional_edges,
                                  alpha=0.7,
                                  width=2,
                                  edge_color='blue',
                                  arrows=True,
                                  arrowsize=25,
                                  arrowstyle='<->',
                                  connectionstyle="arc3,rad=0.1")
    else:
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
            
            # Increased font size from 10 to 14 for features
            ax.text(x + offset_x, y + offset_y, full_text, 
                   fontsize=14,  # Increased from 10
                   fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.4",  # Increased padding
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
                       fontsize=16, 
                       fontweight='bold',
                       horizontalalignment='center',
                       verticalalignment='center')
    
    edge_labels = {}
    for i, j in G.edges():
        weight = adj[i, j]
        if weight != 1.0:  # Solo mostrar si no es peso estándar
            edge_labels[(i, j)] = f"{weight:.2f}"
    
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    # UNIFIED LEGEND: Combine class and edge type legends
    legend_elements_combined = []
    
    # Add class legend elements if labels exist
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            legend_elements_combined.append(
                plt.scatter([], [], c=[colors[i]], s=400, 
                           label=f'Clase {label}', alpha=0.8)
            )
    
    # Add edge type legend elements for directed graphs
    if directed:
        from matplotlib.lines import Line2D
        # Add separator line if we have both types
        if labels is not None:
            legend_elements_combined.append(
                plt.Line2D([0], [0], color='white', lw=0, label='')  # Empty separator
            )
        
        legend_elements_combined.extend([
            Line2D([0], [0], color='black', lw=2, label='Arista Unidireccional', 
                   marker='>', markersize=8, markerfacecolor='black'),
            Line2D([0], [0], color='blue', lw=2, label='Arista Bidireccional',
                   marker='s', markersize=6, markerfacecolor='blue')
        ])
    
    # Create single unified legend if we have elements to show
    if legend_elements_combined:
        title = ""
        if labels is not None and directed:
            title = "Clases y Tipos de Arista"
        elif labels is not None:
            title = "Clases de Nodos"
        elif directed:
            title = "Tipos de Arista"
        
        unified_legend = plt.legend(handles=legend_elements_combined, 
                                   loc='upper left', 
                                   fontsize=legend_fontsize-2,  # Slightly smaller for combined
                                   title=title,
                                   title_fontsize=legend_fontsize,
                                   frameon=True,
                                   fancybox=True,
                                   shadow=True,
                                   framealpha=0.95,
                                   bbox_to_anchor=(0.02, 0.98))  # Better positioning
        
    plt.axis('off')
    plt.margins(0.2)
    plt.tight_layout()    
    plt.show()
    
    return G

def draw_attention_graph(adj, attention_matrix, labels=None, node_names=None, 
                        features=None, title="Attention Weights", directed=True, 
                        legend_fontsize=18):
    Config.print_subsection("VISUALIZACIÓN DE PESOS DE ATENCIÓN")
    
    G = nx.DiGraph() 
    num_nodes = adj.shape[0]
    
    if node_names is None:
        node_names = [f"N{i}" for i in range(num_nodes)]
    
    for i in range(num_nodes):
        G.add_node(i, name=node_names[i])
    
    max_attention = np.max(attention_matrix[attention_matrix > 0])
    min_attention = np.min(attention_matrix[attention_matrix > 0])
    
    attention_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j] > 0 and attention_matrix[i, j] > 0:
                norm_weight = (attention_matrix[i, j] - min_attention) / (max_attention - min_attention + 1e-8)
                G.add_edge(i, j, weight=attention_matrix[i, j], norm_weight=norm_weight)
                attention_edges.append((i, j, attention_matrix[i, j]))
    
    print(f"  Visualizando {len(attention_edges)} conexiones con atención")
    
    fig_width = 20 if features is not None else 16
    plt.figure(figsize=(fig_width, 14))
    
    pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        node_colors = [colors[np.where(unique_labels == labels[i])[0][0]] for i in range(num_nodes)]
    else:
        node_colors = 'lightblue'
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=1500,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=1.5)
    
    edges = G.edges()
    if edges:
        weights = [G[u][v]['norm_weight'] * 6 + 0.5 for u, v in edges]
        
        attention_values = [G[u][v]['weight'] for u, v in edges]
        edge_colors = plt.cm.Reds(np.array(attention_values) / max_attention)
        
        nx.draw_networkx_edges(G, pos, 
                              width=weights,
                              alpha=0.8,
                              edge_color=edge_colors,
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              connectionstyle="arc3,rad=0.05")
    
    node_labels = {i: f"{node_names[i]}" for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, node_labels, font_size=14, font_weight='bold')
    
    edge_labels = {}
    threshold = max_attention * 0.3  # Solo mostrar pesos > 30% del máximo
    for u, v in edges:
        weight = G[u][v]['weight']
        if weight > threshold:
            edge_labels[(u, v)] = f"{weight:.3f}"
    
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, 
                                    bbox=dict(boxstyle="round,pad=0.2", 
                                            facecolor="yellow", alpha=0.7))
    
    if features is not None:
        ax = plt.gca()
        for i in range(num_nodes):
            x, y = pos[i]
            
            feature_str = [f"{val:.2f}" for val in features[i]]
            feature_text = f"[{', '.join(feature_str)}]"
            class_info = f"C{labels[i]}" if labels is not None else ""
            
            node_importance = np.sum(attention_matrix[:, i])
            importance_info = f"Imp: {node_importance:.3f}"
            
            if class_info:
                full_text = f"{class_info} | {importance_info}\n{feature_text}"
            else:
                full_text = f"{importance_info}\n{feature_text}"
            
            offset_x = 0.2
            offset_y = -0.1
            
            # Increased font size for attention graph features too
            ax.text(x + offset_x, y + offset_y, full_text, 
                   fontsize=14,  # Increased from 16
                   fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.4",  # Increased padding
                            facecolor="lightyellow", 
                            edgecolor="orange",
                            alpha=0.9),
                   verticalalignment='center',
                   horizontalalignment='left')
    
    legend_elements_combined = []
    
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            legend_elements_combined.append(
                plt.scatter([], [], c=[colors[i]], s=400, 
                           label=f'Clase {label}', alpha=0.8)
            )
        
        # Add separator and attention info
        legend_elements_combined.extend([
            plt.Line2D([0], [0], color='white', lw=0, label=''),  # Separator
            plt.Line2D([0], [0], color='red', lw=4, alpha=0.8, label='Atención Alta'),
            plt.Line2D([0], [0], color='lightcoral', lw=2, alpha=0.6, label='Atención Baja')
        ])
        
        title = "Clases y Pesos de Atención"
    else:
        # Just attention info if no classes
        legend_elements_combined.extend([
            plt.Line2D([0], [0], color='red', lw=4, alpha=0.8, label='Atención Alta'),
            plt.Line2D([0], [0], color='lightcoral', lw=2, alpha=0.6, label='Atención Baja')
        ])
        title = "Pesos de Atención"
    
    # Create unified legend
    unified_legend = plt.legend(handles=legend_elements_combined, 
                               loc='upper left', 
                               fontsize=legend_fontsize-2,
                               title=title,
                               title_fontsize=legend_fontsize,
                               frameon=True,
                               fancybox=True,
                               shadow=True,
                               framealpha=0.95,
                               bbox_to_anchor=(0.02, 0.98))
    
    avg_attention = np.mean(attention_matrix[attention_matrix > 0])
    title_full = f"{title}\n(Grosor ∝ peso de atención, Promedio: {avg_attention:.4f})"
    plt.title(title_full, fontsize=16, fontweight='bold', pad=20)
    
    plt.axis('off')
    plt.margins(0.25)
    plt.tight_layout()
    plt.show()
    
    print(f"\n  ANÁLISIS DE ATENCIÓN DIRIGIDA:")
    print(f"  Peso promedio: {avg_attention:.4f}")
    print(f"  Peso máximo: {max_attention:.4f}")
    print(f"  Peso mínimo: {min_attention:.4f}")
    
    print(f"\n  TOP 5 CONEXIONES DE ATENCIÓN:")
    flat_indices = np.argsort(attention_matrix.flatten())[-5:][::-1]
    
    for idx in flat_indices:
        i, j = np.unravel_index(idx, attention_matrix.shape)
        weight = attention_matrix[i, j]
        if weight > 0 and adj[i, j] > 0:  # Solo mostrar si existe la arista
            reciprocal = f" (recíproco: {attention_matrix[j, i]:.4f})" if adj[j, i] > 0 else ""
            print(f"    {node_names[i]} → {node_names[j]}: {weight:.4f}{reciprocal}")
    
    if directed:
        print(f"\n  ANÁLISIS DE ASIMETRÍA DE ATENCIÓN:")
        asymmetries = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and adj[i, j] > 0 and adj[j, i] > 0:  # Aristas recíprocas
                    att_ij = attention_matrix[i, j]
                    att_ji = attention_matrix[j, i]
                    asymmetry = abs(att_ij - att_ji)
                    asymmetries.append((i, j, att_ij, att_ji, asymmetry))
        
        if asymmetries:
            asymmetries.sort(key=lambda x: x[4], reverse=True)
            print(f"    Top 3 asimetrías más grandes:")
            for i, (u, v, att_uv, att_vu, asym) in enumerate(asymmetries[:3]):
                print(f"      {i+1}. {node_names[u]}⟷{node_names[v]}: "
                      f"{att_uv:.4f} vs {att_vu:.4f} (diff: {asym:.4f})")
        else:
            print(f"    No se encontraron aristas recíprocas para analizar asimetría")
    
    if features is not None:
        print(f"\n  NODOS MÁS IMPORTANTES POR ATENCIÓN RECIBIDA:")
        node_importance = np.sum(attention_matrix, axis=0)
        top_nodes = np.argsort(node_importance)[-3:][::-1]
        
        for node_idx in top_nodes:
            if node_importance[node_idx] > 0:
                feature_str = [f"{x:6.3f}" for x in features[node_idx]]
                print(f"    {node_names[node_idx]} (importancia: {node_importance[node_idx]:.3f}): "
                      f"[{', '.join(feature_str)}]")