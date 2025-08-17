class Config:
    SEED = 42
    N_NODES = 6  
    IN_FEATURES = 5
    HIDDEN_PER_HEAD = 5 ## mantener en 5 para que coincida con IN_FEATURES
    N_HEADS = 3
    GRAPH_EMBEDDING_DIM = 16  # CAMBIADO: Dimensión del embedding final del grafo
    CONCAT = True
    DIRECTED = True  
    GRAPH_DENSITY = 0.4
    LARGE_NEGATIVE_NUMBER = -1e30

    @staticmethod
    def print_separator(title, char="=", width=80):
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")
    
    @staticmethod
    def print_subsection(title, char="-", width=60):
        print(f"\n{char * width}")
        print(f"{title}")
        print(f"{char * width}")