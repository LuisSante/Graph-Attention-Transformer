class Config:
    SEED = 42
    N_NODES = 6  
    IN_FEATURES = 5
    OUT_FEATURES_PER_HEAD = 5  
    N_HEADS = 3
    CONCAT = True
    DIRECTED = True  
    GRAPH_DENSITY = 0.4
    LARGE_NEGATIVE_NUMBER = -1e30
    
    # Multi-graph configuration
    DETAIL_GRAPH = "multi_graph"  # "single_graph" or "multi_graph"
    N_GRAPHS = 3 
    BATCH_PROCESSING = "padding"  # "padding" or "sequential"
    MAX_NODES = N_NODES

    @staticmethod
    def print_separator(title, char="=", width=80):
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")

    @staticmethod
    def print_title(title, width=80):
        print(f"\n{title:^{width}}")
    
    @staticmethod
    def print_subsection(title, char="-", width=60):
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")