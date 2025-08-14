class Config:
    SEED = 42
    IN_FEATURES = 5
    HIDDEN_PER_HEAD = 8
    N_HEADS = 3
    N_CLASSES = 3
    CONCAT = True
    LARGE_NEGATIVE_NUMBER = -1e30

    def print_subsection(title, char="-", width=60):
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")

    def print_separator(title, char="=", width=80):
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")