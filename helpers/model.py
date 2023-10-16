class ModelLoader:
    def __init__(self, name, model) -> None:
        self.name = name
        self.model = model
        self.type = ''

    def __str__(self) -> str:
        return f'Model: {self.name}, Type: {self.type}'

    def load(self):
        return self.model
