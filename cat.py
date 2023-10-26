class Cat():
    def __init__(self, name):

        if not isinstance(name, str):
            raise ValueError(f"String expected, got {type(name)} instead")
        
        self.name = name

    def __repr__(self) -> str:
        return f"The name of the cat is {self.name}"
    
    def meow(self):
        return f"Meow, I'm {self.name}. \n\rI'm exmeowted to see you!"
    
    def meow_at(self, other):
        if not isinstance(other, Cat):
            raise ValueError(f"Expected {type(self)}, got {type(other)} instead")
        return f"HiMeow, {other.name}, I'm {self.name}"
    
        
