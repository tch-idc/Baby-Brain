class BabyAI:
    def __init__(self):
        self.memory = []
        self.mood = "neutral"

    def learn(self, text):
        self.memory.append(text)

    def think(self):
        if self.mood == "happy":
            return "I feel good today "
        elif self.mood == "angry":
            return "I don't like that "
        else:
            return "I am learning..."

baby = BabyAI()

baby.learn("Fire is hot")
baby.learn("Sky is blue")

print(baby.think())
print("Memory:", baby.memory)

