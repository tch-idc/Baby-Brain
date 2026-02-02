import random

class BabyAI:
    def __init__(self):
        self.memory = []
        self.mood = "neutral"

    def learn(self, text):
        self.memory.append(text)

        # change mood based on input
        if "love" in text:
            self.mood = "happy"
        elif "hate" in text:
            self.mood = "angry"
        else:
            self.mood = random.choice(["neutral", "happy", "curious"])

    def think(self):
        if self.mood == "happy":
            return "I feel good today 😊"
        elif self.mood == "angry":
            return "I don't like that 😡"
        elif self.mood == "curious":
            return "Tell me more..."
        else:
            return "I am learning..."

    def recall(self):
        return f"I remember {len(self.memory)} things: {self.memory}"


baby = BabyAI()

baby.learn("I love python")
baby.learn("torch is mama")
baby.learn("I hate bugs")

print(baby.think())
print(baby.recall())
