from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests

class EnhancedLLMAgent:
    def __init__(self, model_name="t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.memory = []

    def remember(self, context):
        self.memory.append(context)
        if len(self.memory) > 10:
            self.memory.pop(0)

    def recall_memory(self):
        return " ".join(self.memory)

    def plan_task(self, task_description):
        prompt = f"Break down the task '{task_description}' into steps."
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"], max_length=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def reason(self, query):
        prompt = f"Answer logically: {query}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"], max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def search_web(self, query):
        response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
        return response.json().get("Abstract", "No results found.")

# Usage
agent = EnhancedLLMAgent()
task = "Organize a virtual conference on AI ethics."
result = agent.plan_task(task)
print(f"Plan: {result}")
