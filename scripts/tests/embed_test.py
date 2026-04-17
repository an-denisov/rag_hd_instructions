from sentence_transformers import SentenceTransformer

model = SentenceTransformer("C:/AI/models/bge-m3")

text = """
Дата В II — Логика возвратного товара.
Возвратный остаток отображается, если у товара есть основание на возврат.
Если основания нет — строка не отображается.
"""

emb = model.encode(text)

print("Размер вектора:", len(emb))
print("Первые 5 значений:", emb[:5])
