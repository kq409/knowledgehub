from openai import OpenAI

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingService:
    def __init__(self, llm_client: OpenAI, model: str):
        self.llm_client = llm_client
        self.model = model

    def embed(self, text: str, *, prefix: str) -> list[float]:
        payload = f"{prefix}{text}" if prefix else text
        response = self.llm_client.embeddings.create(
            model=self.model,
            input=payload,
        )
        return list(response.data[0].embedding)

    def embed_document(self, text: str) -> list[float]:
        return self.embed(text, prefix=DOCUMENT_PREFIX)

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text, prefix=QUERY_PREFIX)
