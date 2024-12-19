from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.prompts import PromptTemplate
from typing import List, Optional
from utils.helpers import languagechecker
from modules.translators import KoEnTranslator
from utils.ollama.ollama_embedding import get_embedding_from_ollama

class CustomRAGChain(BaseModel):
    retriever: VectorStoreRetriever = Field(..., description="Retriever instance for document retrieval")
    llm: BaseLLM = Field(..., description="LLM instance for text generation")
    prompt_template: PromptTemplate = Field(..., description="Prompt template to be used")

    def __init__(self, **data):
        super().__init__(**data)
        print(f"CustomRAGChain initialized with retriever: {self.retriever}, llm: {self.llm}, prompt_template: {self.prompt_template}")

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(self, inputs):
        """
        체인의 호출 메서드
        :param inputs: {"question": 사용자 질문}
        :return: {"answer": LLM에서 생성된 응답}
        """
        query_embedding = None
        question = inputs["question"]

        try:
            input_text = question

            # 언어 감지
            discriminant = languagechecker(input_text)

            # OllamaClient로 임베딩 생성 (언어에 따라 번역 포함)
            if discriminant:
                translator = KoEnTranslator()
                input_text = translator.translate(input_text)
                query_embedding = get_embedding_from_ollama(input_text)
            else:
                query_embedding = get_embedding_from_ollama(input_text)

            print(f"Translated text: {input_text}")
            docs = self.retriever.invoke(input_text, filter={"source":"question"})
            print(f"[INFO] docs json data : {docs}") 
            if docs is None or len(docs) == 0:
                print("[ERROR] No documents retrieved. The returned value is empty.")
            else:
                context = "\n\n".join([doc.page_content for doc in docs])
                print(f"Context: {context}")
        except Exception as e:
            print(f"[ERROR] Error invoke : {str(e)}")
            raise

        # 2. 프롬프트 생성
        prompt = self.prompt_template.format(context=context, question=question)

        # 3. LLM 호출
        answer = self.llm(prompt)
        return {"answer": answer}


    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResult:
        responses = [self._call({"question": prompt})["answer"] for prompt in prompts]
        return LLMResult(generations=[[{"text": resp}] for resp in responses])

    def __call__(self, inputs: dict) -> dict:
        """추가된 __call__ 메서드로 객체를 함수처럼 호출할 수 있도록 만듭니다."""
        return self._call(inputs)
