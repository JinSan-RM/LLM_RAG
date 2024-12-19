import operator
from typing import Annotated, List, Literal, TypedDict, AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
import requests
import json
from utils.ollama.ollama_client import OllamaClient
import logging, asyncio

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

token_max = 3500

class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str

class SummaryState(TypedDict):
    content: str

class SummaryContentChain:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.summary_graph = self._build_summary_graph()
        
    
    def _build_summary_graph(self):
        graph = StateGraph(OverallState)
        graph.add_node("generate_summary", self.generate_summary)
        graph.add_node("collect_summaries", self.collect_summaries)
        graph.add_node("collapse_summaries", self.collapse_summaries)
        graph.add_node("generate_final_summary", self.generate_final_summary)

        graph.add_conditional_edges(START, self.map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", self.should_collapse)
        graph.add_conditional_edges("collapse_summaries", self.should_collapse)
        graph.add_edge("generate_final_summary", END)

        # Ensure the compiled graph is callable
        compiled_graph = graph.compile()
        logger.debug(f"Compiled graph created: {compiled_graph}")
        return compiled_graph

    def length_function(self, documents: List[Union[Document, str]]) -> int:
        """Get number of tokens for input contents."""
        # Handle both Document objects and strings
        return sum(
            self.ollama_client.get_num_tokens(
                doc.page_content if isinstance(doc, Document) else doc
            )
            for doc in documents
        )

    async def generate_summary(self, state: SummaryState):
        token_count = self.get_num_tokens(state["content"])
        logger.debug(f"Generating summary for content with {token_count} tokens.")
        logger.debug(f"Content Preview: {state['content'][:200]}...")

        prompt = f"요약: {state['content']}"
        response = await self.ainvoke(prompt)
        logger.debug(f"Generated Summary: {response[:200]}...")
        return {"summaries": [response]}

    def map_summaries(self, state: OverallState):
        logger.debug("Mapping summaries for input contents.")
        return [
            Send("generate_summary", {"content": content}) for content in state["contents"]
        ]

    def collect_summaries(self, state: OverallState):
        logger.debug("Collecting summaries into collapsed summaries.")
        # Ensure summaries are added as strings
        return {
            "collapsed_summaries": state["summaries"]  # 그대로 문자열 리스트로 처리
        }
    async def _reduce(self, input: dict) -> str:
        combined_text = ' '.join(input)
        token_count = self.get_num_tokens(combined_text)
        logger.debug(f"Reducing input with {token_count} tokens.")
        logger.debug(f"Combined Text Preview: {combined_text[:200]}...")

        prompt = f"최종 요약: {combined_text}"
        response = await self.ainvoke(prompt)
        logger.debug(f"Reduced Summary: {response[:200]}...")
        return response

    async def collapse_summaries(self, state: OverallState):
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], self.length_function, token_max
        )

        for i, doc_list in enumerate(doc_lists):
            token_count = sum(self.ollama_client.get_num_tokens(doc) for doc in doc_list)
            logger.debug(f"Chunk {i + 1}:")
            for doc in doc_list:
                logger.debug(f"  Content: {doc[:100]}...")  # 문자열이므로 그대로 출력
            logger.debug(f"  Total tokens: {token_count}\n")

        results = []
        for doc_list in doc_lists:
            combined_text = " ".join(doc_list)
            response = await self._reduce([combined_text])
            results.append(response)
        return {"collapsed_summaries": results}

    def should_collapse(
        self, state: OverallState
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = self.length_function(state["collapsed_summaries"])
        logger.debug(f"Total tokens before deciding to collapse: {num_tokens}")
        if num_tokens > token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

    async def generate_final_summary(self, state: OverallState):
        # Safely extract text content from collapsed_summaries
        collapsed_summaries = [
            doc.page_content if isinstance(doc, Document) else doc
            for doc in state["collapsed_summaries"]
        ]
        token_count = self.length_function(collapsed_summaries)
        logger.debug(f"Generating final summary from {token_count} tokens.")

        # Reduce to a final summary
        response = await self._reduce(collapsed_summaries)
        logger.debug(f"Final Summary: {response[:200]}...")
        return {"final_summary": response}

    async def summary(self, input_text: str) -> str:
        logger.debug(f"Preparing graph state for input of length {len(input_text)}.")
        state = {
            "contents": [input_text],
            "summaries": [],
            "collapsed_summaries": [],
            "final_summary": "",
        }
        logger.debug(f"Initial State: {state}")

        # 디버깅으로 실행 가능한 메서드 시도
        try:
            if hasattr(self.summary_graph, "ainvoke"):
                final_state = await self.summary_graph.ainvoke(state)
            elif hasattr(self.summary_graph, "astream"):
                async for final_state in self.summary_graph.astream(state):
                    logger.debug(f"Intermediate state: {final_state}")
            elif hasattr(self.summary_graph, "abatch"):
                final_state = await self.summary_graph.abatch(state)
            else:
                raise AttributeError("No valid execution method found for summary_graph.")
            
            logger.debug(f"Graph execution completed. Final state: {final_state}")
            return final_state["final_summary"]
        except Exception as e:
            logger.error(f"Error in summary graph execution: {e}")
            raise
    
    async def ainvoke(self, prompt: str) -> str:
        """
        Simulate an asynchronous call to an AI model with the given prompt.
        Replace this logic with the actual implementation for the AI model.
        """
        logger.debug(f"Ainvoke called with prompt: {prompt[:200]}...")
        # Simulated delay for asynchronous behavior
        await asyncio.sleep(1)  # Simulate network delay or processing time
        return f"Simulated response for prompt: {prompt[:200]}..."
    
    
    def get_num_tokens(self, text: str) -> int:
        # 간단한 토큰 수 계산 (단어 개수로 간주)
        return len(text.split())
    
    async def run(self, input_text, model: str, value_type="general"):
        logger.debug(f"run operation success \nvalue_type: {value_type}\nmodel: {model}")
        try:
            final_output = await self.summary(input_text)
            logger.debug(f"Final result: {final_output[:200]}...")
            return final_output
        except Exception as e:
            logger.error(f"Error in run: {e}")
            raise