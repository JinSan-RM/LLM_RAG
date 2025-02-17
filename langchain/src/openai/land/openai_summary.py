import asyncio
import json
import re
from typing import List

import time
import textwrap

from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

class OpenAISummaryClient:
    def __init__(self, pdf_data: str, batch_handler):
        self.pdf_data = pdf_data
        self.batch_handler = batch_handler

    @staticmethod
    def extract_json(text):
        try:
            # 불완전한 JSON 문자열 완성 시도
            text = text.strip()
            if text.endswith(','):
                text = text[:-1]
            if not text.endswith('}'):
                text += '}'
            return json.loads(text)
        except json.JSONDecodeError:
            print("JSON 파싱 실패")
            return None


# #========================================    
# #    
# #   일반 CoD 방식
# #
# #========================================    
#     # desired_summary_length는 아마 max_words와 연관있을 듯. 좀 더 파악 후 사용용
#     async def summarize_text_by_CoD(self, text_checks: list, desired_summary_length: int) -> str:
#         prompt = hub.pull("whiteforest/chain-of-density-prompt")

#         json_parser = SimpleJsonOutputParser()

# #========================================        
# #   체인 정의
# #========================================
#         # Chain Inputs
#         # Corresponding to the prompt placeholders:

#         cod_chain_inputs = {
#             'content': lambda d: d.get('content'),
#             'content_category': lambda d: d.get('content_category', "Article"),
#             'entity_range': lambda d: d.get('entity_range', '1-3'),
#             'max_words': lambda d: int(d.get('max_words', 80)),
#             'iterations': lambda d: int(d.get('iterations', 3))
#         }

#         ### Create two chains for testing.
#         # First one we'll stream and it outputs all intermediate
#         # summaries and missing entities that were added to the summary.
#         #
#         # The second one will parse the final summary from the JSON list
#         # of summaries. This one we can't stream because we need the
#         # final result.

#         # 1st chain, showing intermediate results, can async stream
#         cod_streamable_chain = (
#             cod_chain_inputs
#             | prompt
#             | ChatOpenAI(temperature=0, model='gpt-4')
#             | json_parser
#         )

#         # Create the 2nd chain, for extracting the best summary only.
#         # Not streamable, we need the final result.
#         cod_final_summary_chain = (
#             cod_streamable_chain
#             | (lambda output: output[-1].get('denser_summary', 'ERR: No "denser_summary" key in last dict'))
#         )        
# #========================================        
# #   코드 실행 부분
# #========================================
#         cod_summaires = []
        
#         for chunk in text_checks:
#             results = []
#             for partial_json in cod_streamable_chain({"content": chunk, "content_category": "Blog Post"}):
#                 results = partial_json
                
#             cod_summary = cod_final_summary_chain.invoke({"content": results, "content_category": "Blog Post"})
#             cod_summaires.append(cod_summary)
            
#         final_combined_summary = "\n\n".join(cod_summaires)
        
#         for partial_json in cod_streamable_chain({"content": final_combined_summary, "content_category": "Blog Post"}):
#             results = partial_json
            
#         final_summary = cod_final_summary_chain.invoke({"content": final_combined_summary, "content_category": "Blog Post"})
        
#         return final_summary
        
        
    def chunking_text(self, content: str, chunk_size: int = 1000, chunk_overlap: int = 50) -> List[str]:
    

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,   # 한 Chunk의 최대 길이 글자 수를 의미미
            chunk_overlap=chunk_overlap  # Chunk 간 오버랩 (중복 내용 포함)
        )
        text_chunks = text_splitter.split_text(content)
        return text_chunks

    async def summarize_chunked_texts(self, pdf_texts: str) -> str:
        try:
            # 0. 텍스트 청킹 수행
            chunk_texts = self.chunking_text(pdf_texts)  # No need to await since it's now a regular function
            
            # 1. 각 청크별 요약 수행
            chunk_summaries = []
            for chunk in chunk_texts:
                summary = await self.summarize_text_with_CoD(
                    content=chunk,
                    content_category="business report",
                    entity_range=3,
                    max_words=80,
                    iterations=3
                )
                if summary:  # 빈 문자열이 아닌 경우만 추가
                    chunk_summaries.append(summary)
            
            # 요약 결과가 없는 경우 처리
            if not chunk_summaries:
                print("모든 청크 요약에 실패했습니다.")
                return ""
            
            # 2. 모든 요약 데이터 종합합
            combined_summaries = "\n\n".join(chunk_summaries)
            
            # 3. 최종 요약 수행
            final_summary = await self.summarize_text_with_CoD(
                content=combined_summaries,
                content_category="business report",
                entity_range=3,
                max_words=500,
                iterations=5
            )
            
            return final_summary

        except Exception as e:
            print(f"Error during chunk summarization: {str(e)}")
            return ""    
    # async def summarize_chunked_texts(self, pdf_texts: str) -> str:
    #     try:
    #         # 0. 텍스트 청킹 수행
    #         chunk_texts = self.chunking_text(pdf_texts,
    #                                          chunk_size = 1000,
    #                                          chuck_overlap = 50)
            
    #         # 1. 각 청크별 요약 수행
    #         chunk_summaries = []
    #         for chunk in chunk_texts:
                
    #             # NOTE : 
    #             summary = await self.summarize_text_with_CoD(
    #                 content = chunk,
    #                 content_category = "business report",
    #                 entity_range = 3,
    #                 max_words = 80,
    #                 iterations = 3
    #                 )
                
    #             if summary:  # 빈 문자열이 아닌 경우만 추가
    #                 chunk_summaries.append(summary)
            
    #         # 요약 결과가 없는 경우 처리
    #         if not chunk_summaries:
    #             print("모든 청크 요약에 실패했습니다.")
    #             return ""
            
    #         # 2. 모든 요약을 하나의 텍스트로 결합
    #         combined_summaries = "\n\n".join(chunk_summaries)
            
    #         # 3. 최종 요약 수행
    #         final_summary = await self.summarize_text_with_CoD(
    #                 content = combined_summaries,
    #                 content_category = "business report",
    #                 entity_range = 3,
    #                 max_words = 500,
    #                 iterations = 5
    #                 )
            
    #         return final_summary

    #     except Exception as e:
    #         print(f"청크 요약 처리 중 오류 발생: {str(e)}")
    #         return ""


    async def summarize_text_with_CoD(self, content: str, content_category: str = "business report", entity_range:int = 3, max_words:int = 80, iterations:int = 3) -> str:

        # content_category: Title Case, e.g., Article, Video Transcript, Blog Post, Research Paper. Default Article
        # content: Content to summarize.
        # entity_range: String range of how many entities to pick from the content and add to the summary. Default 1-3.
        # max_words: Summary maximum length in words. Default 80.
        # iterations: Number of entity densification rounds. Total summaries is iterations+1. For 80 words, 3 iterations is ideal. Longer summaries could benefit from 4-5 rounds, and also possibly sliding the entity_range to, e.g., 1-4. Default: 3.
    
        messages = [
                    {
                        "role": "system",
                        "content": f"""
                        As an expert copy-writer, you will write increasingly concise, entity-dense summaries of the user provided {content_category}. The initial summary should be under {max_words} words and contain {entity_range} informative Descriptive Entities from the {content_category}.

                        A Descriptive Entity is:
                        - Relevant: to the main story.
                        - Specific: descriptive yet concise (5 words or fewer).
                        - Faithful: present in the {content_category}.
                        - Anywhere: located anywhere in the {content_category}.

                        # Your Summarization Process
                        - Read through the {content_category} and the all the below sections to get an understanding of the task.
                        - Pick {entity_range} informative Descriptive Entities from the {content_category} (";" delimited, do not add spaces).
                        - In your output JSON list of dictionaries, write an initial summary of max {max_words} words containing the Entities.
                        - You now have `[{{"missing_entities": "...", "denser_summary": "..."}}]`

                        Then, repeat the below 2 steps {iterations} times:

                        - Step 1. In a new dict in the same list, identify {entity_range} new informative Descriptive Entities from the {content_category} which are missing from the previously generated summary.
                        - Step 2. Write a new, denser summary of identical length which covers every Entity and detail from the previous summary plus the new Missing Entities.

                        A Missing Entity is:
                        - An informative Descriptive Entity from the {content_category} as defined above.
                        - Novel: not in the previous summary.

                        # Guidelines
                        - The first summary should be long (max {max_words} words) yet highly non-specific, containing little information beyond the Entities marked as missing. Use overly verbose language and fillers (e.g., "this {content_category} discusses") to reach ~{max_words} words.
                        - Make every word count: re-write the previous summary to improve flow and make space for additional entities.
                        - Make space with fusion, compression, and removal of uninformative phrases like "the {content_category} discusses".
                        - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the {content_category}.
                        - Missing entities can appear anywhere in the new summary.
                        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
                        - You're finished when your JSON list has 1+{iterations} dictionaries of increasing density.

                        # IMPORTANT
                        - Remember, to keep each summary to max {max_words} words.
                        - Never remove Entities or details. Only add more from the {content_category}.
                        - Do not discuss the {content_category} itself, focus on the content: informative Descriptive Entities, and details.
                        - Remember, if you're overusing filler phrases in later summaries, or discussing the {content_category} itself, not its contents, choose more informative Descriptive Entities and include more details from the {content_category}.
                        - Answer with a minified JSON list of dictionaries with keys "missing_entities" and "denser_summary".

                        ## Example output
                        [{{"missing_entities": "ent1;ent2", "denser_summary": "<vague initial summary with entities 'ent1','ent2'>"}}, {{"missing_entities": "ent3", "denser_summary": "denser summary with 'ent1','ent2','ent3'"}}, ...]
                        """ 
                    },
                    {
                        "role": "user",
                        "content": f"""
                        {content_category}:
                        {content}
                        """
                    }
                ]

        try:
            
            # NOTE : 출력 확인해서 json_parser 사용. 아마 str으로 나올거여서 json_parser로 형식변환 해줘야 아래처럼 사용 가능할 듯듯
            json_parser = SimpleJsonOutputParser()
            
            request = {
                "model": "/usr/local/bin/models/EEVE-Korean-Instruct-10.8B-v1.0",
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.1,
                "top_p": 0.8
            }
            print(f"[DEBUG] Sending request to LLM with payload:\n{json.dumps(request, indent=2)}")
            
            result = await asyncio.wait_for(
                self.batch_handler.process_single_request(request, 0),
                timeout=120
            )
            
            print(f"summary LLMResult 구조: {type(result.data)}")
            print(f"summary LLMResult 내용: {result.data}")
            
            if result.success:
            # Extract the summary from the response
                if hasattr(result.data, 'generations') and result.data.generations:
                    response_text = result.data.generations[0][0].text.strip()
                    print(f"[DEBUG] Extracted summary: {response_text}")
                    return response_text  # Return the generated summary
                else:
                    print("[ERROR] Missing 'generations' in LLMResult data.")
                    return "[ERROR] Missing 'generations' in LLMResult data."
            else:
                # Log the error details if the request failed
                print(f"[ERROR] Summary generation failed with error: {result.error}")
                return f"[ERROR] Summary generation failed with error: {result.error}"

        except Exception as e:
            # Log any unexpected errors
            print(f"[ERROR] Unexpected error during summarization: {str(e)}")
            return f"[ERROR] Unexpected error during summarization: {str(e)}"
            
        
        
    async def generate_proposal(self, summary: str):
        try:
            """prompt = f
            Create a website proposal based on the following summary:
            {summary}

            The proposal should include:
            - Website name
            - Keywords (7 items)
            - Purpose (200 characters)
            - Target audience (3 items)
            - Core values (3 items)
            - Main service
            - Key achievements (5 items)

            Provide the output in JSON format.
            """
            
            prompt = f"""
            System:
            Your task is to interpret and transform the content of an uploaded summary data into a website proposal. Analyze the data and fit it to the componants below.

            Instructions: 
            Interpret the content to create a comprehensive, engaging a website proposal.
            Write between 1500-2000 characters in summary for the website proposal on main language in summary. 
            Ensure the script is logically structured with clear sections and transitions. Maintain all core information without summarizing. 
            Avoid the hallucination and frame the script as an independent narrative. 
            Do not use bullet points; ensure a smooth narrative flow. 

            Organize the script into json type: start with "key" and end with "value"

            Output Format: 
            "Identify business goals" : "'Description include Business goals you want to achieve through this website / Problems or inconveniences you are currently experiencing / Changes or effects expected through the website'", 
            "Customize your target" : "Description include 'Who will mainly use it? / For what purpose do users visit? / Market trend analysis'", 
            "Derive core functions" : "Description include 'Differentiating features compared to competitors / Need for integration with existing systems or services', 
            "design requirements" : "Description include Colors that match the brand identity"

            ensure that the output mathces the JSON output example below.
            Example JSON Output:
            \\\\\\`json {{ "Identify business goals": "description", "Customize your target" : "description",  "Derive core functions" : "Description", "design requirements" : "description"}}
            
            Content: 
            {summary}
            """            
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                        "prompt": prompt,
                        "max_tokens": 1000,
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "n": 1,
                        "stream": False,
                        "logprobs": None
                    }, request_id=0),
                timeout=60  # 적절한 타임아웃 값 설정
            )
            return response
        except Exception as e:
            print(f"제안서 생성 중 예상치 못한 오류: {str(e)}")
            return ""
        
    async def process_pdf(self):
        try:
            summary = await self.summarize_text(self.pdf_data, 2000)
            if not summary:
                print("요약 생성 실패")
                return ""
            
            proposal = await self.generate_proposal(summary)
            if not proposal:
                print("제안서 생성 실패")
                return ""
            
            return proposal  # 이 값을 직접 사용할 수 있습니다
        except Exception as e:
            print(f"PDF 처리 중 오류: {str(e)}")
            return ""

        
        
        
        
        
#========================================        
#   기존 버전
#========================================        

    async def summarize_text(self, text: str, desired_summary_length: int) -> str:

        try:
            prompt = f"""
            System:
            Summarize the following text in about {desired_summary_length} characters.
            
            User:
            {text}
            """
            
            response = await asyncio.wait_for(
                self.batch_handler.process_single_request({
                        "prompt": prompt,
                        "max_tokens": 1000,
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "n": 1,
                        "stream": False,
                        "logprobs": None
                    }, request_id=0),
                timeout=60  # 적절한 타임아웃 값 설정
            )
            return response
        except asyncio.TimeoutError:
            print("요약 요청 시간 초과")
            return ""
        except Exception as e:
            print(f"요약 중 예상치 못한 오류: {str(e)}")
            return ""

