from langchain.prompts import PromptTemplate
from modules.translators import KoEnTranslator, EnKoTranslator
from utils.ollama.ollama_client import OllamaClient
import re, json


class ContentChain:
    def __init__(self):
        self.ko_en_translator = KoEnTranslator()
        self.en_ko_translator = EnKoTranslator()
        self.ollama_client = OllamaClient()

        # self.text_generator = TextGenerator()
        # 프롬프트 템플릿 설정
        
    # 일괄 처리 방식
    # def run(self, input_text, discriminant, model="llama3.2", value_type = "general"):
    def run(self, input_text, discriminant=False, model="bllossom", value_type = "general"):
        """
        Ollama API 기반 텍스트 생성 체인
        Args:
            input_text (str): 입력 텍스트
            discriminant (bool): 한국어 여부
            model (str): Ollama에서 사용할 모델 이름

        Returns:
            str: 최종 생성 결과
        """
        print(f"run operation success \n value_type : {value_type}\n model : {model}" )
        final_output = None
        
        if value_type == "general":
            if discriminant:
                # 한->영 번역
                translated_text = self.ko_en_translator.translate(input_text)
                # Ollama API 호출
                generated_text = self.ollama_client.generate(model, input_text)
                print("Generated Text:", generated_text)
                # 영->한 번역
                final_output = self.en_ko_translator.translate(generated_text)
                return final_output
            else:
                # Ollama API 호출
                generated_text = self.ollama_client.generate(model, input_text)
                print("Generated Text:", generated_text)
                # 영->한 번역
                final_output = self.en_ko_translator.translate(generated_text)
                return final_output
        if value_type == 'normal':
            
            prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
            - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
            - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
            1. assistant처럼 생성해야 하고 형식을 **절대** 벗어나면 안 된다.
            2. "div, h1, h2, h3, p, ul, li" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
            3. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
            4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
            5. 출력 결과는 코드 형태만 허용된다. 코드는 **절대 생성하지 마라.**
            6. 오직 한글로만 작성하라.
        

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            입력 데이터:
            {input_text}
            


            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            - 너는 코드 구조 응답만을 반환해야 한다.
            """
            generated_text = self.ollama_client.generate(model, prompt)
            # generated_text = self.ollama_client.generate(model, input_text)
            print("Generated Text:", generated_text)
            return generated_text
       
        elif value_type == "menu":
            translated_text = None
            if discriminant and model =='llama3.2':
                translated_text = self.ko_en_translator.translate_length_limit(input_text)
                print(f"Translated Input Text  discriminant True: {translated_text} <=====translated_text, discriminant True")
                generated_text = self.ollama_client.PDF_Menu(model, translated_text)
                print(f"Generated Text: {generated_text}")
                if not generated_text or generated_text == "Empty response received":
                    print("No valid response from PDF_Menu.")
                    return None
                # 데이터를 제대로 생성 못했을 시 한번 더 진행핑 시진핑 도핑 서핑.
                    
                # 필드 추출
                title_structure = self.extract_field(generated_text, "title_structure")
                keywords_structure = self.extract_field(generated_text, "keywords_structure")
                menu_structure = self.extract_field(generated_text, "menu_structure")                

                # 딕셔너리 생성
                translate_list = {
                    'title_structure': title_structure if title_structure else "",  
                    'keywords_structure': keywords_structure if keywords_structure else [],
                    'menu_structure': menu_structure if menu_structure else []
                }

                # 개별 번역
                translated_title = self.en_ko_translator.translate(translate_list['title_structure']) if translate_list['title_structure'] else ""
                translated_keywords = [self.en_ko_translator.translate(kw) for kw in translate_list['keywords_structure']] if translate_list['keywords_structure'] else []
                translated_menu = [self.translate_with_formatting(item) for item in translate_list['menu_structure']] if translate_list['menu_structure'] else []

                # 최종 딕셔너리 생성
                final_output = {
                    'title_structure': translated_title,
                    'keywords_structure': translated_keywords,
                    'menu_structure': translated_menu
                }

                print(f"Final Translated Output: {final_output} <----final_output")
                return final_output
            
            elif model == "bllossom":
                generated_text = self.ollama_client.PDF_Menu(model, input_text)
                print(f"{generated_text} : generated_text")
                return generated_text
            
            else:
                print(input_text,"<======input_text")
                generated_text = self.ollama_client.PDF_Menu(model, input_text)
                print(f"Generated Text: {generated_text}")
                if not generated_text or generated_text == "Empty response received":
                    print("No valid response from PDF_Menu.")
                    return None
                    
                # 필드 추출
                title_structure = self.extract_field(generated_text, "title_structure")
                keywords_structure = self.extract_field(generated_text, "keywords_structure")
                menu_structure = self.extract_field(generated_text, "menu_structure")

                print(f"Extracted Fields: title='{title_structure}', keywords={keywords_structure}, menu={menu_structure}")

                # 딕셔너리 생성
                translate_list = {
                    'title_structure': title_structure if title_structure else "",
                    'keywords_structure': keywords_structure if keywords_structure else [],
                    'menu_structure': menu_structure if menu_structure else []
                }

                # 개별 번역
                translated_title = self.en_ko_translator.translate(translate_list['title_structure']) if translate_list['title_structure'] else ""
                translated_keywords = [self.en_ko_translator.translate(kw) for kw in translate_list['keywords_structure']] if translate_list['keywords_structure'] else []
                translated_menu = [self.translate_with_formatting(item) for item in translate_list['menu_structure']] if translate_list['menu_structure'] else []

                # 최종 딕셔너리 생성
                final_output = {
                    'title_structure': translated_title,
                    'keywords_structure': translated_keywords,
                    'menu_structure': translated_menu
                }

                print(f"Final Translated Output: {final_output} <----final_output")
                return final_output
        else:
            print(f"Unsupported value_type: {value_type}")
            return final_output
    
    # streaming 처리 방식
    # def run(self, input_text, discriminant, model="llama3.2"):
    #     """
    #     Ollama API 기반 텍스트 생성 체인
    #     Args:
    #         input_text (str): 입력 텍스트
    #         discriminant (bool): 한국어 여부
    #         model (str): Ollama에서 사용할 모델 이름

    #     Returns:
    #         str: 최종 생성 결과
    #     """
    #     print("run operation success")
    #     if discriminant:
    #         # 한->영 -> 텍스트 생성 -> 영->한
    #         translated_text = self.ko_en_translator.translate(input_text)
    #         final_output = self._stream_generate_and_translate(model, translated_text, is_korean=True)
    #     else:
    #         # 영문 텍스트 바로 생성 -> 영->한
    #         final_output = self._stream_generate_and_translate(model, input_text, is_korean=False)
        
    #     return final_output
    
    def contents_run(self, model, input_text, menu):
        generated_text = self.ollama_client.PDF_Menu_Contents(model, input_text, menu)
        return generated_text
    def _stream_generate_and_translate(self, model, input_text, is_korean):
        """
        스트리밍 방식으로 Ollama 텍스트 생성 및 번역 처리
        Args:
            model (str): Ollama에서 사용할 모델 이름
            input_text (str): 입력 텍스트
            is_korean (bool): 한국어 여부

        Returns:
            str: 최종 생성 결과
        """
        # 스트리밍 데이터 받아오기
        streamed_text = self.ollama_client.generate(model, input_text)
        
        # 실시간 번역 처리
        translated_output = ""
        for chunk in streamed_text.split(" "):  # 단어 단위로 스트리밍 처리
            if is_korean:
                translated_chunk = self.en_ko_translator.translate(chunk)
            else:
                translated_chunk = chunk
            translated_output += translated_chunk + " "
            print("Translated Chunk:", translated_chunk)  # 디버깅용 출력
        
        return translated_output.strip()
    
    def extract_field(self, text: str, field: str):
        """
        JSON 라이브러리를 사용하지 않고 문자열에서 특정 필드의 값을 추출하는 함수
        Args:
            text (str): 전체 응답 문자열
            field (str): 추출할 필드 이름

        Returns:
            list 또는 str: 추출된 필드의 값
        """
        field_pattern = f'"{field}":'
        start_index = text.find(field_pattern)
        if start_index == -1:
            print(f"Field '{field}' not found.")
            return "" if field != "keywords_structure" else []

        # 값의 시작 위치 찾기
        start_index += len(field_pattern)
        # 공백과 시작 괄호, 따옴표 스킵
        while start_index < len(text) and (text[start_index].isspace() or text[start_index] in ['"', '[']):
            start_index += 1

        # 필드에 따라 다르게 처리
        if field == "title_structure":
            # 따옴표 안의 값 추출
            end_quote = text.find('"', start_index)
            if end_quote == -1:
                print(f"End quote for field '{field}' not found.")
                return ""
            value = text[start_index:end_quote]
            return value

        elif field == "keywords_structure":
            # 대괄호 안의 값 추출
            end_bracket = text.find(']', start_index)
            if end_bracket == -1:
                print(f"End bracket for field '{field}' not found.")
                return []
            list_content = text[start_index:end_bracket]
            # 콤마로 분리하고 따옴표와 공백 제거
            keywords = [kw.strip().strip('"') for kw in list_content.split(',')]
            return keywords

        elif field == "menu_structure":
        # 정규 표현식을 사용하여 메뉴 항목 추출
            pattern = r'(\d+\.\s*[^,"]+|-\s*[^,"]+)'
            menu_items = re.findall(pattern, text)
            return menu_items
        else:
            print(f"Unknown field '{field}'.")
            return ""

    
    def translate_with_formatting(self, text: str) -> str:
        pattern = r'^(\d+\.\s*|- )(.+?)(,)?$'
        match = re.match(pattern, text)
        if match:
            prefix = match.group(1)  # 숫자. 또는 - 
            main_text = match.group(2)  # 번역할 텍스트
            suffix = match.group(3) if match.group(3) else ''  # 뒤에 오는 콤마
            try:
                translated_text = self.en_ko_translator.translate(main_text)
                translated_item = f"{prefix}{translated_text}{suffix}"
                print(f"Translated with formatting: {translated_item}")
                return translated_item
            except Exception as e:
                print(f"Error translating '{main_text}': {e}")
                return text  # 번역 실패 시 원본 텍스트 반환
        else:
            # 포맷팅 문자가 없는 경우 전체를 번역
            try:
                translated_text = self.en_ko_translator.translate(text)
                print(f"Translated without formatting: {translated_text}")
                return translated_text
            except Exception as e:
                print(f"Error translating '{text}': {e}")
                return text  # 번역 실패 시 원본 텍스트 반환

        
    def translate_structure(self, data):
        """
        딕셔너리나 리스트를 재귀적으로 순회하면서 문자열 값을 번역하는 함수
        Args:
            data (dict, list, str): 번역할 데이터 구조

        Returns:
            dict, list, str: 번역된 데이터 구조
        """
        if isinstance(data, dict):
            return {k: self.translate_structure(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.translate_structure(item) for item in data]
        elif isinstance(data, str):
            # 포맷팅 문자를 유지하며 번역
            return self.translate_with_formatting(text=data)
        else:
            return data
    # def run(self, input_text, discriminant):
    #     if discriminant:
    #         # 한->영 -> 텍스트 생성 -> 영->한
    #         translated_text = self.ko_en_translator.translate(input_text)
    #         generated_text = self.text_generator.generate(self.prompt_template.format(text=translated_text))
    #         final_output = self.en_ko_translator.translate(generated_text)
    #     else:
    #         # 영문 텍스트 바로 생성 -> 영->한
    #         generated_text = self.text_generator.generate(self.prompt_template.format(text=input_text))
    #         final_output = self.en_ko_translator.translate(generated_text)
    #     return final_output

