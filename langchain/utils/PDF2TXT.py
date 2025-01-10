import fitz, re, requests
from io import BytesIO
from fastapi import HTTPException

class PDFHandle():
    
    def __init__(self, path: str, path2: str = '', path3: str = ''):
        self.path = path
        self.path2 = path2
        self.path3 = path3
        
    def PDF2TEXT(self, pdf_list) -> str:
        """
        PDF 파일 리스트에서 텍스트를 추출하고 정리하는 함수
        
        Args:
            pdf_list: PDF 파일 객체들의 리스트
            
        Returns:
            str: 추출 및 정리된 텍스트
        """
        total_text = ""
        
        for pdf_file in pdf_list:
            doc = fitz.open(stream=pdf_file, filetype="pdf")
            num_pages = doc.page_count
            print(f"총 페이지 수: {num_pages}")
            
            def clean_text(text):
                # 각 줄을 분리한 후 빈 줄을 제거하고 다시 합침
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                cleaned_text = "\n".join(lines)
                # 여러 개의 공백을 하나로 줄임
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                return cleaned_text

            extracted_text = ""
            # 모든 페이지 텍스트 추출
            for page_num in range(num_pages):
                page = doc[page_num]
                text = page.get_text()
                
                # 기본 청소
                cleaned_text = clean_text(text)
                if cleaned_text:
                    # 추가 텍스트 정리
                    final_cleaned_text = self.clean_pdf_text(cleaned_text)
                    total_text += f"\n{final_cleaned_text}\n"
                
                extracted_text += f"\n{text}\n"
            
            total_text += extracted_text
            doc.close()

        # 최종 텍스트 정리
        final_text = self.clean_pdf_text(total_text)
        return final_text

    def clean_pdf_text(self, text):
        """
        PDF에서 추출한 텍스트를 정리하는 함수
        
        Args:
            text (str): PDF에서 추출한 원본 텍스트
            
        Returns:
            str: 정리된 텍스트
        """
        # 1. 연속된 공백을 하나로 통일
        text = re.sub(r'\s{2,}', ' ', text)
        
        # 2. 줄바꿈 처리
        # 문장 끝(마침표, 느낌표, 물음표 등) 다음의 줄바꿈은 유지
        text = re.sub(r'([.!?])\s*\n', r'\1\n', text)
        # 그 외의 줄바꿈은 공백으로 변환
        text = re.sub(r'(?<![.!?])\n', ' ', text)
        
        # 3. 연속된 줄바꿈을 최대 2개로 제한
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 4. 문장 시작과 끝의 불필요한 공백 제거
        text = text.strip()
        
        # 5. 문장 부호 앞의 불필요한 공백 제거
        text = re.sub(r'\s+([.,!?:])', r'\1', text)
        
        # 6. 괄호 주변의 불필요한 공백 정리
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        return text

    def PDF_request(self):
        """
        PDF에서 추출한 텍스트를 정리하는 함수
            
        Returns:
            str: PDF 기반 전체 텍스트
        """
        pdf_list = []
        response = requests.get(self.path)
        
        if response.status_code == 200:
            pdf_data = BytesIO(response.content)
            pdf_list.append(pdf_data)
            
            if self.path2 != ''  :
            # print("path2 : ", path2)
                response2 = requests.get(self.path2)
                pdf_data2 = BytesIO(response2.content)
                pdf_list.append(pdf_data2)
            if self.path3 != '' :
            # print("path3 : ", path3)
                response3 = requests.get(self.path3)
                pdf_data3 = BytesIO(response3.content)
                pdf_list.append(pdf_data3)
        
        # 응답 상태 확인
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="PDF 파일을 다운로드할 수 없습니다."
            )
        
        
        
        return self.PDF2TEXT(pdf_list=pdf_list)
            