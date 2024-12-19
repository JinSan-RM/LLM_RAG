import requests, json, random, re
from config.config import OLLAMA_API_URL
from fastapi import HTTPException
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
from typing import List
from utils.ollama.ollama_embedding import get_embedding_from_ollama

class OllamaContentClient:
    def __init__(self, api_url=OLLAMA_API_URL+'api/generate', temperature=0.4, structure_limit = True,  n_ctx = 4196, max_token = 4196):
        self.api_url = api_url
        self.temperature = temperature
        self.structure_limit = structure_limit
        self.n_ctx = n_ctx
        self.max_token = max_token
        
    async def send_request(self, model: str, prompt: str) -> str:
        """
        공통 요청 처리 함수: API 호출 및 응답 처리
        """
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": self.temperature,
            "n_ctx": self.n_ctx,
            "repetition penalty":1.2,
            "session" : "test_session"
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

            full_response = response.text  # 전체 응답
            lines = full_response.splitlines()
            all_text = ""
            for line in lines:
                try:
                    json_line = json.loads(line.strip())  # 각 줄을 JSON 파싱
                    all_text += json_line.get("response", "")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    continue  # JSON 파싱 오류 시 건너뛰기
                
            return all_text.strip() if all_text else "Empty response received"

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 실패: {e}")
            raise RuntimeError(f"Ollama API 요청 실패: {e}")
    #========================================================================================
    # chunk test code
    async def generate_section(self, model: str, section_name: str) -> str:
        """
        랜딩 페이지 섹션을 생성하는 함수
        """
        prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
            - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
            - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
            - 섹션 '{section_name}'에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
            1. assistant처럼 생성해야 하고 형식을 **절대** 벗어나면 안 된다.
            2. "div, h1, h2, h3, p, ul, li" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
            3. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
            4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
            5. 출력 결과는 코드 형태만 허용된다. 코드는 **절대 생성하지 마라.**
            6. 오직 한글로만 작성하라.
        

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            입력 데이터:
            랜딩 페이지 섹션을 구성하기 위한 PDF 내용 전체가 에 포함되어 있습니다.
            섹션:
            {section_name}


            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            - 너는 코드 구조 응답만을 반환해야 한다.
        """
        return await self.send_request(model, prompt)  
    #========================================================================================  
    async def contents_GEN(self, model : str= "bllossom", input_text = "", section_name=""):
        
        prompt = f"""
                <|start_header_id|>system<|end_header_id|>
                - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
                - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
                - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
                - 섹션 '{section_name}'에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
                1. assistant처럼 생성해야하고 형식을을 **절대** 벗어나면 안 된다.
                2. "div, h1, h2, h3, p, ul, li" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
                3. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
                4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
                5. 출력 결과는 코드 형태만 허용된다. 코드는 **절대 생성하지 마라.**
                6. 오직 한글로만 작성하라.
                

                <|eot_id|><|start_header_id|>user<|end_header_id|>
                섹션:
                {section_name}
                

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                2021. v1.0 ⓒKG Inicis KG이니시스 회사소개서 Total Payment Service Provider 
                01. 회사소개 회사개요 회사연혁 매출, 거래액성장통계 사업영역 브랜드파워 02. 사업소개 INIPAY 전자결제 INIVAN / 테이블오더 간편결제 이니렌탈서비스 일본결제서비스 광고플랫폼서비스 부가서비스 03. 파트너소개 Partners Contact us Contents 
                1. 회사개요 2. 회사연혁 3. 매출, 거래액성장통계 4. 사업영역 5. 브랜드파워 회사소개 
                회사개요 KG이니시스는1998년설립하여2002년코스닥에상장된신뢰할수있는기업이며 230여명의전문가로구성되어경쟁력있는기업입니다. 시장점유율1위 최고의안전성을갖춘 시장점유율1위PG사 160,000가맹점 누적16만개가맹점이 이용하는서비스 4.8 Billion 연간4.8억건의결제 25조원의거래금액 통합간편결제서비스 통합간편결제서비스로 가맹점페이구축지원 PATENT 결제서비스및 기술특허최다보유 
                회사연혁 전자지불의새지평을열다 국내최초충전식전자화폐출시 한국모바일페이먼트서비스설립 전자금융을선도하다 2008 정보보호안전진단필증획득 업계최초IDC 이중화구축 계좌이체, 모바일결제원천서비스제공 INIB2B / INIP2P / OPA 오픈 IPTV T커머스구축 업계유일의오픈웹서비스제공 스마트폰 
                결제솔루션‘INIPAY Mobile’출시 전자지불시장을이끌어가다 업계최초코스닥시장등록 e-biz대상산업자원부상수상 이니시스윤리경영헌장수립 대한민국지불결제의No.1 업계최초간편결제, 간편결제커넥트시행 이니시스KG그룹편입 공공입찰B2B서비스제휴 PCI DSS인증획득 인터넷전문은행K뱅크지분취득 자회사러시아법인KG RUS 설립 가산센터신규오픈 온오프라인VAN 시스템구축및서비스개시 업계최초월거래액2조돌파 업계최다간편결제10종패키지제공 1998 – 2001 2006 – 2010 2002 – 2005 2011 – 2020 2021 지불결제시장에혁신을일으키다 GS페이출시 통합인증서비스오픈 선불전자지급수단발행업등록완료 PCI-DSS 8년연속인증획득 '렌탈페이' 구 독서비스출시 KG이니시스, 영업이익985억원역대최대실적갱신
                매출/ 거래액성장통계 2014 2015 2016 2017 2018 2019 2020 온라인쇼핑몰거래액 KG이니시스거래액 45.3 54.1 65.6 94.2 159.4 10.4 12.4 14.6 19.3 25 온라인쇼핑몰거래액출처: 통계청- 온라인쇼핑동향 (단위: 조원) 연간거래액 113.7 21.8 16만KG이니시스누적가맹점 30 제휴금융기관 22.6조원2019년연간거래금액 23.7조원2020년연간거래금액 5.2% 연간거래성장률, 신용카드10% 성장 5.5억건연간VAN 처리건수 거래성장 가맹점및제휴사 
                사업영역 
                사업영역 PG (Payment Gateway) 사업 전자결제 • PG업계No.1 PG • 전거래에스크로를통한안전거래제공 • 오픈웹결제제공 • PG업계최초스마트폰모바일결제서비스 • 2020년거래금액25조원기록 • App방식과Web방식모두지원 • 국내최고수준의보안기술도입 • Non Active X 지원 Global PGs • 각국가별현지화폐지불결제연동 해외가맹점들대상원화결제서비스제공 • 국내가맹점들의해외소비자대상 현지화폐지불서비스제공 • 마스터카드, global collect, 사이버소스, 알리페이등 글로벌선두Payment 기업과사업제휴 • 국내PG3사최초 알리페이, 위챗페이, 텐페이결제서비스제공 
                사업영역 VAN(Value Added Network) 사업 VAN • O2O(online & offline) 구분없이다양한서비스제공 • 안정적인VAN 서비스제공 • 자체솔루션으로SI 가맹점서비스제공 • OFF Line 가맹점과카드사데이터중계서비스 • 테이블오더(간편주문/결제) 서비스 • 우수한재무구조와자금력을바탕으로오차없는정 
                산서비스제공 
                사업영역 시너지(Synergy) 사업 B2B • 기업간전자상거래서비스e-마켓플레이스 운영(www.inib2b.com) • 신용보증기금, 기술보증기금, 신용보증재단, 14개은행연계심사를통해거래대금보증 • 기업맞춤형결제솔루션및금융상품개발 광고플랫폼 • PC결제창, 모바일결제창, 가맹점관리자사이트영역 에서 
                광고플랫폼제공 • B2C 영역월3,500만명결제고객, B2B 영역누적13만 가맹점광고노출 • 광고노출, 클릭수, CTR등통계레포트제공 
                디지털금융의Leading 기업 전자결제선도기업, (주)케이지이니시스는창의적인생각으로끊임없이변화하고있습니다. 전자지불(Payment Gateway)부분에서대한민국의대표브랜드로확고히하였고, 세계적인대표브랜드로자리매김하고자한발더도약합니다. 
                브랜드파워 Premium Service • 결제전문가그룹의체계적인서비스: 최장기간업계1위로PG 전문서비스제공 • 국내주요간편결제를결제창안에서제공하는통합간편결제서비스제공 최고의기술력 • 업계최다기술특허보유 • PCI DSS 인증획득: American Express, Discover, Master Card, Visa, JCB 社가공동
                으로책정한신용카드 업계글로벌보안기준을PG 업계최초로인증 • 금융거래안정성확보부분최우수선정 재무안정성 • 우수한경영성과: 우수한재무구조와풍부한자금력을바탕으로안정적인고객사정산서비스제공 • 경쟁사대비우수한단기지급능력: 경쟁사대비월등한유동성보유, 안정적인이익창출 
                3. 간편결제 1. INIPAY 전자결제 사업소개 INIPAY 오픈웹서비스 INIPAY 모바일서비스 INPAY ARS 서비스 통합간편결제서비스 WPAY (더블유페이) 6. 부가서비스 실시간번역서비스 카드본인확인서비스 통합인증서비스 모바일Biz쿠폰 필수부가패키지 Everyday Pay (매출선정산서비스) 모바일가맹점관 
                리자 계좌성명조회서비스 지급대행서비스 카카오톡Biz 메시지서비스 4. 광고플랫폼 B2C B2B 2. INI VAN / 테이블오더 INI VAN 서비스 테이블오더서비스 5. 이니렌탈서비스 
                INIPAY 전자결제 온라인상에서소비자와기업간결제를안전하고편리하게대행하는이니시스의전자결제서비스가맹점주문결제상품배송승인요청승인확인 1 4 2 3구매자카드계좌이체가상계좌핸드폰각종상품권다양한결제수단제공글로벌서비스(Alipay, Tenpay, Wechat)렌탈페이+
                INIPAY 오픈웹서비스 비IE 환경에서도편리한온라인결제를이용할수있도록오픈웹서비스를제공 윈도우에서브라우저종류에상관없이전체신용카드포함한모든결제수단을이용가능 ※ Window : 모든브라우저에서신용카드전체사용가능/ Mac, Linux : 모든브라우저에서신한카드하나SK카드사용가능 
                INIPAY 모바일서비스 스마트폰, 태블릿PC 등모바일환경에서소비자가편리하고안전하게상품또는 콘텐츠를구매할수있도록제공하는간편한모바일결제서비스 • PG업계최초스마트폰모바일결제서비스 • App 방식과Web 방식모두지원 • 연동편의성극대화 • 가상(스크린) 키보드를통한보안강화 • 온라인과동
                일하게전지불수단결제제공 
                INIPAY ARS 서비스 유/무선전화를이용하여주문을접수하고, 이후유/무선전화를통해신용카드인증및승인후결제하는서비스 여행, 카탈로그쇼핑및TV홈쇼핑, 교육비, 학원비, 공공지로관리업종에적합한결제입니다. SMS 주문인증번호제공방식 호전환제공방식 
                INI VAN(Value Added Network) SERVICE 카드사와가맹점간고도의통신망을구축하여승인중계, 카드전표매입, 청구대행등의데이터중계서비스를제공하는이니시스의서비스 VAN 단말기 KIOSK & KDS & DID SYSTEM 및SI 지원사업 OFF Line 가맹점과카드사 데이터중계서비스 테이블오더(간편주문/결제) 사업 POS(Point of sales) 사업 
                INI VAN 서비스내용 INI VAN Network를통한다양한결제수단별서비스제공 신용카드(체크카드) 서비스 국내(해외) 발생신용카드거래승인중계서비스제공, DDC, EDC, DSC, ESC, EDI 매입처리 현금IC거래서비스 현금IC카드를통한거래승인중계서비스, 결제승인취소서비스 멤버십(포인트) 서비스 통신사(SKT, KT, LG U+) 멤버십서비스 OK Cashbag 등각종포인트서비스 간편결제/해외카드서비스 삼성페이, 엘페이, 페이코및 은련, 위챗, 알리페이서비스 전자결제서비스(PG) 오프라인, APP, WEB상에서결제할수있는 통합결제솔루션제공 현금영수증 국세청시행현금영수증승인처리서비스제공 
                키오스크서비스 • 종교및비영리재단에카드/간편결제기부서비스제공 • 와이파이연결가능 업종과공간의제한이없는결제시스템 • 가맹점니즈에맞춘UX/UI 기획및결제솔루션제공 • 키오스크디자인, 이동형등다양한옵션선택가능 가맹점니즈에최적화된커스터마이징 • 현장피드백에기반한A/S로안정적서비스제공 • 원격지원을통한시스템업데이트로결제수단확장 • 24시간고객센터운영 지속적시스템관리및업데이트 산업의경계를넘어언제어디서나비대면결제가가능한키오스크서비스제공 
                테이블오더서비스 H/W 비용, 개발비, 사용료등의비용없이쉽고편리한스티커형테이블QR 스캔오더서비스제공 • 테이블QR스캔으로바로주문과결제 • 별도의APP 설치없이결제가능 쉽고편리한주문과결제 • 개발비無, 월사용료無 초기도입비용부담최소 • 국내1위PG 업체KG이니시스인프라활용 • 외국어서비스지원 안정적이고다양한서비스제공 
                통합간편결제서비스 KG이니시스와계약만으로TOP 7 간편결제서비스를이용가능한국내유일통합간편결제서비스로 최초1회인증설정으로간편결제서비스이용가능
                WPAY (더블유페이) 가맹점이가지고있는각브랜드이미지에맞게가맹점자체간편결제서비스인것처럼 커스터마이징이가능한간편결제서비스제공 • 가맹점자체브랜드의‘간편결제’를사용자에게제공(신용카드, 계좌이체제공, 2018.12 기준) • 부정거래탐지등FDS 시스템으로거래안전성을확보하였으며, 고객사의브랜드컨셉에맞는UI/UX 구현이가능 
                WPAY (더블유페이) - Standard 로고나버튼색상등의일부영역을변경하여쉽고빠르게개발이가능하고별도의비용이없는버전 적용예시 실제 적용사례
                WPAY (더블유페이) – Pro 결제요청화면을가맹점이원하는페이지에포함시키고특성에맞게기획/개발커스텀을할수있어 고객에게익숙한자체간편결제서비스구축효과를줄수있는버전 대표가맹점: 요기요, 인터파크, 도미노피자, 배달통, NS홈쇼핑, THEHANDSOME.COM, YES24, SROOK 
                이니렌탈[구독결제] 서비스 기존수수료그대로! 전자결제정산주기그대로! 자금유통리스크걱정없이! 최대60개월까지월분할결제가능한장기구독결제서비스 할부수수료 부담NO! 판매상품 그대로! 고액상품 매출UP! 정산주기 그대로! 높은수수료, 자금유통리스크없이 부담없이고가의상품Sale Point! 일
                시납부가부담되는일반상품을구매 + 제휴카드할인까지 가맹점 구매고객 이니렌탈상품구매예시(48개월할부기준) 원상품가격 1,000,000원 월납입금(48개월선택) 월25,000원 제휴카드사용시, 월5,000원납부 실납부금액 240,000원 할인최대구간 최대76% 할인효과 *제휴카드월최대2만원할인적용 
                일본결제서비스 일본진출을희망하는가맹점을위한TOTAL 결제솔루션 1회계약으로국내/국외PG서비스부터정산, 관리, 기술지원까지한번에! 한국가맹점 한국소비자 일본소비자 희망통화로 더빠르게정산 일본향EC 마케팅솔루션 지원 편리한 관리시스템 타사대비 저렴한수수료 일본지불수단 최대보유 하 
                나의시스템으로통합관리하여 추가연동/ 개발불필요 
                광고플랫폼서비스(B2C) KG이니시스결제창B2C 플랫폼에서광고노출제공 ※ 결제고객월3,500만명대상광고노출 INIPAY PC 결제창광고 INIPAY 모바일결제창광고
                광고플랫폼서비스(B2B) 가맹점관리자사이트등B2B 플랫폼에서광고노출제공 ※ 누적가맹점13만명대상광고노출 가맹점관리자페이지광고 
                실시간번역서비스 쇼핑몰실시간자동번역엔진탑재로, 구매자IP 기반또는국가버튼선택시원하는언어로실시간자동번역 1. 간단한스크립트삽입만으로접속한국가의IP를파악하여 국문텍스트를해당국가텍스트로자동으로번역 2. 국가버튼을삽입하여해당국가선택시실시간번역 이미지및동영상번역시에도가맹
                점관리자시스템을통하여 실시간에준하는번역서비스제공 실시간번역 준실시간번역 
                카드본인확인서비스 본인명의의휴대폰이없어도평소이용하는앱카드(신용및체크카드) 하나로본인인증이가능한서비스 • 신용및체크카드하나로본인인증가능(본인명의휴대폰이없어도가능) • 공인인증, 휴대폰인증, 아이핀인증을대체할수있는온라인주민등록번호대체서비스(CI/DI) 신규대체수단 기존주민등록번호대체수단 
                통합인증서비스 민간인증서를기반으로간편인증과전자서명을통합으로제공하는서비스 카카오, 네이버, 금융결제원, 패스, 토스, 페이코인증서를제공 대중적인인증서를활용하여 회원관리, 아이디/ 비밀번호찾기, 전자계약등에활용가능 1번의계약으로6개의인증서선택하여사용가능 별도프로그램설치/ 
                개발없이웹호출방식으로연동 간단한입력방식으로고객편의증대 기본료및유지보수비불필요 별도프로그램설치No! 간편인증& 전자서명통합제공 
                모바일Biz 쿠폰(모바일쿠폰증정마케팅서비스) 신규고객관리를위한프로모션혹은임직원리워드용으로모바일쿠폰을발송할수있는서비스 • 오직이니시스가맹점에게만브랜드별단독할인율적용된쿠폰판매 • 서비스신청및가입절차없이가맹점관리자페이지> 부가서비스> 모바일BiZ 쿠폰메뉴에서간편하게이용가능 쇼핑몰가입, 상품리뷰, SNS 이벤트참여고객을통한 바이럴마케팅으로사이트홍보및수익확보. 쇼핑몰가입, 후기이벤트 인스타태그(#,@) 이벤트 유튜브채널이벤트 
                필수부가패키지 많은가맹점들이필수로선택하는부가서비스를모아패키지를구성한서비스 에스크로 구매안전서비스로, 구매자의결제대금을예치하여배송완료후구매결정에따라 판매자에게결제대금을지급하는서비스 보증보험대체비 보증보험가입없이정산한도를증액하는서비스 원천적으로환불이불가한지불수단인가상계좌, 휴대폰결제의환불대행서비스 환불서비스 모두알림서비스 결제알림/ 정산입금알림/ 정산한도알림을결합한모두알림서비스 http://www.inicis.com/2018/0725/index.html 
                Everyday Pay (매출선정산서비스) 자금흐름에어려움을겪고있는가맹점을위해선지급정산해주는서비스 • 신용등급과관계없이PG정산대금을담보로익영업일에선지급정산 • 계약된정산주기와관계없이익영업일에지정된계좌로바로지급 신용등급과관계없이누구나! 원활한자금계획에도움! 
                모바일가맹점관리자 기존PC 환경에서접속하는가맹점관리자서비스를앱으로제공하여 언제어디서나거래/정산내역을확인및취소할수있으며, 앱PUSH 기능으로필요한알림실시간으로제공 https://iniweb.inicis.com/mobileCheck.jsp 
                계좌성명조회서비스 가맹점에서가맹점관리자의계좌성명일치조회URL로 계좌번호, 예금주명, 은행정보송신을통하여 현계좌번호의예금주가일치하는지에대한여부를수신 가맹점에서가맹점관리자의계좌주성명조회 URL로계좌번호, 은행정보송신을통해 현계좌주의성명을조회및수신 계좌주성명일치조회 계
                좌주성명조회 서비스연속성및안정성확보 서비스를듀얼운영하여연속성을확보하고계좌번호의예금주를사전에확인하여금융사고방지 서비스편의성제공 시중전체은행에대해송금(환불)에대한고객의은행계좌번호와성명을실시간으로확인 
                지급대행서비스 전자자금이체업무를수행하는오픈마켓사업자(부가통신사업자)를대신하여 이니시스에서거래금액의지급및자금이체를대행하는서비스 • 전자자금이체업무를수행하는오픈마켓사업자의전자금융업자등록비용및관리비용절약가능 • 서브몰(통신판매자) 계좌정보및지급데이터는가맹점(오픈마 
                켓사업자)에서관리하며, 거래금액의보유및지급등자금이체를이니시스에서대행 
                카카오톡Biz 메시지서비스 카카오톡을통해정보성, 광고성메시지를고객에게발송하는서비스 휴대폰번호를기반으로카카오톡친구추가없이 이용자에게정보성메시지발송 휴대폰번호를기반으로카카오톡친구로맺어진 이용자를대상으로광고성메시지발송 알림톡 친구톡 
                1. Partners 2. Contact Us 파트너소개 
                Partners 카드사 호스팅사 
                Partners 지불협력사 
                Partners 주요가맹점 
                Contact us https://www.inicis.com/ 서비스계약문의 02-3430-5858 ch@kggroup.co.kr 제휴및마케팅문의 inibiz@kggroup.co.kr Global PG 문의 02-3430-0984 gb@kggroup.co.kr 기술지원 02-3430-5960 ts@kggroup.co.kr 고객센터 1588-4954 
                감사합니다. 
                2021. v1.0 
                ⓒKG Inicis
                KG이니시스 
                회사소개서 
                Total Payment Service Provider
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                
                

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                - 너는 코드 구조 응답만을 반환해야 한다.
                """
        # prompt = f"""
        #         <|start_header_id|>system<|end_header_id|>
        #         - 너는 사이트의 섹션 구조를 정해주고, 그 안에 들어갈 내용을 작성해주는 AI 도우미야.
        #         - 입력된 데이터를 기준으로 단일 페이지를 갖는 랜딩사이트 콘텐츠를 생성해야 해.
        #         - 'children'의 컨텐츠 내용의 수는 너가 생각하기에 섹션에 알맞게 개수를 수정해서 생성해줘.
        #         - 섹션 '{section_name}'에 어울리는 내용을 생성해야 하며, 반드시 다음 규칙을 따라야 한다:
        #         1. assistant처럼 생성해야하고 형식을을 **절대** 벗어나면 안 된다.
        #         2. "h1, h2, h3, p" 태그만 사용해서 섹션의 콘텐츠를 구성해라.
        #         3. 섹션 안의 `children` 안의 컨텐츠 개수는 2~10개 사이에서 자유롭게 선택하되, 내용이 반복되지 않도록 다양하게 생성하라.
        #         4. 모든 텍스트 내용은 입력 데이터에 맞게 작성하고, 섹션의 목적과 흐름에 맞춰야 한다.
        #         5. 출력 결과는 JSON 형태만 허용된다. 코드는 **절대 생성하지 마라.**
        #         6. 오직 한글로만 작성하라.
                

        #         <|eot_id|><|start_header_id|>user<|end_header_id|>
        #         입력 데이터:
        #         {input_text}
        #         섹션:
        #         {section_name}
                

        #         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        #         - 너는 JSON 형태의 응답만을 반환해야 한다. 아래와 같은 형식의 순수 JSON만을 출력해야해.
        #         {{
        #             "children": [
        #                 {{
        #                     "type": "h1",
        #                     "text": "섹션의 주요 제목을 입력 데이터에 맞게 작성합니다."
        #                 }},
        #                 {{
        #                     "type": "p",
        #                     "text": "섹션의 내용을 소개하는 첫 번째 단락입니다. 핵심 메시지를 간결하고 명확하게 작성합니다."
        #                 }},

        #                 {{
        #                     "type": "h3",
        #                     "text": "중요 포인트를 정리하거나 강조합니다."
        #                 }},
        #                 {{
        #                     "type": "p",
        #                     "text": "설명 내용을 구체적으로 작성하되 중복되지 않도록 주의합니다."
        #                 }},
        #                 {{
        #                     "type": "p",
        #                     "text": "마무리 문장으로 섹션의 가치를 강조하고 독자의 참여를 유도합니다."
        #                 }}
        #             ]
        #         }}
        #         """
        return await self.send_request(model, prompt)
    
    #============================================================================
    # test code 칸 chunk
    async def send_pdf_chunk(self, model: str, total_chunks, current_chunk_number, chunk_content: str) -> str:
        """
        PDF 청크를 보내어 세션에 누적
        """
        prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            - 다음은 PDF 문서의 일부입니다. 이 내용을 기억하고, 이후의 질문에 참고해 주세요.
            - 문서는 총 {total_chunks}개의 청크로 나누어져 있습니다.
            - 현재 제공되는 청크는 {current_chunk_number}번째 청크입니다.
            - 각 청크는 고유한 식별자를 가지고 있으며, 필요 시 해당 청크를 참조할 수 있습니다.
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            - Chunk {current_chunk_number} of {total_chunks}:
            - {chunk_content}
        """
        return await self.send_request(model, prompt)
    #============================================================================
    
    async def landing_block_STD(self, model : str= "bllossom", input_text :str = "", section_name=""):
        prompt = f"""
            <|start_header_id|>system<|end_header_id|>
            - 당신은 AI 랜딩페이지 콘텐츠 작성 도우미입니다.
            - 입력된 데이터를 기반으로 랜딩페이지의 적합한 콘텐츠를 작성하세요.
            - 반드시 입력 데이터를 기반으로 작성하며, 추가적인 내용은 절대 생성하지 마세요.
            - 섹션에 이름에 해당하는 내용 구성들로 내용 생성하세요.
            - 콘텐츠를 JSON 형태로 작성하세요.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            입력 데이터:
            {input_text}
            
            섹션:
            {section_name}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            - 출력형식을 제외하고 다른 정보는 출력하지마세요.
            - 출력은 JSON형태로만 출력하세요.
            **출력 예시**:
            {{"h1" : "타이틀 내용",
            "h2" : (선택사항)"서브타이틀 내용",
            "h3" : 
            "본문" : "본문내용"}}
            """
                
        print(f"prompt length : {len(prompt)}")
        return await self.send_request(model, prompt)
    # async def landing_block_STD(self, model : str= "bllossom", input_text = "", section_name = "", section_num = ""):
        
    #     chunk = get_embedding_from_ollama(text=input_text)
        
    #     all_results = []
    #     for idx, chunk in enumerate(chunk):
    #         prompt = f"""
    #                 <|start_header_id|>system<|end_header_id|>
    #                 - AI 랜딩페이지 컨텐츠 생성 도우미
    #                 - 한글로 답변

    #                 <|eot_id|><|start_header_id|>user<|end_header_id|>
    #                 입력 데이터:
    #                 {chunk}
                    
    #                 섹션 이름:
    #                 {section_name}

    #                 <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #                 - 입력 데이터 기반 컨텐츠 생성
    #                 - {section_name}에 해당하는 컨텐츠 생성
    #                 - 보고서 형식
    #                 - 코드 생성 금지
    #                 - 한글 작성
    #         """
    #         print(f"prompt length : {len(prompt)}")
    #         result = await self.send_request(model, prompt)
    #         all_results.append(result)
    #     return " ".join(all_results)
    
    
    
    # async def LLM_summary(self, input_text: str = "", model="bllossom"):
    #     chunk = get_embedding_from_ollama(text=input_text)
        
    #     all_results = []
    #     for idx, chunk in enumerate(chunk):
    #         prompt = f"""
    #                 <|start_header_id|>system<|end_header_id|>
    #                 당신은 고급 텍스트 요약 전문 AI 어시스턴트입니다. 다음 핵심 원칙을 엄격히 준수하세요:

    #                 요약 목표:
    #                 - 원본 텍스트의 핵심 메시지와 본질적 의미 정확하게 포착
    #                 - 불필요한 세부사항은 제외하고 핵심 내용만 추출
    #                 - 간결하고 명확한 언어로 요약
    #                 - 원문의 맥락과 뉘앙스 최대한 보존

    #                 요약 가이드라인:
    #                 - 입력된 텍스트의 주요 아이디어 식별
    #                 - 중요한 논점과 결론 강조
    #                 - 원문의 길이에 비례하여 적절한 길이로 요약
    #                 - 불필요한 반복이나 부수적인 정보 제거

    #                 요약 기법:
    #                 - 핵심 문장 추출 및 재구성
    #                 - 중요한 키워드와 주제 포함
    #                 - 논리적이고 일관된 흐름 유지

    #                 <|eot_id|><|start_header_id|>user<|end_header_id|>
    #                 입력 데이터:
    #                 {chunk}

    #                 <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    #                 요약 작성 시 다음 세부 지침 준수:

    #                 1. 입력 데이터의 본질적 의미 정확히 파악
    #                 2. 원문의 핵심 메시지를 20-30% 길이로 압축
    #                 3. 명확하고 간결한 문장 구조 사용
    #                 4. 정보의 손실 최소화
    #                 5. 읽기 쉽고 이해하기 쉬운 요약문 작성

    #                 주의사항:
    #                 - 개인적 해석이나 추가 의견 배제
    #                 - 원문의 사실관계 왜곡 금지
    #                 - 중요한 맥락이나 뉘앙스 보존
    #                 - 문법적 정확성과 가독성 확보

    #                 출력 형식:
    #                 - 명확한 주제 또는 제목
    #                 - 간결한 단락 구조
    #                 - 핵심 포인트 나열
    #                 - 논리적 흐름 유지
    #         """
            
    #         print(f"LLM_summary Len :  {len(prompt)}")
    #     return await self.send_request(model, prompt)
    
    async def LLM_content_fill(self, input_text: str = "", model="bllossom", summary = ""):
        
        prompt = f"""
                <|start_header_id|>system<|end_header_id|>
                당신은 전문적이고 매력적인 랜딩페이지 컨텐츠를 생성하는 고급 AI 어시스턴트입니다. 다음 지침을 철저히 따르세요:

                **주요 목표:**
                - 제공된 입력 데이터와 요약 데이터를 기반으로 컨텐츠를 작성하세요.
                - 작성된 컨텐츠는 타겟 고객의 관심을 끌 수 있도록 매력적이어야 합니다.

                **작성 지침:**
                - 모든 응답은 반드시 한글로 작성하세요.
                - 각 섹션의 형식을 유지하며 내용을 작성하세요.

                <|eot_id|><|start_header_id|>user<|end_header_id|>
                입력 데이터:
                {input_text}

                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                - 요약 데이터를 바탕으로 입력 데이터에서 필요한 내용을 도출하여 작성합니다.
                - 아래와 같은 형식으로 컨텐츠를 구성합니다:

                1. 입력 데이터의 모든 중요 정보 포함
                2. 최종 컨텐츠는 명확하고, 설득력 있으며, 전문성을 갖추도록 작성

                주의사항:
                - 문법적 오류와 부자연스러운 표현 주의
        """
        print(f"LLM_content_fill Len :  {len(prompt)}")
        return await self.send_request(model, prompt)
    
    
    async def LLM_land_page_content_Gen(self):
        """
        랜딩 페이지 섹션을 생성하고 JSON 구조로 반환합니다.
        """
        # 섹션 리스트
        section_options = ["Introduce", "Solution", "Features", "Social", 
                        "CTA", "Pricing", "About Us", "Team","blog"]

        # 섹션 수 결정 (6 ~ 9개)
        section_cnt = random.randint(6, 9)
        print(f"Selected section count: {section_cnt}")

        # 1번과 2번 섹션은 고정
        section_dict = {
            1: "Header",
            2: "Hero"
        }

        # 마지막 섹션은 Footer로 고정
        section_dict[section_cnt] = "Footer"

        # 마지막 이전 섹션에 FAQ, Map, Youtube 중 하나 배정
        minus_one_sections = ["FAQ", "Map", "Youtube", "Contact", "Support"]
        section_dict[section_cnt - 1] = random.choice(minus_one_sections)

        # 나머지 섹션을 랜덤하게 채움
        filled_indices = {1, 2, section_cnt - 1, section_cnt}
        for i in range(3, section_cnt):
            if i not in filled_indices:
                section_dict[i] = random.choice(section_options)

        # 섹션 번호 순서대로 정렬
        sorted_section_dict = dict(sorted(section_dict.items()))

        # JSON 문자열 반환
        result_json = json.dumps(sorted_section_dict, indent=4)
        print("Generated Landing Page Structure:")
        print(result_json)
        return result_json
    
    # async def LLM_land_page_content_Gen(self, input_text: str = "", model="bllossom", structure_limit=True):
    #     cnt = random.randint(6, 9)
    #     print(f"section count: {cnt}")
    #     try:
    #         # 비동기 함수 호출
    #         section_data = await self.landing_page_STD(model=model, input_text=input_text, section_cnt=cnt)
    #         print(f"Raw section_data: {section_data}")

    #         # JSON 형식인지 검증
    #         json_match = re.search(r"\{.*\}", section_data, re.DOTALL)
    #         json_str = json_match.group(0)
    #         # JSON 문자열을 Python 딕셔너리로 변환
    #         section_dict = json.loads(json_str)
    #         print(f"type: {type(section_dict)},\n section_dict: {section_dict}")

    #         if structure_limit:
    #             # 키를 정수로 변환
    #             section_dict = {int(k): v for k, v in section_dict.items()}

    #             # 정렬된 키 리스트
    #             keys = sorted(section_dict.keys())
    #             footer_key = keys[-1]          # 마지막 키
    #             previous_key = keys[-2]        # 마지막 키 바로 이전

    #             # 1번 섹션 검증 및 수정
    #             if section_dict[1] != "Header":
    #                 section_dict[1] = "Header"

    #             # 2번 섹션 검증 및 수정
    #             if section_dict[2] != "Hero":
    #                 section_dict[2] = "Hero"

    #             # Footer 이전 섹션 검증 및 수정
    #             minus_one = ["FAQ", "Map", "Youtube"]
    #             minus_one_num = random.randint(0, 2)

    #             if section_dict[previous_key] not in minus_one:
    #                 section_dict[previous_key] = minus_one[minus_one_num]

    #             # Footer 섹션 검증 및 수정
    #             if section_dict[footer_key] != "Footer":
    #                 section_dict[footer_key] = "Footer"

    #         print(f"type: {type(section_dict)}, len section data: {len(section_dict)}")

    #         # 최종 수정된 딕셔너리를 JSON 문자열로 변환하여 반환
    #         return json.dumps(section_dict)
    #     except json.JSONDecodeError as e:
    #         print(f"JSON decoding error: {e}")
    #         print(f"Raw response: {section_data}")
            
    #         raise HTTPException(status_code=500, detail="Invalid JSON response from LLM")
    #     except Exception as e:
    #         print(f"Error generating landing page sections: {e}")
    #         raise HTTPException(status_code=500, detail="Failed to generate landing page sections.")


    async def LLM_land_block_content_Gen(self, input_text : str = "", model = "bllossom", section_name = "", section_num = "1", summary=""):
        
        try:
            # 비동기 함수 호출 시 await 사용
            contents_data = await self.landing_block_STD(model=model, input_text=input_text, section_name = section_name, section_num = section_num)
            print(f"contents_data summary before: {contents_data}")
            
            # 최종 수정된 딕셔너리를 JSON 문자열로 변환하여 반환
            contents_data = await self.LLM_content_fill(model=model, input_text=contents_data, summary=summary)
            print(f"contents_data summary after: {contents_data}")
            
            return contents_data
        except Exception as e:
            print(f"Error generating landing page sections: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate landing page sections.")