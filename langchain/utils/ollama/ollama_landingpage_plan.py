from typing import List, Annotated, Optional
from pydantic import BaseModel, Field, constr #, validator, root_validator
import regex

class WebsitePlanException(Exception):
    """웹사이트 기획 관련 커스텀 예외"""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


# Custom type definitions with Annotated for better type hints
KeywordStr = Annotated[str, constr(max_length=20)]
ContentStr = Annotated[str, constr(max_length=200)]
TitleStr = Annotated[str, constr(max_length=100)]
PurposeStr = Annotated[str, constr(max_length=500)]
# VersionStr = Annotated[str, constr(regex=r"^\d+\.\d+\.\d+$")]


class WebsitePlan(BaseModel):
    """웹사이트 기획 정보를 담는 메인 모델"""

    website_name: TitleStr = Field(
        description="웹사이트의 공식 명칭",
        example="커리어 성장 플랫폼 'GrowthHub'",
    )

    keywords: List[KeywordStr] = Field(
        max_items=10,
        min_items=1,
        description="웹사이트를 대표하는 검색 키워드 (최대 10개, 각 20자 제한)",
        example=["커리어성장", "온라인학습", "실무교육", "멘토링"],
    )

    purpose: PurposeStr = Field(
        description="웹사이트의 제작 목적과 주요 미션",
        example="직장인들의 지속적인 성장과 경력 개발을 지원하는 온라인 학습 플랫폼 구축",
    )

    target_audiences: List[ContentStr] = Field(
        max_items=5,
        min_items=1,
        description="주요 타겟 사용자 그룹 정보 (최대 5개, 각 200자 제한)",
        example=[
            "25-35세 직장인으로, 자기계발과 커리어 성장에 관심이 많으며 온라인 학습에 익숙한 밀레니얼 세대. 특히 IT, 디자인 분야 종사자들이 주요 대상",
            "35-45세의 중간관리자급 전문직 종사자로, 팀 관리와 리더십 향상에 관심이 많으며, 효율적인 업무 프로세스 개선을 추구하는 직장인",
        ],
    )

    core_values: List[ContentStr] = Field(
        max_items=5,
        min_items=1,
        description="웹사이트가 추구하는 핵심 가치 (최대 5개, 각 200자 제한)",
        example=[
            "지속가능한 성장: 사용자가 자신의 페이스에 맞춰 꾸준히 학습하고 성장할 수 있도록 맞춤형 학습 경로와 피드백을 제공���여 장기적인 발전을 지원",
            "실무 중심 교육: 현업에서 바로 적용 가능한 실전 프로젝트와 케이스 스터디를 통해 실질적인 업무 역량 향상을 도모",
        ],
    )

    main_services: List[ContentStr] = Field(
        max_items=10,
        min_items=1,
        description="웹사이트에서 제공하는 주요 서비스 목록 (최대 10개, 각 200자 제한)",
        example=[
            "맞춤형 학습 로드맵: 사용자의 현재 수준과 목표를 분석하여 개인화된 학습 경로를 제시하고, AI 기반 추천 시스템을 통해 최적화된 콘텐츠를 제공",
            "실시간 온라인 멘토링: 현업 전문가들과의 1:1 화상 멘토링 세션을 통해 실무 경험과 노하우를 공유하고, 즉각적인 피드백과 조언을 받을 수 있는 서비스",
        ],
    )

    additional_functions: Optional[List[ContentStr]] = Field(
        default=[],
        max_items=15,
        min_items=0,
        description="웹사이트의 부가 기능 목록 (최대 15개, 각 200자 제한)",
        example=[
            "학습 진도 관리 대시보드: 개인별 학습 진행 상황, 성취도, 참여율 등을 시각화하여 제공하며, 목표 달성률과 개선점을 한눈에 파악할 수 있는 분석 도구",
            "스터디 그룹 매칭 시스템: 관심사와 학습 목표가 비슷한 사용자들을 자동으로 매칭하여 스터디 그룹을 형성하고, 온라인 협업 도구를 제공",
        ],
    )

    expected_effects: List[ContentStr] = Field(
        max_items=10,
        min_items=1,
        description="웹사이트 구축을 통해 기대되는 효과 (최대 10개, 각 200자 제한)",
        example=[
            "사용자의 실무 역량 강화: 현업 기반의 실전 프로젝트와 전문가 멘토링을 통해 실질적인 업무 수행 능력이 향상되며, 취업과 승진에 필요한 핵심 역량 확보",
            "효율적인 학습 경험 제공: AI 기반 맞춤형 학습 시스템으로 불필요한 학습 시간을 줄이고, 개인의 성장 속도에 맞는 최적화된 학습 경로 제시",
        ],
    )

    # version: VersionStr = Field(
    #     default="1.0.0",
    #     description="기획서 버전 (Semantic Versioning)",
    # )

    #@validator("keywords")
    def validate_keywords(cls, v):
        if not v:
            raise WebsitePlanException(
                "keywords", "최소 1개 이상의 키워드가 필요합니다"
            )
        if len(set(v)) != len(v):
            raise WebsitePlanException("keywords", "중복된 키워드가 있습니다")
        return v

    #@validator("target_audiences", "core_values", "main_services", "expected_effects")
    def validate_unique_items(cls, v, field):
        if len(set(v)) != len(v):
            raise WebsitePlanException(field.name, "중복된 항목이 있습니다")
        return v

    #@root_validator
    def validate_dependencies(cls, values):
        additional_functions = values.get("additional_functions", [])
        main_services = values.get("main_services", [])

        if len(additional_functions) > 0 and not main_services:
            raise WebsitePlanException(
                "additional_functions",
                "부가 기능은 주요 서비스가 정의된 경우에만 추가할 수 있습니다",
            )
        return values

    class Config:
        schema_extra = {
            "example": {
                "website_name": "커리어 성장 플랫폼 'GrowthHub'",
                "keywords": ["커리어성장", "온라인학습", "실무교육", "멘토링"],
                "purpose": "직장인들의 지속적인 성장과 경력 개발을 지원하는 온라인 학습 플랫폼 구축",
                "target_audiences": ["25-35세 직장인...", "35-45세의 중간관리자급..."],
                "core_values": ["지속가능한 성장...", "실무 중심 교육..."],
                "main_services": ["맞춤형 학습 로드맵...", "실시간 온라인 멘토링..."],
                "additional_functions": [
                    "학습 진도 관리 대시보드...",
                    "스터디 그룹 매칭 시스템...",
                ],
                "expected_effects": [
                    "사용자의 실무 역량 강화...",
                    "효율적인 학습 경험 제공...",
                ],
                # "version": "1.0.0",
            }
        }
        allow_mutation = False  # 불변성 보장
        validate_assignment = True  # 할당 시에도 유효성 검증
        anystr_strip_whitespace = True  # 문자열 양쪽 공백 제거