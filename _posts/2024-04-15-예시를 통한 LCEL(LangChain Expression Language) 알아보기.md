---
layout: single
title:  "예시를 통한 LCEL(LangChain Expression Language) 알아보기"
---

# LangChain의 LCEL을 적용해 Chain 생성해보기



## LCEL이란?

<img src="/Users/lymansong/Documents/GitHub/lymanstudio.github.io/images/2024-04-15-예시를 통한 LCEL(LangChain Expression Language) 알아보기/lcel_image_1.png" alt="lcel_image_1" style="zoom:20%;" />

LCEL은 LangChain Expression Language의 약자로 LangChain에서 개발한 언어라기보단 간단한 문법이다. Language Model을 활용하는 어플리케이션을 개발할 때 프로픔트, 모델, 출력 파서 등 다양한 컴포넌트들을 따로 개발하여 이어준 객체를 Chain이라고 하는데 이 체인 생성을 간단하게 구성하게끔 만든 것이다.

LangChain 공식 홈페이지에서 [LCEL을 소개하는 페이지](https://python.langchain.com/docs/expression_language/)의 첫 문단은 다음과 같다.

> LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to **support putting prototypes in production, with no code changes**, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production).

LM을 사용한 어플리케이션 개발 중 각 컴포넌트들을 블록 단위로 구성해 이리저리 조합하면서 기능과 성능을 테스트할 때 LCEL을 사용하면 복잡한 과정을 거치지 않고 간편하게 사용할 수 있다.



## 예제 실습

#### 1. 필요한 API 키를 환경 변수에 등록

LLM 모델로 OpenAI의 ChatGPT를 사용할 것으므로 이번 실습에 필요한 API 키는 OpenAI API 키 하나이다. 실습하는 사람이 사용하는 LLM의 종류에 따라 다른 API key가 필요하다.


```python
from dotenv import load_dotenv
load_dotenv(dotenv_path= "/Users/yourname/yourdirectory/.env")
```

\>\> 출력 (잘 적용됐으면 True, 안됐으면 False)


    True



#### 2. 관련 라이브러리 import 및 모델 인스턴스 생성

관련 라이브러리는 모두 langchain 라이브러리에 있다. LLM 모델은 gpt 3.5를 사용했다.


```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model = 'gpt-3.5-turbo', max_tokens=2048)
```



#### 3. 프롬프트 생성

프롬프트는 유저에게 두개의 동의어를 입력받고 두 단어 사이의 차이점 몇가지를 출력받는 형식으로 설정한다.

우선 입력받은 두 단어를 타이틀로 설정하고 다음으로 차이점들을 bullet points로 구분지어 설명해달라고 요청했다.


```python
## 템플릿 설정
template = """
What are the differences between {word1} and {word2}? 
The output should have title in the first section contains the two words.
And tell me some differences partitioned by bullet points."
"""

## 설정한 템플릿을 이용해 프롬프트 생성
prompt = PromptTemplate(
    template= template,
    input_variables=['word1', 'word2']
)
```



#### 4. LCEL을 사용한 Chain 생성

- or 연산자 character(`|`)를 사용해여 간단하게 컴포넌트들을 이어줄 수 있다.
- 한 컴포넌트의 출력은 이어지는 다음 컴포넌트의 입력으로 들어간다.
- 현재 예시에선 `프롬프트` => `LM모델` => `출력 파서`로 이어진다.


```python
chain = prompt | model | StrOutputParser()
```

위 과정은 LLMChain을 임포트해서 구성하는 다음 코드와 동일하다.


```python
from langchain.chains import LLMChain
llmchain = LLMChain(llm = model, prompt = prompt)
```



#### 5. 실행

- LCEL 체인: invoke 함수를 썼으며 parser까지 붙여줬기에 바로 스트링으로 출력한다.


```python
result = chain.invoke({"word1": "triad", "word2": "triplet"})
print(result)
```

\>\> 출력

    Triad vs Triplet
    
    - A triad is a group of three notes played together, typically forming a chord, while a triplet is a rhythmic grouping of three notes played in the space of two.
    - Triads are commonly used in harmony and chord progressions, while triplets are used to subdivide beats in music.
    - Triads are more related to harmony and chords in music theory, while triplets are related to rhythm and timing.
    - Triads are often notated using chord symbols or stacked notes, while triplets are notated with a "3" above or below the notes to indicate the rhythmic grouping.

- batch로 여러개 입력을 연달아 호출하는 것도 가능하다.

```python
results = chain.batch(inputs = [
        {'word1':"triad", 'word2': "triplet"},
        {'word1':"state", 'word2': "nation"},
        {'word1':"conquer", 'word2': "dominate"},
    ]  
    , config = {"max_concurrency" : 2}
)

for result in results:
    print(result)
    print("===========================")
```

\>\> 출력

    Triad vs. Triplet
    
    - A triad typically refers to a group of three related elements or individuals, whereas a triplet specifically refers to a group of three siblings born at the same time.
    - In music theory, a triad is a chord consisting of three notes played simultaneously, while a triplet is a rhythmic division of three notes played in the time of two of the same value.
    - Triads can have various meanings in different contexts such as in sociology, psychology, or music, while triplets are mainly used in the context of siblings or music notation.
    - Triads are commonly used in visual design, where three elements are arranged in a balanced way, while triplets are more focused on the relationship between three individuals born at the same time.
    ===========================
    State vs Nation:
    
    - Political entity vs Cultural entity
    - Defined borders vs Shared identity
    - Government structure vs Common history and language
    - Sovereignty vs Unity and solidarity
    - Legal system vs Common values and traditions
    ===========================
    Conquer vs Dominate
    
    - Conquer implies gaining control or victory over a place or people through force or military means.
    - Dominate suggests having power or influence over others through strength or superiority.
    - Conquer is often a one-time event, while dominate implies ongoing control or influence.
    - Conquer can involve physical occupation of territory, while dominate can be achieved through psychological or social means.
    - Conquer may involve defeating an enemy in battle, while dominate can involve outperforming or outmaneuvering competitors.
    ===========================

- LLMChain 객체: run() 메서드로 호출하며 템플릿에서 넣어줬던 변수들은 함수의 인자로 역할을 한다.
  - 이때 설정한 변수들을 모두 넣어주지 않으면 에러가 난다.  

```python
try:
    print(llmchain.run(word1 = 'triad', word2 = 'triplet'))
except:
    print("somethings wrong.")

print("===========================")
try:
    llmchain.run(word1 = 'state')
except:
    print("somethings wrong.")
```
\>\> 출력

    Triad vs Triplet
    
    - A triad is a group of three notes, often forming a chord, while a triplet is a rhythmic grouping of three notes played in the space of two.
    - Triads are used in harmony and chord progressions, while triplets are used in rhythmic patterns and phrasing.
    - Triads are typically written as stacked notes on sheet music, while triplets are indicated by a triplet marking above the notes.
    - Triads are commonly found in Western music theory, while triplets are used in various musical genres across the world.
    - Triads are essential in building chords and harmonies, while triplets add variety and complexity to rhythm and timing.
    ===========================
    somethings wrong.





#### 참고 링크

---

- [테디노트 - <랭체인LangChain 노트> - CH01-03](https://wikidocs.net/233344)
