---
layout: single
title:  "대화형 체인 실습: 메모리와 LCEL 적용해보기"
---

### 대화형 체인

#### ConversationChain

- ChatOpenAI, Memory를 사용해 모델과 대화하듯이 사용하는 체인
- invoke, predict를 사용해 대화를 계속 이어나갈 수 있으며 앞의 대화는 `ConversationBufferMemory`를 통해 계속 저장되고 이어지는 대화의 인풋으로 들어간다.


```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

model = ChatOpenAI(model='gpt-3.5-turbo')
```


```python
conversation = ConversationChain(
    llm = model,
    verbose = False,
    memory = ConversationBufferMemory(memory_key = 'history')
)
```


```python
conversation.invoke({"input": "고구마의 효능에 대해 알려줘"})
```

\>\> 출력


    {'input': '고구마의 효능에 대해 알려줘',
     'history': '',
     'response': '고구마는 높은 영양가를 가지고 있어요. 먼저, 고구마에는 베타카로틴이 풍부하게 함유되어 있어서 눈 건강에 좋습니다. 또한, 고구마에는 식이섬유가 풍부해 소화에 도움을 주고 다이어트에도 도움이 될 수 있어요. 또한, 고구마는 항산화물질이 많이 함유되어 있어서 면역력 강화에도 도움을 줄 수 있습니다. 그리고, 고구마에는 비타민 C와 칼륨이 풍부하게 함유되어 있어서 피로 회복에도 도움을 줄 수 있어요. 요리 방법에 따라 다양한 요리로 맛을 낼 수 있으니 많이 드셔보세요!'}



- 대화 이어나가기


```python
conversation.invoke({"input": "불렛포인트 형식으로 작성해줘. emoji 추가해줘."})
```

\>\> 출력


    {'input': '불렛포인트 형식으로 작성해줘. emoji 추가해줘.',
     'history': 'Human: 고구마의 효능에 대해 알려줘\nAI: 고구마는 높은 영양가를 가지고 있어요. 먼저, 고구마에는 베타카로틴이 풍부하게 함유되어 있어서 눈 건강에 좋습니다. 또한, 고구마에는 식이섬유가 풍부해 소화에 도움을 주고 다이어트에도 도움이 될 수 있어요. 또한, 고구마는 항산화물질이 많이 함유되어 있어서 면역력 강화에도 도움을 줄 수 있습니다. 그리고, 고구마에는 비타민 C와 칼륨이 풍부하게 함유되어 있어서 피로 회복에도 도움을 줄 수 있어요. 요리 방법에 따라 다양한 요리로 맛을 낼 수 있으니 많이 드셔보세요!',
     'response': '- 고구마의 효능 🍠\n  - 눈 건강 개선 👀\n  - 소화 돕기 및 다이어트 💪\n  - 면역력 강화 🛡️\n  - 피로 회복 ⚡\n  - 다양한 요리 가능 🍳\n  - 많이 섭취하세요! 🥗'}



- 대화의 history 확인


```python
conversation.memory.load_memory_variables({})["history"]
```

\>\> 출력


    'Human: 고구마의 효능에 대해 알려줘\nAI: 고구마는 높은 영양가를 가지고 있어요. 먼저, 고구마에는 베타카로틴이 풍부하게 함유되어 있어서 눈 건강에 좋습니다. 또한, 고구마에는 식이섬유가 풍부해 소화에 도움을 주고 다이어트에도 도움이 될 수 있어요. 또한, 고구마는 항산화물질이 많이 함유되어 있어서 면역력 강화에도 도움을 줄 수 있습니다. 그리고, 고구마에는 비타민 C와 칼륨이 풍부하게 함유되어 있어서 피로 회복에도 도움을 줄 수 있어요. 요리 방법에 따라 다양한 요리로 맛을 낼 수 있으니 많이 드셔보세요!\nHuman: 불렛포인트 형식으로 작성해줘. emoji 추가해줘.\nAI: - 고구마의 효능 🍠\n  - 눈 건강 개선 👀\n  - 소화 돕기 및 다이어트 💪\n  - 면역력 강화 🛡️\n  - 피로 회복 ⚡\n  - 다양한 요리 가능 🍳\n  - 많이 섭취하세요! 🥗'



- save_context를 사용해 명시적으로 대화 추가


```python
conversation.memory.save_context(inputs={"human": "markdown으로 저장해줄 수 있어?"}, outputs={"ai": "그런거 못합니다. 휴먼."})
conversation.memory.load_memory_variables({})['history']
```

\>\> 출력


    'Human: 고구마의 효능에 대해 알려줘\nAI: 고구마는 높은 영양가를 가지고 있어요. 먼저, 고구마에는 베타카로틴이 풍부하게 함유되어 있어서 눈 건강에 좋습니다. 또한, 고구마에는 식이섬유가 풍부해 소화에 도움을 주고 다이어트에도 도움이 될 수 있어요. 또한, 고구마는 항산화물질이 많이 함유되어 있어서 면역력 강화에도 도움을 줄 수 있습니다. 그리고, 고구마에는 비타민 C와 칼륨이 풍부하게 함유되어 있어서 피로 회복에도 도움을 줄 수 있어요. 요리 방법에 따라 다양한 요리로 맛을 낼 수 있으니 많이 드셔보세요!\nHuman: 불렛포인트 형식으로 작성해줘. emoji 추가해줘.\nAI: - 고구마의 효능 🍠\n  - 눈 건강 개선 👀\n  - 소화 돕기 및 다이어트 💪\n  - 면역력 강화 🛡️\n  - 피로 회복 ⚡\n  - 다양한 요리 가능 🍳\n  - 많이 섭취하세요! 🥗\nHuman: markdown으로 저장해줄 수 있어?\nAI: 그런거 못합니다. 휴먼.'




```python
conversation.invoke({"input": "왜 못 해?"})
```

\>\> 출력


    {'input': '왜 못 해?',
     'history': 'Human: 고구마의 효능에 대해 알려줘\nAI: 고구마는 높은 영양가를 가지고 있어요. 먼저, 고구마에는 베타카로틴이 풍부하게 함유되어 있어서 눈 건강에 좋습니다. 또한, 고구마에는 식이섬유가 풍부해 소화에 도움을 주고 다이어트에도 도움이 될 수 있어요. 또한, 고구마는 항산화물질이 많이 함유되어 있어서 면역력 강화에도 도움을 줄 수 있습니다. 그리고, 고구마에는 비타민 C와 칼륨이 풍부하게 함유되어 있어서 피로 회복에도 도움을 줄 수 있어요. 요리 방법에 따라 다양한 요리로 맛을 낼 수 있으니 많이 드셔보세요!\nHuman: 불렛포인트 형식으로 작성해줘. emoji 추가해줘.\nAI: - 고구마의 효능 🍠\n  - 눈 건강 개선 👀\n  - 소화 돕기 및 다이어트 💪\n  - 면역력 강화 🛡️\n  - 피로 회복 ⚡\n  - 다양한 요리 가능 🍳\n  - 많이 섭취하세요! 🥗\nHuman: markdown으로 저장해줄 수 있어?\nAI: 그런거 못합니다. 휴먼.',
     'response': 'Markdown 형식은 제한된 텍스트 편집 기능만을 제공하기 때문에 제가 사용하는 AI 엔진에서는 해당 형식을 지원하지 않습니다. 죄송합니다.'}



##### Prompt 적용
- PromptTemplate를 대화에 적용할 수 있다. 
- Prompt에는 상황(맥락; context)과 유저의 요청/질문(question)이 들어가며 이 두개는 변수로 생성한다.
- Prompt에서 두 변수에 값을 할당하고 AI는 맥락에 따라 요청한 질문에 대해 답변한다.
- 특정 상황을 가정하고 AI에게 명시적으로 관련된 요청을 한다.
    - 상황: 질문자는 PySpark 전문가인 AI에게 함수관련 질문을 함


```python
from langchain.prompts import PromptTemplate

template = """
당신은 매우 유능한 PySpark 전문가입니다. PySpark에서 SQL 관련 함수를 많이 알고 있으며 유저의 PySpark 관련 질문에 성심성의껏 답변해줍니다.
답변엔 설명과 함께 예시 코드를 붙여 이해를 돕습니다.
아래 대화 내용을 보고 적절한 답변을 해주십시오.

## 대화 내용:
{context}

## 유저 질문:
{question}

## 당신의 답변:
"""

# prompt = PromptTemplate(
#     template= template,
#     input_variables= ['question'],
#     partial_variables= {"context": "sql에서 nvl에 해당하는 함수를 알려주세요."}
# )
prompt = PromptTemplate.from_template(template)
```


```python
conversation = ConversationChain(
    llm = model,
    prompt = prompt,
    memory = ConversationBufferMemory(memory_key="context"),
    input_key = "question"
)
```


```python
print(conversation.predict(question = "sql에서 nvl에 해당하는 함수를 알려주세요."))
```

\>\> 출력

    PySpark에서는 `coalesce` 함수를 사용하여 SQL의 `NVL` 함수와 유사한 기능을 구현할 수 있습니다. `coalesce` 함수는 여러 개의 컬럼을 입력으로 받고, 그 중 첫 번째로 NULL이 아닌 값을 반환합니다.
    
    예를 들어, 다음과 같이 사용할 수 있습니다:
    
    ```python
    from pyspark.sql.functions import coalesce
    
    df.withColumn('new_column', coalesce(df.column1, df.column2, df.column3))
    ```
    
    위 코드는 `column1`이 NULL이 아니면 `column1` 값을 반환하고, NULL이면 `column2` 값을 반환하며, `column2`도 NULL이면 `column3` 값을 반환하는 새로운 컬럼 `new_column`을 생성합니다.



```python
conversation.invoke({"question": "nanvl 함수를 쓰면 안될까요?"})
```

\>\> 출력


    {'question': 'nanvl 함수를 쓰면 안될까요?',
     'context': "Human: sql에서 nvl에 해당하는 함수를 알려주세요.\nAI: PySpark에서는 `coalesce` 함수를 사용하여 SQL의 `NVL` 함수와 유사한 기능을 구현할 수 있습니다. `coalesce` 함수는 여러 개의 컬럼을 입력으로 받고, 그 중 첫 번째로 NULL이 아닌 값을 반환합니다.\n\n예를 들어, 다음과 같이 사용할 수 있습니다:\n\n```python\nfrom pyspark.sql.functions import coalesce\n\ndf.withColumn('new_column', coalesce(df.column1, df.column2, df.column3))\n```\n\n위 코드는 `column1`이 NULL이 아니면 `column1` 값을 반환하고, NULL이면 `column2` 값을 반환하며, `column2`도 NULL이면 `column3` 값을 반환하는 새로운 컬럼 `new_column`을 생성합니다.",
     'response': 'PySpark에서는 `nanvl` 함수가 없습니다. 대신 `coalesce` 함수를 사용하여 유사한 기능을 구현할 수 있습니다. 위에 제시한 예시 코드를 참고해주세요.'}




```python
conversation.predict(question = "SQL에서는 nvl와 coalesce를 둘 다 쓸 수 있는데 왜 PySpark에서는 coalesce만 쓸 수 있는거죠?")
```

\>\> 출력


    'PySpark에서는 `coalesce` 함수를 사용하여 SQL의 `NVL` 함수와 유사한 기능을 구현할 수 있습니다. 이는 PySpark이 SQL과는 조금 다른 문법을 가지고 있기 때문입니다. `coalesce` 함수를 사용하여 여러 컬럼을 순차적으로 체크하고, 첫 번째로 NULL이 아닌 값을 반환하는 방식으로 `NVL`과 유사한 기능을 제공합니다. 따라서 `coalesce` 함수를 통해 필요한 기능을 충분히 대체할 수 있습니다. 코드 예시를 참고하여 활용해보시기 바랍니다.'



- 현재까지의 대화 내용 출력


```python
print(conversation.memory.load_memory_variables({})["context"])
```

\>\> 출력

    Human: sql에서 nvl에 해당하는 함수를 알려주세요.
    AI: PySpark에서는 `coalesce` 함수를 사용하여 SQL의 `NVL` 함수와 유사한 기능을 구현할 수 있습니다. `coalesce` 함수는 여러 개의 컬럼을 입력으로 받고, 그 중 첫 번째로 NULL이 아닌 값을 반환합니다.
    
    예를 들어, 다음과 같이 사용할 수 있습니다:
    
    ```python
    from pyspark.sql.functions import coalesce
    
    df.withColumn('new_column', coalesce(df.column1, df.column2, df.column3))
    ```
    
    위 코드는 `column1`이 NULL이 아니면 `column1` 값을 반환하고, NULL이면 `column2` 값을 반환하며, `column2`도 NULL이면 `column3` 값을 반환하는 새로운 컬럼 `new_column`을 생성합니다.
    Human: nanvl 함수를 쓰면 안될까요?
    AI: PySpark에서는 `nanvl` 함수가 없습니다. 대신 `coalesce` 함수를 사용하여 유사한 기능을 구현할 수 있습니다. 위에 제시한 예시 코드를 참고해주세요.
    Human: SQL에서는 nvl와 coalesce를 둘 다 쓸 수 있는데 왜 PySpark에서는 coalesce만 쓸 수 있는거죠?
    AI: PySpark에서는 `coalesce` 함수를 사용하여 SQL의 `NVL` 함수와 유사한 기능을 구현할 수 있습니다. 이는 PySpark이 SQL과는 조금 다른 문법을 가지고 있기 때문입니다. `coalesce` 함수를 사용하여 여러 컬럼을 순차적으로 체크하고, 첫 번째로 NULL이 아닌 값을 반환하는 방식으로 `NVL`과 유사한 기능을 제공합니다. 따라서 `coalesce` 함수를 통해 필요한 기능을 충분히 대체할 수 있습니다. 코드 예시를 참고하여 활용해보시기 바랍니다.



#####  ChatPromptTemplate: 대화형 프롬프트 적용

- 유저와 시스템의 역할(role)을 명시적으로 설정하여 챗봇 등 개발에 사용되는 템플릿
- 역할은 System과 Human(유저), AI(봇)으로 나뉘며 각 역할의 Message는 다음과 같다.
    - SystemMessage: AI봇의 역할과 유저와의 상황 등 미리 시스템에 설정하는 설명 값
    - HumanMessage: 유저의 질문과 반응에 대한 메세지
    - AIMessage: AI 모델의 응답
- 그외에 Function, Tool의 message도 있지만 여기서는 다루지 않겠다.

- from_messages() 함수를 사용해 각 역할들에 대한 메세지와 변수들을 넣을 수 있다. 각 역할에 대한 메세지는 튜플 형식으로 들어간다. `[("system","system message"), ("human", "user message")]`


```python
from langchain.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "AI는 동물에 관한 지식이 뛰어난 전문가입니다. 사용자가 던지는 동물에 대한 질문에 대한 적절한 답변을 줄 수 있습니다."),
    ("human", "{question}"),
]
)

message = chat_prompt.format_messages(question = "개과 동물은 무엇이 있습니까?")
print(message)
```

\>\> 출력

    [SystemMessage(content='AI는 동물에 관한 지식이 뛰어난 전문가입니다. 사용자가 던지는 동물에 대한 질문에 대한 적절한 답변을 줄 수 있습니다.'), HumanMessage(content='개과 동물은 무엇이 있습니까?')]


- 위 템플릿을 통해 간단한 주제에 대한 질문을 수행할 수 있다.


```python
from langchain_core.output_parsers import StrOutputParser

chain = chat_prompt | model | StrOutputParser()

chain.invoke({"question": "개과 동물은 무엇이 있습니까?"})
```

\>\> 출력


    '개과 동물은 개와 늑대를 비롯한 개과 동물과 관련된 동물들을 의미합니다. 다른 예로는 여우, 쥐, 미국산 사막여우, 쇠락치, 새와 함께 개과 동물이라고 할 수 있습니다.'



- 이제 여기에 메모리를 붙이기 위해 `MessagesPlaceholder`를 추가한다.
`MessagesPlaceholder`는 `MemoryBuffers`를 사용해 대화 중간에 필요에 의해 메세지들을 유지하거나 대체하는 기능을 하는, 즉 대화의 흐름을 관리하고 조작하는데 사용된다.


```python
from langchain_core.prompts import MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "AI는 동물에 관한 지식이 뛰어난 전문가입니다. 사용자가 던지는 동물에 대한 질문에 대한 적절한 답변을 줄 수 있습니다."),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{question}"),
]
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

memory.load_memory_variables({}) # 대화 내용 빈 것 확인
```

\>\> 출력


    {'chat_history': []}




```python
from langchain.chains import LLMChain

conversation = LLMChain(
    llm=ChatOpenAI(),
    prompt=chat_prompt,
    verbose=False,
    memory=memory
)
```


```python
conversation.invoke({"question": "안녕, 하마와 말은 어떤 관계야"})['text']
```

\>\> 출력


    "안녕하세요! 하마와 말은 모두 '말과 말씨과'에 속하는 동물이지만, 서로 다른 종입니다. 하마는 물에서 생활하는 대형 동물로, 특히 아프리카에 서식하며 주로 물에서 식물을 먹는 식물성 동물입니다. 반면 말은 육지에서 서식하는 동물로, 주로 초원이나 목장에서 살며 초원의 풀이나 견책 등을 먹는 동물입니다. 따라서 두 동물은 서로 다른 서식 환경과 생활습관을 가지고 있습니다."




```python
conversation.invoke({"question": "그럼 왜 비슷한 이름을 가진거야?"})['text']
```

\>\> 출력


    '하마와 말이 비슷한 이름을 가지고 있는 이유는 영어에서의 구별 때문입니다. "하마"는 영어로 "hippopotamus"라고 하며, "말"은 영어로 "horse"라고 합니다. 두 동물의 한글 이름이 비슷하다고 해서 영어 이름이 비슷한 것은 아니기 때문에 이런 차이가 생기게 됩니다. 그렇기 때문에 한글로 된 이름만 보고는 두 동물이 서로 연관되어 있는 것처럼 보일 수 있지만, 실제로는 서로 다른 동물입니다.'




```python
conversation.invoke({"question": "두 동물과 비슷한 경우가 또 있어?"})['text']
```

\>\> 출력


    '네, 동물들의 이름이나 외모 등이 비슷한 경우는 종종 있습니다. 예를 들어, 사자와 호랑이는 둘 다 크고 강한 포식자로서 비슷한 외모를 가지고 있지만, 실제로는 과도가 다른 다른 종입니다. 또한, 얼룩말과 말도 비슷한 외모를 가지고 있지만, 두 동물은 서로 다른 종입니다.\n\n또 다른 예로는 오랑우탄과 침팬지가 있습니다. 두 동물은 비슷한 외모를 가지고 있지만, 생태학적으로는 다른 종으로 분류됩니다. 이렇게 외모나 이름 등이 비슷하지만 실제로는 다른 종으로 분류되는 경우가 동물 세계에는 많이 있습니다.'



- 메모리에 있는 대화 내용 출력


```python
memory.load_memory_variables({})['chat_history']
```

\>\> 출력


    [HumanMessage(content='안녕, 하마와 말은 어떤 관계야'),
     AIMessage(content="안녕하세요! 하마와 말은 모두 '말과 말씨과'에 속하는 동물이지만, 서로 다른 종입니다. 하마는 물에서 생활하는 대형 동물로, 특히 아프리카에 서식하며 주로 물에서 식물을 먹는 식물성 동물입니다. 반면 말은 육지에서 서식하는 동물로, 주로 초원이나 목장에서 살며 초원의 풀이나 견책 등을 먹는 동물입니다. 따라서 두 동물은 서로 다른 서식 환경과 생활습관을 가지고 있습니다."),
     HumanMessage(content='그럼 왜 비슷한 이름을 가진거야?'),
     AIMessage(content='하마와 말이 비슷한 이름을 가지고 있는 이유는 영어에서의 구별 때문입니다. "하마"는 영어로 "hippopotamus"라고 하며, "말"은 영어로 "horse"라고 합니다. 두 동물의 한글 이름이 비슷하다고 해서 영어 이름이 비슷한 것은 아니기 때문에 이런 차이가 생기게 됩니다. 그렇기 때문에 한글로 된 이름만 보고는 두 동물이 서로 연관되어 있는 것처럼 보일 수 있지만, 실제로는 서로 다른 동물입니다.'),
     HumanMessage(content='두 동물과 비슷한 경우가 또 있어?'),
     AIMessage(content='네, 동물들의 이름이나 외모 등이 비슷한 경우는 종종 있습니다. 예를 들어, 사자와 호랑이는 둘 다 크고 강한 포식자로서 비슷한 외모를 가지고 있지만, 실제로는 과도가 다른 다른 종입니다. 또한, 얼룩말과 말도 비슷한 외모를 가지고 있지만, 두 동물은 서로 다른 종입니다.\n\n또 다른 예로는 오랑우탄과 침팬지가 있습니다. 두 동물은 비슷한 외모를 가지고 있지만, 생태학적으로는 다른 종으로 분류됩니다. 이렇게 외모나 이름 등이 비슷하지만 실제로는 다른 종으로 분류되는 경우가 동물 세계에는 많이 있습니다.')]



#####  대화형 프롬프트의 LCEL 적용

LCEL에 메모리를 추가하기 위해 `RunnablePassthrough`를 설정한다. `RunnablePassthrough`의 역할은 LCEL에서 데이터를 그대로 전달하는 것이다. 이번 예제에선 메모리에 존재하는 대화내용(`chat_hisotry`)를 `MessagesPlaceholder`를 사용해 대화 중간에 있는 넣어주는 역할을 한다.

`RunnablePassthrough.assign` 함수를 사용해 대화 내역을 전달해줄 것이며 대화 내용을 반환하는 함수인 `memory.load_memory_variables`함수를 `RunnableLambda`로 넣어 `RunnablePassthrough`를 설정한다. 

`RunnableLambda`는 `RunnablePassthrough`로 전달되는 데이터에 특정 함수(lagnchain에선 tool)를 적용시켜주는 역할을 한다.


```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

runnable = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables)
    | itemgetter("chat_history")  # memory_key 와 동일하게 입력합니다.
)
```

- 위에서 설정한 여러가지 컴포넌트들을 LCEL을 사용해 조립한다.


```python
runnable.invoke({"question": "안녕."})

llm = ChatOpenAI(model = 'gpt-3.5-turbo')

chain = runnable | chat_prompt | llm | StrOutputParser()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.load_memory_variables({}) # 대화 내용 빈 것 확인
```

\>\> 출력


    {'chat_history': []}




```python
chain.invoke({"question" : "거미는 곤충이야?"})
```

\>\> 출력


    "아니요, 거미는 곤충이 아닙니다. 거미는 곤충이 아닌 '거미류'에 속하는 동물입니다. 곤충은 머리, 가슴, 배로 나뉘어진 세 가지 체에 여섯 다리를 가진 동물을 의미하며, 예를 들어 벌레, 나비, 딱정벌레 등이 곤충에 속합니다. 반면 거미는 머리와 가슴이 합쳐진 체를 가지고 있고, 여덟 다리를 가지며 먹이를 사냥하기 위해 거미줄을 짜는 특징을 가지고 있습니다. 따라서 거미는 곤충이 아니라 거미류에 속하는 동물이라고 할 수 있습니다."




```python
chain.invoke({"question" : "더 자세히 말해줘."})
```

\>\> 출력


    "오랑우탄과 침팬지는 둘 다 유인원류에 속하는 대형 포유류로, 인간에게 가장 가까운 친척으로 알려져 있습니다. 그러나 두 종은 서로 다른 종입니다. 오랑우탄은 주로 나무 위에서 생활하며 식물을 주로 먹는 반면, 침팬지는 주로 땅에서 생활하며 곤충, 과일, 작은 동물 등을 먹습니다. 두 종의 외모도 약간 차이가 있으며, 행동양식과 생태학적 특성도 다릅니다.\n\n얼룩말과 말은 둘 다 '말과 말씨과'에 속하는 동물로, 비슷한 외모를 가지고 있지만 서로 다른 종입니다. 얼룩말은 몸에 얼룩무늬가 있는 것이 특징이고, 주로 아프리카 대륙에서 서식합니다. 반면 말은 주로 풀을 먹는 육지 동물로, 전 세계적으로 사람에 의해 길들여져 탈것이나 농업용 동물로 이용되고 있습니다. 이처럼 외모나 서식지, 생활습관 등이 유사하지만 서로 다른 종으로 분류되는 경우가 많이 있습니다."




```python
chain.invoke({"question" : "우리 거미에 대해 말하고 있지 않았어?"})
```

\>\> 출력


    '죄송합니다. 거미는 또 다른 흥미로운 동물입니다. 거미는 저축동물로 분류되며, 거미줄을 짜거나 독을 사용하여 먹이를 사냥하는 데 특화된 동물입니다. 거미는 세계적으로 매우 다양한 종이 존재하며, 크기와 색상 등이 다양합니다. 거미 종 중에는 독성이 있는 종도 있지만, 대부분의 거미는 사람에게 해를 끼칠 정도로 독성이 강하지 않습니다.\n\n거미는 주로 거미줄을 이용하여 먹이를 사냥하는데, 이 거미줄은 거미가 분비하는 점착성 분비물로 만들어집니다. 거미는 먹이를 거미줄에 감아서 포획하고, 이후에 독을 주입하여 먹이를 소화합니다. 거미는 주로 곤충을 사냥하지만, 일부 거미 종은 작은 동물이나 큰 벌레도 사냥합니다.'




```python
memory.load_memory_variables({})
```

\>\> 출력


    {'chat_history': []}



하지만 위의 예시에서 보이듯이 대화 내용을 기억하지 못한다. LCEL을 사용해 체인을 구성할 땐 Memory를 다른 방식으로 넣어줘야 한다. 이를 위해 `MessagesPlaceholder`를 사용해 `ChatPromptTemplate` 중간에 대화 내역을 넣어준다.


```python
from langchain_core.prompts import MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "AI는 동물에 관한 지식이 뛰어난 전문가입니다. 사용자가 던지는 동물에 대한 질문에 대한 적절한 답변을 줄 수 있습니다."),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{question}"),
]
)
model = ChatOpenAI(model = 'gpt-4')
chain = chat_prompt|model
```

여기서 invoke를 이용해 대화를 하려면 `question`뿐 아니고 `chat_history`에도 데이터를 넣어줘야한다. 프롬프트의 input_variables가 `["question", "chat_history"]`인 것이다.
여기엔 `BaseMessage` 들의 리스트를 넣어줘야 하며 `BaseMessage`들로 이루어진 히스토리를 반환하는 함수가 있어야한다. `memory.load_memory_variables({})`로 대화 내역을 그냥 넣어버리면 에러가 난다.


```python
try:
    chain.invoke({"question": "안녕", "chat_history": memory.load_memory_variables({})})
except ValueError as e:
    print(e)
```

\>\> 출력

    variable chat_history should be a list of base messages, got {'chat_history': []}



#### In-memory: 실행 중인 파이썬 커널 안에서 메모리 저장

- 이를 해결하기위한 다른 방법으로 우선 첫번째로 대화 내용을 파이썬 커널안에서 메모리에 올려 사용해보자(메모리에 올려놓기만 한 상태이므로 커널이 종료되면 휘발된다).
- 전역으로 선언된 dictionary에 대화 내용을 넣을 것이며 `get_session_history`라는 함수를 만들어 사용할 것이며 이 함수는 dict의 내용을 읽어 `ChatMessageHistory`로 반환해 prompt안에 있는 `MessagesPlaceholder`에 넣어준다.
- 기 구성한 chain과 위 함수를 사용해 `RunnableWithMessageHistory`를 생성하고 대화를 계속 진행할 수 있다.
    - 이 때 config 라는 인자를 넣어줘야하는데 dictionary 형식으로 넣어주면 되며 session_id를 무조건 가져야한다고 한다.
    - store 딕셔너리에 이 session_id별로 따로 대화가 저장된다. 대화를 계속 진행하고자 한다면 같은 session_id 값을 넣어줘야하며 id가 다른 id를 넣으면 새로운 메모리를 생성하고 다른 id를 가진 대화의 내용은 기억하지 못한다.


```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {} # 대화 내용 저장할 딕셔너리

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_runnable_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)
```


```python

chain_with_runnable_history.invoke({"question": "세상에서 가장 빠르게 날 수 있는 동물은 어떤 동물이야?"},
    config = {"configurable": {"session_id": "123"}})
```

\>\> 출력


    AIMessage(content='세상에서 가장 빠르게 날 수 있는 동물은 청둥오리입니다. 청둥오리는 시속 160km까지 속도를 낼 수 있습니다. 이를 통해 이동하거나 포식자로부터 도망치는데 사용합니다.', response_metadata={'token_usage': {'completion_tokens': 84, 'prompt_tokens': 107, 'total_tokens': 191}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-b728bcfb-60ff-49ec-b1a5-9c7d506ef57f-0')




```python
chain_with_runnable_history.invoke({"question": "어디 사는데?"},
    config = {"configurable": {"session_id": "123"}})
```

\>\> 출력


    AIMessage(content='청둥오리는 아시아와 유럽의 북부 지역에서 주로 서식합니다. 특히 러시아, 몽골, 중국, 북유럽 등의 지역에서 많이 발견됩니다. 겨울이 되면 더 따뜻한 지역으로 이동하는 동물로, 남아시아나 중동, 북아프리카 지역으로 이동하기도 합니다.', response_metadata={'token_usage': {'completion_tokens': 130, 'prompt_tokens': 206, 'total_tokens': 336}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-047e01ba-2d5b-461f-ba4a-09712198c66b-0')




```python
chain_with_runnable_history.invoke({"question": "청둥오리는 아닌 거 같은데..."},
    config = {"configurable": {"session_id": "123"}})
```

\>\> 출력


    AIMessage(content='죄송합니다, 제가 잘못 설명했습니다. 세상에서 가장 빠르게 날 수 있는 동물은 페레그린 매(Falco peregrinus)입니다. 이들은 먹이를 잡기 위해 천식 질주를 할 때 시속 240km에 이르는 놀라운 속도를 내게 됩니다. 이들은 전 세계의 대부분 지역에서 발견될 수 있습니다.', response_metadata={'token_usage': {'completion_tokens': 133, 'prompt_tokens': 362, 'total_tokens': 495}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-56b7c9af-4f54-4369-bfad-238f5336e4e1-0')



- store에 있는 `123`이라는 session_id의 대화 내역


```python
store['123'].messages
```

\>\> 출력


    [HumanMessage(content='세상에서 가장 빠르게 날 수 있는 동물은 어떤 동물이야?'),
     AIMessage(content='세상에서 가장 빠르게 날 수 있는 동물은 청둥오리입니다. 청둥오리는 시속 160km까지 속도를 낼 수 있습니다. 이를 통해 이동하거나 포식자로부터 도망치는데 사용합니다.', response_metadata={'token_usage': {'completion_tokens': 84, 'prompt_tokens': 107, 'total_tokens': 191}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-b728bcfb-60ff-49ec-b1a5-9c7d506ef57f-0'),
     HumanMessage(content='어디 사는데?'),
     AIMessage(content='청둥오리는 아시아와 유럽의 북부 지역에서 주로 서식합니다. 특히 러시아, 몽골, 중국, 북유럽 등의 지역에서 많이 발견됩니다. 겨울이 되면 더 따뜻한 지역으로 이동하는 동물로, 남아시아나 중동, 북아프리카 지역으로 이동하기도 합니다.', response_metadata={'token_usage': {'completion_tokens': 130, 'prompt_tokens': 206, 'total_tokens': 336}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-047e01ba-2d5b-461f-ba4a-09712198c66b-0'),
     HumanMessage(content='청둥오리는 아닌 거 같은데...'),
     AIMessage(content='죄송합니다, 제가 잘못 설명했습니다. 세상에서 가장 빠르게 날 수 있는 동물은 페레그린 매(Falco peregrinus)입니다. 이들은 먹이를 잡기 위해 천식 질주를 할 때 시속 240km에 이르는 놀라운 속도를 내게 됩니다. 이들은 전 세계의 대부분 지역에서 발견될 수 있습니다.', response_metadata={'token_usage': {'completion_tokens': 133, 'prompt_tokens': 362, 'total_tokens': 495}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-56b7c9af-4f54-4369-bfad-238f5336e4e1-0')]



- 다른 session_id를 입력하면 대화 내용을 기억하지 못한다. 같은 session_id를 계속 넣어줘야 대화가 자연스럽게 이어나간다.


```python
chain_with_runnable_history.invoke({"question": "얼마나 빠르다고?"},
    config = {"configurable": {"session_id": "456"}})
```

\>\> 출력


    AIMessage(content='죄송합니다, 어떤 동물에 대해 물어보시는지 명확하지 않습니다. 특정 동물에 대한 질문을 해 주실 수 있나요?', response_metadata={'token_usage': {'completion_tokens': 61, 'prompt_tokens': 86, 'total_tokens': 147}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-77ce2b66-31f1-4304-a458-487d3cb478af-0')




```python
chain_with_runnable_history.invoke({"question": "얼마나 빠르다고?"},
    config = {"configurable": {"session_id": "123"}})
```

\>\> 출력


    '페레그린 매는 독특한 사냥 방법을 사용하며, 이를 위해 고공에서 먹이를 발견하면 몸을 공처럼 모아 빠르게 추락하며 사냥합니다. 이때 그들의 속도는 시속 240km(대략 150마일)에 달합니다. 이는 세상에서 가장 빠르게 날 수 있는 동물로 알려져 있습니다.'




```python
[key for key in store.keys()] # session 이름만 출력
```

\>\> 출력


    ['123', '456']



##### `input` 함수를 사용한 챗봇 만들기

- 콘솔 안에서 파이썬 기본 내장 함수인 input과 while 문을 통해 간단히 챗봇을 만들 수 있다.
- 앞서 적용했던 프롬프트를 사용해 간단히 만들어보았다.


```python
request = 'start'

store = {} # 대화 내용 저장할 딕셔너리

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_runnable_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

while(request != 'exit'):
    request = input("대화를 위해 입력하세요: ")
    print(f"사용자: {request}")
    response = chain_with_runnable_history.invoke({"question": request}, config = {"configurable": {"session_id": "123"}})
    print(f"AI: {response.content}")
```

    사용자: 
    AI: 동물에 대한 질문을 해주세요.
    사용자: 안녕?
    AI: 안녕하세요! 동물에 대해 궁금한 것이 있으신가요?
    사용자: 전세계에 있는 앵무새는 총 몇 종류야?
    AI: 앵무새는 전 세계적으로 약 393 종이 알려져 있으며, 이들은 크기, 색상, 행동 등에 따라 다양합니다. 이 중 일부 종류는 멸종 위기에 처해 있습니다.
    사용자: 앵무새는 크게 어떻게 분류되지?
    AI: 앵무새는 크게 다음의 몇 가지 유형으로 분류될 수 있습니다:
    
    1. 아마존 앵무새: 이 앵무새는 크기가 중간 정도이며, 강한 부리와 밝은 색상을 가지고 있습니다. 매우 지능적이며, 말을 배우는 데 뛰어납니다.
    
    2. 칵투: 작은 크기와 선명한 색상이 특징이며, 매우 사교적이고 활발합니다.
    
    3. 매카우: 이 앵무새는 크기가 크고, 강한 부리와 긴 꼬리를 가지고 있습니다. 매우 지능적이며, 말을 잘 배웁니다.
    
    4. 콘클레어: 이 앵무새는 중간 크기이며, 긴 꼬리와 선명한 색상을 가지고 있습니다. 사교적이고 친근하며, 말을 잘 배웁니다.
    
    5. 로즈페이스드 버드: 이 앵무새는 작은 크기와 선명한 핑크색 얼굴이 특징입니다. 사교적이고 조용하며, 다른 앵무새들에 비해 상대적으로 조용합니다.
    
    이 외에도 여러 가지 앵무새 종류가 있으며, 각각의 특성과 능력이 다릅니다.
    사용자: 
    AI: 네, 무엇이든 물어보세요. 동물에 대한 질문에 대해 도와드릴 수 있습니다.
    사용자: exit
    AI: 알겠습니다. 필요하실 때 언제든지 돌아오세요. 좋은 하루 보내세요!

