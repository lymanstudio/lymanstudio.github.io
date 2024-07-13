---
layout: single
title:  "LangChain의 Configurables을 통한 LLM 모델 쉽게 바꿔 사용/비교해보기"
classes: wide
categories: LangChain
tags: [LangChain, Configurables]
---





랭체인은 언어모델 어플리케이션 개발에 필요한 여러 도구들의 모음이기에 여러 가지 편리한 기능들이 꽤나 많다.

이번 포스트에선 하나의 모델 객체 선언으로 여러 가지 다양한 언어 모델, 또는 설정을 손쉽게 바꿔 사용 가능한 Configurables에 대해 알아보도록 하겠다. 간단하게 웹 페이지 데이터를 가져와 RAG를 구축해 테스트해볼 예정이다.

오늘 사용해볼 모델들은 가장 대중적인 OpenAI의 gpt 모델군과 Anthropic의 Claude 모델 군이다.

<br>

#### API Key 세팅 및 관련 패키지 import

```python
import os

os.environ['ANTHROPIC_API_KEY'] = "YOUR API KEY"
os.environ['OPENAI_API_KEY'] = "YOUR API KEY"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
## !pip install langchain_anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import ConfigurableField, RunnablePassthrough
from operator import itemgetter
from langchain_community.vectorstores.faiss import FAISS
```

<br><br>

## 웹페이지 데이터를 통한 RAG 구축

우선 간단하게 웹페이지의 데이터를 가져와 문답을 할수 있는 RAG환경을 구축해야 한다. RAG chain 구축이 이번 포스트의 주제는 아니기에 빠르게 진행하며 넘어가겠다.

<br>

#### **웹페이지 기반 Documents 생성, VectorStore 구축**

우선 RAG 구축을 위해 간단한 코드들을 선언한다. 다음 두개의 함수 정의로 끝낼 수 있다.

- `get_documnet_form_web`: URL을 하나 받아 그 안에 있는 텍스트들을 가져온 뒤 `RecursiveCharacterTextSplitter`를 사용해 여러 개의 Document로 분할
- `create_db`: 위에서 나온 Document 리스트를 받아 DB로 생성 후 반환

```python
def get_documnet_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
        )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding = embedding)
    return vectorStore
```

<br>

#### **체인 구성**

다음으로 간단한 프롬프트를 만들어 문답을 수행하는 체인을 생성한다. 앞서 정의한 벡터 스토어와 LLM 모델을 입력으로 받아 체인을 구성한다.

프롬프트의 입력으로 들어갈 context는 앞서 생성한 벡터 스토어의 리트리버를 통해 넣어준다. RunnablePassthrough에 대한 자세한 설명은 [앞선 포스트](https://lymanstudio.github.io/langchain/lcel_chat_chain_example/)를 참조하면 된다.

```python
def get_chain(vectorStore, model):
    prompt = ChatPromptTemplate.from_template("""
        Answer the users's QUESTION regarding CONTEXT. Use the language that the user types in.
        If you can not find an appropriate answer, return "No Answer found.".
        CONTEXT: {context}
        QUESTION: {question}
    """)

    retriever = vectorStore.as_retriever()

   def concat_docs(docs) -> str: # retriever가 반환한 모든 Document들의 page_content를 하나의 단일 string으로 붙여주는 함수
        return "".join(doc.page_content for doc in docs)

    chain = {
        "context" : itemgetter('question') | retriever | concat_docs
        , "question" : itemgetter('question') | RunnablePassthrough()
    } | prompt| model | StrOutputParser()

    return chain
```

<br>

### 테스트

위에서 만든 함수들로 빠르고 간단하게 테스트를 해보자. 대상 페이지는 [The Shapely User Manual](https://shapely.readthedocs.io/en/stable/manual.html)로 Shaply라는 기하객체 분석 파이썬 패키지에 대한 Documentation이다. 해당 페이지는 하나의 길다란 웹페이지에 전체 내용이 들어가 있어 테스트에 적절한 문서로 판단된다. LLM모델은 gpt-4o 모델로 구성했다.

```python
docs = get_documnet_from_web("https://shapely.readthedocs.io/en/stable/manual.html")
vectorStore = create_db(docs)
chain = get_chain(vectorStore, ChatOpenAI(model = 'gpt-4o'))

print(chain.invoke({"question": "How can I determine whether a point is in a polygon?"}))
```

> output

````
You can determine if a point is in a polygon by using the `contains` method from the `shapely` library. Here is an example:

```python
from shapely.geometry import Point, Polygon

# Define the polygon
polygon = Polygon([(0, 0), (1, 1), (1, 0)])

# Define the point
point = Point(0.5, 0.5)

# Check if the point is in the polygon
is_contained = polygon.contains(point)
print(is_contained)  # This will print True if the point is within the polygon, otherwise False
```

This code will return `True` if the point (0.5, 0.5) is inside the polygon defined by the coordinates [(0, 0), (1, 1), (1, 0)].

Additionally, if you need to test a large batch of points, you can use the `prepared.prep()` function to optimize the process, as shown in the context:

```python
from shapely.prepared import prep

# Define a large list of points
points = [...]  # Replace with your list of points
polygon = Polygon([(0, 0), (1, 1), (1, 0)])
prepared_polygon = prep(polygon)

# Filter points that are contained within the polygon
hits = filter(prepared_polygon.contains, points)
```

This will give you an iterable `hits` containing all points that are within the polygon.
````

결과를 보니 간단한 코드임에도 잘 작동하며 shaply 패키지를 사용한 예시를 들며 웹페이지 기반 document들에서 정보를 가져와 답변했다는 것을 알 수 있다. 한국어도 질문해보았다. 

```python
print(chain.invoke({"question": "하나의 폴리곤에서 다른 폴리곤과의 겹쳐진 영역을 잘라낼 수 있어? 샘플 코드를 알려줘"}))
```

> output

````
네, Shapely를 사용하여 하나의 폴리곤에서 다른 폴리곤과의 겹쳐진 영역을 잘라낼 수 있습니다. 아래는 샘플 코드입니다.

```python
from shapely.geometry import Polygon

# 두 개의 예제 폴리곤 정의
polygon1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
polygon2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

# 겹쳐진 영역 계산
intersection = polygon1.intersection(polygon2)

# 겹쳐진 영역이 있으면 출력
if not intersection.is_empty:
    print("겹쳐진 영역:", intersection)
else:
    print("겹쳐진 영역이 없습니다.")
```

이 코드는 두 개의 폴리곤을 정의하고, 그들 간의 겹쳐진 영역을 계산하여 출력합니다. Shapely의 `intersection` 메서드를 사용하면 겹쳐진 영역을 쉽게 계산할 수 있습니다.
````

한국어로 질문하니 한국어로도 잘 답변하는 모습이다.



<br>

<br>

## **Configurables: 모델/설정 변경**

만약 우리가 하나의 모델 객체 선언으로 다양한 LLM 모델을 쓰고 싶다면 어떻게 해야할까? langchain에선 Configurables을 통해 쉽게 모델이나 설정 값을 바꿀 수 있게 해준다. 빠르게 예시를 통해 설명해보면 다음과 같다.
```python
llm = (
    ChatAnthropic(model_name='claude-3-5-sonnet-20240620')
    .configurable_alternatives(
        ConfigurableField(
            id = 'llm',
            name="LLM Model",
            description="The base LLM model",
        ),
        default_key="claude3_5_sonnet",
        claude3_haiku=ChatAnthropic(model_name='claude-3-haiku-20240307'),
        gpt4o = ChatOpenAI(model = 'gpt-4o'),
        gpt3_5 = ChatOpenAI(model = 'gpt-3.5-turbo'),
    )
    .configurable_fields(
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        ),
        max_tokens = ConfigurableField(
            id="max_token",
            name="Maximum input Tokens",
            description="Maximum limit of input Tokens",
        ),
    )
)
```

<br>

### configurable_alternatives: LLM 모델 선택

이번엔 먼저 베이스 모델을 클로드의 `claude-3.5-sonnet-20240620`모델을 사용해 생성한다.

그 이후 configurable_alternatives를 설정에 들어가 `llm`이라는 필드명으로 ConfigurableField를 생성해준 뒤 (`default_key`)로 `claude3_5_sonnet`라는 이름을 준다. 이는 앞서 생성한 `claude-3.5-sonnet-20240620`모델을 사용한 객체를 디폴트 모델로 설정해준 것이다.

다음으로 llm 필드에 여러 옵션을 계속 붙여준다. 파라미터명이 llm 필드에 들어갈 key 값이고 뒤에 반환될 객체들을 선언해주면 된다.

- 예를 들어 `{"llm": "gpt4o"}`으로 불러 gpt-4o모델을 사용하고 싶으면 `gpt4o = ChatOpenAI(model = 'gpt-40')`와 같이 붙여주면 된다.
- 위 예시에선 `claude3_haiku=ChatAnthropic(model_name='claude-3-haiku-20240307')`, `gpt3_5 = ChatOpenAI(model = 'gpt-3.5-turbo')`도 더 추가해주어 총 4개의 모델을 고를 수 있는 상태가 됐다.

<br>

### configurable_fields: LLM 모델 설정 변경

`configurable_fields`를 사용해 설정 값도 바꿀 수 있다. ChatGPT와 Claude 두 모델군 모두 temparature값과 max_tokens 값을 모델 객체 생성 시 설정할 수 있으므로 configurable_fields 세팅을 통해 설정 변경 목록에 넣어주어 바꾸어 쓰고 싶을 때 변경하여 사용할 수 있다.



### 사용법

```python
chain.with_config(configurable={
    	"llm": "MODEL_NAME", 
    	"temparature": .5,
		"max_tokens" : 2048
	}
 ).invoke(question)
```

사용법은 미리 정의된 chain객체의 with_config 메서드를 사용하면 된다. 간단하게 설정해둔 선택지의 id를 key로 그리고 선택 값을 value로 하는 딕셔너리를 넣어주면 된다. 위에서 정의한 코드를 사용해 실행해보면 다음과 같다.

```python
docs = get_documnet_from_web("https://shapely.readthedocs.io/en/stable/manual.html")
vectorStore = create_db(docs)
chain = get_chain(vectorStore, llm)

result = chain.with_config(configurable={
    	"llm": "claude3_5_sonnet", 
    	"temparature": .5,
		"max_tokens" : 2048
	}
).invoke({"question": "하나의 폴리곤에서 다른 폴리곤과의 겹쳐진 영역을 잘라낼 수 있어? 샘플 코드를 알려줘"})
```

> output

````
네, Shapely를 사용하여 한 폴리곤에서 다른 폴리곤과 겹치는 부분을 잘라낼 수 있습니다. 다음은 이를 수행하는 샘플 코드입니다:

```python
from shapely.geometry import Polygon

# 두 개의 폴리곤 생성
polygon1 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
polygon2 = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])

# 겹치는 부분 제거
result = polygon1.difference(polygon2)

print(result)
```

이 코드에서:

1. `Polygon` 클래스를 사용하여 두 개의 폴리곤을 생성합니다.
2. `difference()` 메소드를 사용하여 `polygon1`에서 `polygon2`와 겹치는 부분을 제거합니다.
3. 결과는 `polygon1`에서 `polygon2`와 겹치지 않는 부분만 남은 새로운 geometry 객체입니다.

이 예제에서 `result`는 원래 `polygon1`에서 `polygon2`와 겹치는 부분이 제거된 새로운 폴리곤이 됩니다. 

결과를 시각화하거나 좌표를 확인하려면 추가적인 코드가 필요할 수 있습니다. 예를 들어, 결과의 좌표를 보려면 `print(list(result.exterior.coords))`를 사용할 수 있습니다.
````

개인적으로 질문한 바에 대해 앞선 답변보다 더 정확한 답변을 얻었다. 

<br>



### 예시를 통한 비교

손쉬운 비교를 위해 모델별로, 설정 값 별로 같은 질문에 대해 비교 가능한 함수를 간단하게 만들어 사용해 보았다.

```python
def get_answer(chain, question,  temp = .1, max_tokens = 2048):
    for model_name in ['claude3_haiku', 'claude3_5_sonnet', 'gpt3_5', 'gpt4o']:
        try:
            start = time()
            result = chain.with_config(configurable={
                        "llm": model_name, 
                        "temparature": temp,
                        "max_tokens": max_tokens
                    }).invoke({"question": question})
            print("\n\nModel: {} \t\t Answer: {} \t\t elapsed_time: {}".format(model_name, result, round(time() - start, 1)))
        except:
            print("\n\nModel: {} \t\t Answer: {}".format(model_name, "Error."))
question = "지역 인접 그래프를 만들 수 있어? 가능하다면 샘플 코드를 알려줘."
get_answer(chain, question)
```

각 모델의 결과 비교를 위해 문서 내에서 쉽게 찾기 힘든 질문을 해 보았다. 지역 인접 그래프(Regional Adjacency Graph) 모델은 해당 패키지로 만들 수 있지만 문서에는 직접적으로 그 내용이 나오진 않는다.



> output

````
Model: claude3_haiku 		 Answer: Based on the provided context, it seems that the question is regarding the ability to create a regional adjacency graph using the Shapely library in Python. The context discusses how Shapely is a Python package for set-theoretic analysis and manipulation of planar features, and that it can be used for computational geometry tasks outside of a traditional RDBMS.

To create a regional adjacency graph using Shapely, you can follow these general steps:

1. Load the geographic data (e.g., polygons representing regions) into Shapely objects.
2. Use the `Shapely.relation` module to determine the spatial relationships between the regions, such as which regions are adjacent or touch each other.
3. Construct a graph data structure (e.g., using a library like NetworkX) where the nodes represent the regions and the edges represent the adjacency relationships between them.

Here's a sample code snippet to give you an idea of how this can be done:

```python
from shapely.geometry import Polygon
import networkx as nx

# Define the regions as Shapely Polygon objects
region1 = Polygon([(0, 0), (0, 5), (5, 5), (5, 0)])
region2 = Polygon([(5, 0), (5, 5), (10, 5), (10, 0)])
region3 = Polygon([(5, 5), (5, 10), (10, 10), (10, 5)])

# Create a graph
G = nx.Graph()

# Add the regions as nodes to the graph
G.add_node("region1", geometry=region1)
G.add_node("region2", geometry=region2)
G.add_node("region3", geometry=region3)

# Check for adjacency between the regions and add edges to the graph
for n1 in G.nodes:
    for n2 in G.nodes:
        if n1 != n2 and G.nodes[n1]["geometry"].touches(G.nodes[n2]["geometry"]):
            G.add_edge(n1, n2)

# The resulting graph G now represents the regional adjacency relationships
```

This is a simplified example, and you may need to adapt it to your specific use case and data format. Additionally, the Shapely library provides various other spatial analysis functions that you can use to further enrich the graph, such as finding the shared boundaries between adjacent regions or calculating the distances between them. 		 elapsed_time: 4.8


Model: claude3_5_sonnet 		 Answer: 죄송합니다. 주어진 문맥에서는 지역 인접 그래프를 만드는 방법에 대한 정보를 찾을 수 없습니다. 문맥은 주로 Shapely 라이브러리와 공간 분석에 대한 일반적인 설명을 포함하고 있지만, 특정 그래프 생성 방법에 대한 세부 정보나 샘플 코드는 제공하지 않습니다. 따라서 이 질문에 대해 "No Answer found."라고 답변해야 할 것 같습니다. 		 elapsed_time: 3.1


Model: gpt3_5 		 Answer: No Answer found. 		 elapsed_time: 0.9


Model: gpt4o 		 Answer: No Answer found. 		 elapsed_time: 1.2
````

놀랍 게도 claude3_haiku 만이 shaply를 이용해 답을 내주었다. 다른 고성능 모델에선 모두 답을 못 내어주었는데 claude3_haiku는 지역인접 그래프에 대한 정확한 정의와 shaply에서 구현 가능한 정확한 방법까지 알려주었다.  하지만 왜인지 영어로 답변해주었다.



> input

```python
question = "어느 한 폴리곤에서 가장 가까운 선을 구하는 방법이 있어?"
get_answer(chain, question)
```

> output

````
Model: claude3_haiku 		 Answer: The given context suggests that the Shapely library in Python can be used to perform spatial analysis and manipulation of planar features. To find the closest line within a polygon, you can use the `polylabel()` function from the `shapely.ops` module.

Here's an example code snippet that demonstrates how to find the closest point within a polygon:

```python
from shapely.ops import polylabel
from shapely.geometry import LineString

# Create a sample polygon
polygon = LineString([(0, 0), (50, 200), (100, 100), (20, 50), (-100, -20), (-150, -200)]).buffer(100)

# Find the closest point within the polygon
label = polylabel(polygon, tolerance=10)

print(label)  # Output: <POINT (59.356 121.839)>
```

The `polylabel()` function takes a polygon as input and returns the point within the polygon that is closest to the centroid of the polygon. The `tolerance` parameter specifies the accuracy of the calculation.

So, to find the closest line within the polygon, you can use the `nearest()` method of the `label` object to find the closest point on the polygon's boundary, and then use that point to find the nearest line segment.

```python
# Find the closest point on the polygon's boundary
closest_point = polygon.boundary.nearest(label)

# Find the nearest line segment
nearest_line = polygon.boundary.interpolate(polygon.boundary.project(closest_point)).coords[0]

print(nearest_line)  # Output: (59.356, 121.839)
```

This will give you the coordinates of the nearest line segment within the polygon. 		 elapsed_time: 4.4


Model: claude3_5_sonnet 		 Answer: 주어진 문맥에서 특정 폴리곤에서 가장 가까운 선을 구하는 방법에 대한 직접적인 답변을 찾을 수 없습니다. 하지만 Shapely 라이브러리를 사용하여 유사한 작업을 수행할 수 있을 것 같습니다. 

예를 들어, Shapely의 `polylabel` 함수는 폴리곤 내부의 대표점(pole of inaccessibility)을 찾는 데 사용됩니다. 이 개념을 응용하여 폴리곤 주변의 선들과의 거리를 계산하고 가장 가까운 선을 찾는 방식으로 접근할 수 있을 것 같습니다.

그러나 정확한 답변을 제공하기 위한 충분한 정보가 문맥에 없으므로, 정확한 답변은 "No Answer found."입니다. 		 elapsed_time: 4.4


Model: gpt3_5 		 Answer: No Answer found. 		 elapsed_time: 0.9


Model: gpt4o 		 Answer: No Answer found. 		 elapsed_time: 0.8
````

이번에는 문서의 내용을 활용해 주어진 문제에 대한 해결방법을 물어보았는데 이번에도 claude3_haiku가 가장 자세한 답변을 주었다. claude3_5_sonnet은 힌트만 주었을 뿐 결론은 No Answer found를 내주었다.

<br>

<br>



#### 참고 링크

---

- [테디노트 - 랭체인LangChain 노트](https://wikidocs.net/235704)
