---
layout: single
title:  "RAG 실습 1: 논문 PDF파일과 대화하기 [하] 최종 모델 구성 및 Streamlit으로 배포하기)"
classes: wide
categories: RAG
tags: [LangChain, RAG, Streamlit]
---

[이전 포스트](https://lymanstudio.github.io/langchain/rag_2_paper_cleansing/)에서 load된 Document들의 내용을 정제하기 위해 클렌징 체인을 구성하고 적용하여 그 결과를 의미론적으로 chunking하였다.  간단한 TextSpliter를 사용하기 보단 문장 사이의 유사도를 구해 그것을 기반으로 문장들의 부분 집합으로 나누어 의미론적으로 유사한 문장들을 묶어 하나의 Document로 구성하였다.

이번 포스트에선 앞선 포스트들에서 사용한 기능들과 기타 필요한 기능들을 합하여 하나의 완전한 시스템으로 구성하고 streamlit으로 웹 기반 앱을 만들어 배포까지 하는 과정을 소개할 예정이다.

이번 포스트에선 다뤄볼 내용을 간단히 정리하면 다음과 같다.

* 타겟 서비스는 아래와 같은 두가지 큰 단계로 구성된다.
  * 논문 준비 단계: 논문 업로드 ➡ 체인을 통한 텍스트 클렌징 ➡ 의미적으로 묶이게 문장 chunking ➡ 벡터 DB로 저장
  * 질의문답 단계: 사용자 질의 ➡ Q체인을 통한 질문 편집 ➡ A체인을 통한 편집된 질문에 대한 단변 생성 ➡ 사용자에게 최종 답변 전달
* 위 단계를 구성하는 종합적인 파이썬 파일들을 정리하고 main에서 상황에 맞는 분기로 시스템을 조립한다.
* 조립된 코드를 streamlit으로 감싸 앱을 구성한다.

---



# Step 5. 체인 분리

우선 체인을 분리하여 사용자가 입력한 질문에 더욱 유연하고 탄력적으로 대처가 가능하게 구성해보자.

## Paper Clean Chain

첫번째 체인은 업로드된 논문 파일을 loader로 불러와 텍스트 데이터를 편집해주는 Paper Clean Chain이다.

이전 포스트에서 소개한 그대로이며 다음과 같이 함수로 구성했다.

```python
def paper_clean_chain():
    prompt = PromptTemplate.from_template("""
    You are an editor who is an expert on editing thesis papers into rich and redundant-erased writings. Your job is to edit PAPER.
    If the client gives you PAPER(a part of the thesis paper) with PRV_PAGE(the summary of the previous page).
    To make an edited version of PAPER, you have to keep the following rules.
    1. Erase all the additional information that is not directly related to the idea and content of the paper, such as the name of a journal, page numbers, and so on.
    In most cases, the additional information is located in the first or the last part of PAPER. 
    2. Erase all the reference/citation marks of numbers in the middle of PAPER.
    3. Edit PAPER in a rich manner and should contain all the ideas and content. Do not discard any content. 
    4. It has to be related and successive to the content of PRV_PAGE. But should not repeatedly have the PRV_PAGE content.
    5. Note that successive pages are waiting to be edited, so the result should not end with the feeling that it is the last document.
    6. Do not conclude at the end of the current editing, unless PAPER only contains references(imply that current PAPER is the end of the thesis). 

    ## PRV_PAGE: {prv_page}

    ## PAPER: {content} 
    """
    )

    model = ChatOpenAI(model = 'gpt-3.5-turbo')

    return prompt | model | StrOutputParser()
```



## Q Chain

사용자는 자신이 업로드 한 논문에 대한 질문을 할 것이다. 하지만 이 질문은 모델에 들어가기 전 편집되야하는데 그 이유는 크게

- 사람마다 질문 스타일은 매우 다를 수 있고 때에 따라선 간결하지 않고 필요 없는 말이 덧붙여져 있는 경우가 많다.
- 질문은 논문이 사용한 언어로 작성되는 것이 좋다. 논문의 메타 데이터를 참고하여 메타 데이터에 쓰여진 언어로 질문을 변환시켜줄 필요가 있다.
- 사용자의 질문에 대한 답변은 질문에 사용된 언어로 구성돼야 한다.

위의 이유로 사용자 자유분방한 질문을 논문이 사용한 언어로 쓰여진 깔끔한 질문으로 바꿔주며 원래 질문의 언어도 알려주는 체인을 Q(question) chain이라는 이름으로 구성했다.

```python
def q_chain(llm) -> str:
    class response(BaseModel):
        processed_query: str = Field(description="Processed version of user input query")
        language: str = Field(description="The language that the user spoken")
    
    structured_parser = JsonOutputParser(pydantic_object=response)
    
    processing_prompt = PromptTemplate.from_template("""
    Your job is translating and converting a user input query into an easy-to-understand LLM model input query regarding CONTEXT.
    The CONTEXT is a set of metadata for a thesis paper consisting of the title, abstract, and other additional information. Its structure is a sequence of pairs of keys and values.
    
    Here is the sequence of your job:
    1. If needed(using different languages), translate the user input QUERY into the language that CONTEXT is written.
    2. Depending of the CONTEXT values, convert the user input QUERY into a question that a QA LLM model for the CONTEXT paper would be easy to comprehend
    3. OUTPUT is json format object that holds converted output and language that user speaks in QUERY.
        The converted output should go to "processed_query" key and the language go to "language" key.
    # CONTEXT:
    {context}

    # QUERY:
    {question}

    # OUTPUT:          
    """)

    return (
        {
            "context": itemgetter('context') | RunnablePassthrough(),
            "question": itemgetter('question') | RunnablePassthrough()
        }
        | processing_prompt
        | llm
        | structured_parser
    )

```

### 프롬프트

우선 프롬프트를 보자. 프롬프트는 이전과 마찬가지로 상황 설명, 행동 지침, 입력으로 구성된다. 

- 상황은 간단하게 유저의 질의문을 CONTEXT를 감안해서 LLM이 잘 이해할 수 있게끔 CONTEXT가 사용하는 언어로 바꿔주라는 요구사항과 CONTEXT에 대한 배경 설명으로 구성했다.
- 행동 지침은 주목적과 OUTPUT 형태를 특정 형태로 고정 시켜 내밷어 달라고 요구하는 내용이다.
- 입력 받을 파라미터는 CONTEXT: {`context`}, QUERY: {`question`}으로 구성된다. 이 두개의 파라미터는 실행단계에서 RunnablePassthrough로 들어간다.

### 입출력

다음으로 알아볼 q 체인의 특징은 입력과 출력의 구조이다.

> 입력: 사용자의 자연어 질의문과 논문의 메타 데이터

우선 입력은 당연하게도 사용자의 질의문이 된다. 우리의 목적은 질의문을 논문의 목적에 맞는 질의문으로 편집해주는 것이기에 체인의 context에 추가로 논문의 메타 데이터를 넣어주었다.



> 출력: 편집된 영어로 된 질의문(processed_query)과 원래 언어(language)로 구성된 JSON 데이터

llm 모델이 출력을 미리 정해놓은 구조로 내뱉게 하기 위해 pydantic 클래스의 BaseModel을 사용해 JSON 구조로 출력하게 구성했다.
```python
class response(BaseModel):
        processed_query: str = Field(description="Processed version of user input query")
        language: str = Field(description="The language that the user spoken")

structured_parser = JsonOutputParser(pydantic_object=response)
```

위 처럼 `response`라는 클래스를 간단하게 정의해준 뒤 chain의 가장 마지막 출력 파서로 정의한 `response`의 형식으로 내뱉는 JsonOutputParser 객체를 만들어준다.

### Chain 구성

```python
{
    "context": itemgetter('context') | RunnablePassthrough(),
    "question": itemgetter('question') | RunnablePassthrough()
}
| processing_prompt
| llm
| structured_parser
```

- 입력으로 두개의 파라미터를 받아야 하기에 입력으로 들어오는 Dict형태의 인풋에서 key에 따른 각 value을 RunnablePassthrough를 사용해 프롬프트에 넣어준다.

- 구성된 프롬프트는 llm을 거친다.
- llm을 통해 나온 결과는 `structured_parser`로 들어가 미리 정의한 JSON 형식으로 출력된다.



## A Chain

마지막 체인은 최종 답변을 생성해줄 A(answer) Chain이다. 우선 구성을 보면 다음과 같다.

```python
def a_chain(vector_store, retriever, llm):

    prompt = PromptTemplate.from_template("""
    당신의 임무는 논문에 대한 정보를 활용해 사용자가 던지는 질문에 대해 답변을 해주는 것입니다. 주어진 정보는 논문의 제목(TITLE), 논문의 초록(ABSTRACT), 질문에 대한 세부 정보를 담은 컨텍스트(CONTEXT), 그리고 논문에 대한 기타 정보(ADDITIONAL_INFO)입니다.
    답변은 CONTEXT를 기반으로 작성하되 CONTEXT에서 질문과 관련 없는 내용은 무시하고 논문의 제목과 초록을 참고하여 사용자가 이해하기 쉽게 설명해야합니다. 주어진 CONTEXT를 기반으로 답변을 찾을 수 없는 경우 "답변을 찾을 수 없습니다."라고 답변해 주세요.
    답변의 언어는 {language}로 해주세요.                          
    # TITLE:
    {title}

    # ABSTRACT:
    {abstract}
                                        
    # ADDITIONAL_INFO:
    {add_info}                

    # CONTEXT:
    {context}

    # 질문:
    {question}

    # 답변:
    """
    )

    def get_metadata(key:str) -> str: # 벡터 스토어의 첫 Document의 metadata 딕셔너리에서 key에 맞는 value를 뱉어주는 함수
        return next(iter(vector_store.docstore._dict.values())).metadata[key]

    def get_metadata_otherthen(exclude_keys:List[str]) -> str: # 벡터 스토어의 첫 Document의 metadata 딕셔너리에서 인자로 받은 key들을 제외한 다른 key들과 value 쌍을 스트링으로 뱉어주는 함수
        return "\n".join(f"{k} : {v}" for k, v in next(iter(vector_store.docstore._dict.values())).metadata.items() if k not in (exclude_keys))

    def concat_docs(docs:List[Document]) -> str: # retriever가 반환한 모든 Document들의 page_content를 하나의 단일 string으로 붙여주는 함수
        return "".join(doc.page_content for doc in docs)

    return (
        {
            "title": itemgetter('title') | RunnableLambda(get_metadata), # 입력 받은 데이터 중 title을 get_metadata함수의 인자로 넣고 반환 받은 value 값을 title로 prompt에 넣어줌
            "abstract": itemgetter('abstract') | RunnableLambda(get_metadata),
            "add_info": itemgetter('add_info') | RunnableLambda(get_metadata_otherthen),
            "context": itemgetter('question') | retriever | concat_docs, # 입력 받은 데이터 중 question을 retriever에 전달, 반환 받은 k개의 Document들을 concat_docs 함수에 전달, 내용들이 concat된 하나의 스트링을 context로 prompt에 넣어줌
            "question": itemgetter('question') | RunnablePassthrough(), # 입력 받은 데이터 중 question을 그냥 받은 형태 그대로 전달, question으로 prompt에 넣어줌
            "language": itemgetter('language') | RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
```

### 프롬프트

프롬프트는 Q chain과 달리 간단하게 한국어로 입력했다. 주어진 정보들을 종합해 답변을 내주되 입력받은 language 로 써달라는 간단한 내용이다.

### 입출력

A Chain은 마지막 단계인 만큼 많은 정보를 입력으로 받는다. 입력의 종류는 다음과 같다. 총 6개의 파라미터를 받아야 한다.

- 앞서 구성한 Q chain에서 생성된 질문:{`question`}, 언어:{`language`} 정보
- 질문에 답변을 줄 수 있는 Document들로 구성된 CONTEXT:{`context`}
- 답변에 도움을 줄 수 있는 메타 데이터들
  - TITLE:{`title`}
  - ABSTRACT(초록):{`abstract`}
  - ADDITIONAL_INFO(기타 부가 정보):{`add_info`}


출력은 마지막 답변이기에 단순한 스트링이다. 따라서 체인의 마지막 파서는 `StrOutputParser()`를 사용했다.

### Chain 구성

프롬프트로 들어가는 파라미터가 6개로 많기에 각각에 대해 체인에 입력으로 들어온 Dictionary 객체에서 알맞은 value들을 할당해준다.

-  `question`, `language`는 앞선 q chain에서와 마찬가지로 RunnablePassthrough를 통해 그대로 전달된다.
- `title`, `abstract`, `add_info`는 메타 데이터에서 가져와야하는 값들이다. 단순히 RunnablePassthrough로 넣어줄 수 없으며 각 get_metadata, get_metadata_otherthen이라는 함수를 정의해 RunnableLambda를 통해 람다함수처럼 동작하게 한 후 입력 Dict의 각 key에 해당하는 value들을 입력하고 그 결과를 프롬프트의 각 파라미터로 넣어준다.
- `context`는 기 구성된 retriever를 통해 반환된 값을 넣어줘야 한다. 이를 위해 작은 체인이 구성됐다.
  - 입력 Dict의 `question` key에 해당하는 value ➡ retriever로 반환된 Doc들 ➡ concat_docs를 통해 텍스트들을 모두 하나의 스트링으로 이어줌 ➡ 결과를 `context`에 넣어줌

이후 prompt와 llm을 통해 결과를 구한 뒤 결과를 단순 스트링으로 출력한다.



# Step 6. 전체 실행 순서 디자인 및 실행 파일 생성

앞선 여러 단계를 거쳐 구성한 시스템을 조립해 하나의 실행 코드로 만들어보자. 이를 위해 우선 앱 동작 시퀀스를 확정해야 한다.

사용자의 시작 단계 부터  마지막 단계 까지 모든 흐름을 플로우 차트로 구성해보았다.

<img src="./../../images/2024-05-17-rag_3_deploy_model/main_file_flowchart.png" alt="main_file_flowchart" style="zoom: 80%;" />

막연히 생각한 순서는 꽤나 간단한 앱이라고 생각했지만 막상 모든 단계를 그려보니 조금은 복잡한 감이 있다.

다음으로 흐름도를 기반으로 위에서 정의한 chain들을 포함하여 우리가 가진 함수/기능들을 용도에 따라 구분해 각자 파일로 구성해보자.

### RAG에 사용되는 체인들 => `rag_chains.py`

- Cleansing Chain

- Q chain

- A chain


### 기타 기능 => `utils.py`

- PDF 로딩(load_pdf)
- 문서 클렌징(cleansing chain 이용, clean_paper)
- 문서 Chunking(chunk_paper)
- Document 체킹(check_docs_str)

### 벡터 스토어 생성, 로딩 관련 => `vectorstore.py`

- 현재 논문에 대한 벡터 스토어가 기 구축된 경우 로딩(load_store)
- 신규 벡터 스토어 생성(create_store)

### 로컬 저장 대상 => 각자 경로 생성

- 데이터: 논문 PDF등
- 벡터 스토어: 논문을 통해 구성된 vectorStore들

### 전체 실행 파일 => `main.py`

- Streamlit을 사용한 시퀀스에 따른 실행 코드
- API key 체크 함수(is_api_key_valid)
- 쿼리 생성 함수(query)

>  각 파일별 실제 결과는 Github Repo([🔗](https://github.com/lymanstudio/thesis_qa_rag))에 올라가 있는 코드를 참고하면 된다.

위 흐름도를 바탕으로 main_test.py 코드를 구성한 결과는 다음과 같다(chain 생성, 기타 부가 기능, 벡터 스토어 관련 기능은 위에서 정리한 것과 같이 따로 python 파일로 정리한 상태).

```python
import os
import openai
import argparse

from utils import *
from vectorstore import *
from rag_chains import *
from langchain_openai import OpenAIEmbeddings

def is_api_key_valid(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.embeddings.create(input = ["Hello"], model="text-embedding-3-small")
    except:
        return False
    else:
        return True

def query(q_chain, reference, q: str, params : dict = None):
    question = q_chain.invoke(
        {
            "context": reference,
            "question": q  
        }
    )
    
    if params is None:
        params = {
            "title": "title",
            "abstract": "subject",
            "add_info": ['title', 'subject'],
        } 
    print(question)
    return {
        "title": params["title"],
        "abstract": params["abstract"],
        "add_info": params["add_info"],
        "context": question["processed_query"],
        "question": question["processed_query"],
        "language": question["language"]
    }

def main(q):

    file_name = "framework_for_indoor_elements_classification_via_inductive_learning_on_floor_plan_graphs.pdf"
    loaded_pdf = load_pdf_local(file_name = file_name)
    thesis_name = file_name.split('.')[0]
    vectorstore_path = os.path.join("./", "model/vectorstore/", thesis_name + "_index")

    if os.path.exists(vectorstore_path) == False:
        print("▶ Constructing a new vector store for the uploaded paper.")
        print("\t* Cleaning the paper...")

        cleaned_paper, cleaned_paper_concat = clean_paper(loaded_pdf, chain = paper_clean_chain())
        
        print("\t* Chunking pages into sets of relevant sentences...")
        chunked_docs = chunk_paper(cleaned_paper)

        print("\t* Creating a new vector store...")
        vs = create_store(
            chunked_docs, 
            embedding_model= OpenAIEmbeddings(), 
            vdb= 'faiss', 
            save_store = True, 
            save_path = vectorstore_path
        )
        
    else:
        print("▶ Pre-constructed vector store exitsts! Load it from the local directory.")
        vs = load_store(
            embedding_model=OpenAIEmbeddings(),
            load_path= vectorstore_path
        )

    print("▶ Setting up QA bot...")
    # Set Up retriever out of vector store
    retriever = vs.as_retriever(search_type = "mmr", search_kwargs = {"k": 10})

    # Make a QA Chains
    ## Q chain
    meta_data_dict = "\n".join(f"{k} : {v}" for k, v in next(iter(vs.docstore._dict.values())).metadata.items())
    query_chain = q_chain(llm = ChatOpenAI(model = 'gpt-3.5-turbo'))

    ## A chain
    paper_qa_chain = a_chain(
        vector_store = vs,
        retriever = retriever,
        llm = ChatOpenAI(model = 'gpt-3.5-turbo')
    )

    answer = paper_qa_chain.invoke(
        query(
            query_chain,
            reference = meta_data_dict,
            q = q
        )
    )
    
    print(answer)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--q', help='query')
    parser.add_argument('--key', help='API key')
    args = parser.parse_args()

    key_status = is_api_key_valid(args.key)
    print("▶ API Key status : {}".format("Good to go." if key_status else "Invaild or missing API key."))
    if key_status:
        os.environ['OPENAI_API_KEY'] = args.key
        main(args.q)
    else:
        print("exit")
```

위 파일을 커맨드라인으로 아래와 같이 실행하면

```tex
 python main_test.py --key {Your OPENAI_API_KEY} --q 既存の論文との違いは何？ (파파고 번역)기존 논문과의 차이점은 뭐야?
```

다음과 같이 출력된다.
```
▶ API Key status : Good to go.
▶ Pre-constructed vector store exitsts! Load it from the local directory.
▶ Setting up QA bot...
{'processed_query': 'What is the difference from existing papers?', 'language': 'Japanese'}
既存のアプローチとの違いは、この論文が最初に入力される間取り図像をベクトルデータに変換し、グラフニューラルネットワークを活用するという点です。従来のアプローチでは、最初に画像ピクセルをセグメント化するための画像ベースの学習フレームワークが使用されていましたが、本論文では異なります。このフレームワークは、画像の前処理と間取り図像のベクトル化、隣接領域グラフへの変換、変換された間取り図グラフ上のグラフニューラルネットワークの3つのステップで構成されています。これにより、壁、ドア、シンボルなどの基本要素だけでなく、部屋や廊下など の空間要素もキャプチャできるようになりました。また、提案された方法は要素の形状も検出できます。その結果、95%のF1スコアで室内要素を分類できることが実験結果から示されています。その他にも、ノード間の距離を考慮に入れた新しいグラフニューラルネットワークモデルが提案され ており、これは空間ネットワークデータの貴重な特徴です。

(파파고 번역)
기존 접근 방식과의 차이점은 이 논문이 처음 입력되는 방 배치 도상을 벡터 데이터로 변환하고 그래프 신경망을 활용한다는 점입니다.종래의 접근법에서는, 처음에 화상 픽셀을 세그먼트화하기 위한 화상 베이스의 학습 프레임워크가 사용되고 있었지만, 본 논문에서는 다릅니다.이 프레임워크는 이미지 전처리와 방 배치 도상의 벡터화, 인접 영역 그래프로의 변환, 변환된 방 배치도 그래프 상의 그래프 신경망의 세 단계로 구성되어 있습니다.이를 통해 벽, 문, 상징물 등 기본 요소뿐만 아니라 방이나 복도 등 의 공간 요소도 캡처할 수 있게 되었습니다.또한 제안된 방법은 요소의 형상도 검출할 수 있습니다.그 결과 95%의 F1 점수로 실내 요소를 분류할 수 있음이 실험 결과에서 나타났습니다.이외에도 노드 간 거리를 고려한 새로운 그래프 신경망 모델이 제안되며 있으며, 이는 공간 네트워크 데이터의 귀중한 특징입니다.
```

결과를 보니 일어로 질문한 쿼리가 논문이 쓰여진 언어인 영어로 번역된 질문으로 변환됐고 결과는 다시 일어로 변환돼 생성된 것을 확인할 수 있다. 또한 막연한 질문임에도 답변의 길이와 내용이 일반적으로 유용한 정보를 담고 있었다.

마지막으로 위 코드를 streamlit으로 wrapping 하여 main 파일을 만들 수 있다.  streamlit으로 wrapping하는 과정은 streamlit소개와 함께 따로 포스트로 구성할 예정이다. 최종 결과 main.py는 [여기](https://github.com/lymanstudio/thesis_qa_rag/blob/main/main.py)에 있다.



# 결과 및 시연

streamlit을 통해 웹 어플리케이션을 만들고 deploy하면 실제로 배포가 진행된다. 저장소는 개발자가 구성한 github repo와 연결돼있으며 퍼블릭 상태여야 한다고 한다.

최종 결과는 아래와 같다.

![demonstration](./../../images/2024-05-17-rag_3_deploy_model/demonstration.gif)
