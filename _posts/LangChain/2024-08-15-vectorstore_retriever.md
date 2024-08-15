---
layout: single
title: "LangChain의 VectorStore, Retriever의 로직 동작 설명, 코드 분석"
classes: wide
categories: LangChain
tags: [LangChain, RAG, VectorStore]
---

## 데이터 Ingestion flow Overview

텍스트 소스=> Loader => List[Document] => 



<br>

## 벡터 스토어(VectorStore)

벡터 스토어는 RAG의 중요한 컴포넌트 중 하나로 비정형 데이터를 임베딩 벡터로 변환 & 저장하고 사용자의 비정형 질의 또한 임베드 하여 저장된 임베딩 벡터과 유사도를 계산, 가장 유사한 임베딩 벡터를 반환하는 기능까지 수행한다. 

![Diagram illustrating the process of vector stores: 1. Load source data, 2. Query vector store, 3. Retrieve &#39;most similar&#39; results.](https://python.langchain.com/v0.1/assets/images/vector_stores-125d1675d58cfb46ce9054c9019fea72.jpg)

<div style="text-align: center; color: gray;">그림 출처: <a href="https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/">LangChain: Vector Stores</a> </div>

#### 임베딩 벡터

임베딩 벡터에 대해 간단히 설명하자면, 흔하게 관계형 DB에서 쓰이는 정형데이터는 tabalur 형태, 즉 테이블 형태로 행과 열로 구성된 정리된 데이터이다. 이러한 데이터가 아닌 자연어, 사진, 음성파일 등의 비정형 데이터를 다룰 땐 임베딩 벡터라는 형태로 변환하여 저장하고 다룬다. 

임베딩 벡터는 특정한 차원으로 정해진 숫자들의 array로 벡터 하나가 특정 객체를 표현한다고 볼 수 있다. 벡터 형태로 변환해 사용하는 이유는 계산을 하기 위함인데, 벡터 사이의 연산을 통해 벡터 사이의 유사도를 계산할 수 있으며 계산 값에 따라 벡터들, 즉 객체들이 의미적으로 유사한지 상이한지 간접적으로 알 수 있다.

> 참고로 벡터 DB는 확장된 벡터 스토어라고 볼 수 이다. 단순히 임베딩 벡터의 변환 및 저장, 그리고 벡터 서치의 기능을 제공하는 벡터 스토어와 달리 벡터 DB는 벡터 스토어의 기본 기능에 더해 보통 DB의 다른 복합적 기능, 예를 들면 트랜젝션 관리, 메타데이터, SQL 쿼리 지원 등을 통합적으로 제공한다.

<br>

### 기본 클래스 구조 분석 : [VectorStore](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore.__init__)

LangChain에서의 벡터 스토어는 앞서 말한 벡터스토어의 기본 기능만을 수행하는 객체로 한정된다. 모든 벡터스토어는 기본 베이스 클래스인 `VectorStore`를 상속받아 구현되며 기본적인 기능들과 그 기능에 대한 관련된 대표적인 메서드들을 리스트업하면 다음과 같다.

- 생성
  - from_texts
  - from_documents
- 추가/삭제
  - add_documents
  - add_texts
  - delete
- 검색
  - id로 검색
  - 벡터 검색(시멘틱 검색)
- 리트리버 생성
  - as_retriever(아래 Retriver 섹션에서 후술)

#### 벡터 스토어 생성

벡터스토어를 생성하기 위해 필요한 것은 입력으로 주어지는 스트링들의 리스트이다. 보통은 Loader를 통해 생성된 문서(Document)들의 리스트( `List[Document]`)가 입력이 주어지며 `from_documents` 라는 클래스 메서드를 호출, 여기에 입력된 `Document` 객체 각자를 하나의 임베딩 벡터로 만든다.

```python
def from_documents(
        cls: Type[VST],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> VST: # 벡터 스토어 타입 (VST = TypeVar("VST", bound="VectorStore"))
    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]
    return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)
```

 `from_documents`를 보면 입력으로 들어온 `List[Document]`에서 각 Document들의 page_content와 메타 데이터를 뽑아 따로 리스트로 만들고 `from_texts`라는 다른 클래스 메서드를 호출한다.

`from_texts` 는 `VectoreStores` 객체를 반환, 즉 생성하는 기능을 한다.  `@abstractmethod` 데코레이터로 꾸며진 추상 클래스이기에 상속받는 클래스에서 반드시 오버라이드로 재정의하여 사용해야 한다. 

아래 코드에서 `VST`는 벡터스토어 객체타입 또는 그 하위 클래스를 의미한다. `typing.TypeVar`를 사용해 `VST`라는 타입을 임시로 정의했고 반환 되는 객체는 `VST`, 즉 VectorStore 또는 그 하위의 클래스여야만 한다. 위의 `from_documents`에도 마찬가지로 적용돼있다.

```python
@classmethod
@abstractmethod
def from_texts(
    cls: Type[VST],
    texts: List[str],
    embedding: Embeddings,
    metadatas: Optional[List[dict]] = None,
    **kwargs: Any,
) -> VST: # 벡터 스토어 타입 (VST = TypeVar("VST", bound="VectorStore"))
```

####  Document 추가/삭제

기 구성된 vectorStore에 Document를 추가할 수 있다. `from_documents`, `from_texts`의 관계와 마찬가지로 `add_documents`와 `add_texts`가 비슷한 구조로 구현돼있다. `from_documents` 에 추가되는 `List[Document]`를 인자로 주고 page_content와 메타 데이터를 뽑아 따로 리스트로 만들고 두 리스트를 인자로 하여`self.add_texts` 를 호출해 넣어준다. `self.add_texts`는 하위 클래스에서 정의돼야 사용 가능하다. (`raise NotImplementedError("delete method must be implemented by subclass.")`)

삭제 또한 `delete` 메서드로 가능하며 하위 클래스에서 정의된다. 보통은 아래 검색에서 설명할 document_id로 삭제시킨다.

#### Document 검색

##### id로 검색

검색은 크게 두가지로 나뉜다. 우선 일반적인 검색은 저장된 `Document`의 ID를 통해 수행 가능하다. 벡터스토어에 현재 저장된 `Document` 들은 `vectorStore.docstore._dict` 로 직접 볼 수 있는데, `document_id(str): Document`의 딕셔너리 형태이다. key값인 document_id를 통해 각 `Document` 객체들을 직접 인덱싱 가능하며 삭제 역시 바로 가능하다.

만약 어떤 한 Document의 ID를 알면 `self.docstore.search(_id)`로 직접 인덱싱이 가능하다. (* 기본적으로 `get_by_ids` 라는 메서드가 있는데 구현은 안되고 있는 것 같다.)

##### 시멘틱 검색

시멘틱 검색, 즉 의미적으로 검색할 수 있는 기능은 벡터 스토어나 벡터 DB를 사용하는 가장 주된 이유일 것이다. 일반적인 검색이 단순히 단어들 간의 매칭이라면 시멘틱 검색은 벡터 스페이스 내에 존재하는 임베딩 벡터사이의 거리 또는 각도를 기반으로 한 유사도를 통한 검색을 의미한다. 

가령 `앨범` 이라는 단어와 `음반`이라는 단어는 스트링 자체로 보면 아무 연관이 없지만 그 의미적으론 상당히 유사하다는 것을 우리는 알 수 있다. 잘 훈련된 임베딩 모델을 통해 두 단어를 임베딩 벡터로 표현하면 두 벡터는 상당히 유사한 값으로 구성될 것이며 특정 차원의 공간에서 거리적으로나 각도적으로 유사하게 나타난다.

이러한 의미적 유사도를 통해 검색하는 기능을 시멘틱 검색이라고 하며 벡터스토어에선 가장 주된 기능 중 하나로 다양한 유사도 계산법에 따른 다양한 검색 메서드를 제공한다.

![vectorStore_searchmethod_structure.drawio](./../../images/2024-08-15-vectorstore_retriever/vectorStore_searchmethod_structure.drawio.png)

- `search`: 이함수는 단순히 `search_type`에 따라 아래 함수들을 분기하여 실행하는 인스턴스 메서드로 override 대상이 아니다. `search_type` 파라미터에 따라 다음 함수를 실행한다.
    
    - similarity: `similarity_search`
    - similarity_score_threshold: `similarity_search_with_relevance_scores`
    - mmr: `max_marginal_relevance_search`
    - else: ValueError 리턴
    
- `similarity_search`: 가장 기본적인 시멘틱 서치 함수이다. 인자로 자연어 질문인 `query`와 반환하는 문서의 수인 `k`를 받는다. 추상 메서드(`@abstractmethod`)이기에 하위 클래스에서 필수로 override하여 사용해야 한다. `k`개의 Document의 리스트를 반환한다.
    ```py 
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
    ```

- `similarity_search_with_score`: `similarity_search`에서 스코어를 추가하여 반환하는 메서드이다. 따라서 리턴 타입이 `List[Tuple[Document, float]]`이다. 또한 argument를 자유롭게 받을 수 있다. 역시 `raise NotImplementedError`로 하위 클래스에서 override해야 사용 가능하다.
    ```py 
    def similarity_search_with_score(self, *args: Any, **kwargs: Any) -> List[Tuple[Document, float]]:
        raise NotImplementedError
    ```

    > 참고로  `@abstractmethod`와 `NotImplementedError`의 차이는 ABC 모듈을 사용하는지 안하는지의 차이인데, `@abstractmethod`는 해당 메서드를 하위 클래스에서 override하지 않으면 import 자체가 안되며 이와 달리 `NotImplementedError`는 하위 클래스에서 해당 메서드를 호출할 때 override 되지 않으면 에러가 발생하는 것이다. 즉 하위 클래스에서 메서드별 필수 override 대상인지 아닌지의 차이이다.

- `similarity_search_with_relevance_scores`: [0,1] 사이의 연관 스코어와 함께 `Document`들을 반환한다. 이 메서드 자체는 껍데기이며 `_similarity_search_with_relevance_scores`를 통해 로직이 구현된다. 이 메서드에선 `score_threshold`인자를 kwargs에 줘서 threshold가 일정 이상인 similarity를 가진 `Document`들만 각자의 스코어와 함께 반환해주는 것이다.
    ```py
    def similarity_search_with_relevance_scores(self,
        query: str,k: int = 4,**kwargs: Any,) -> List[Tuple[Document, float]]:
        
        (... kwargs:score_threshold 사이에 들어가는 score를 가진 Document들만 거르는 로직...)
    
        return docs_and_similarities
    ```
- `_similarity_search_with_relevance_scores`: 

    ```py
    def similarity_search_with_relevance_scores(self,
        query: str,k: int = 4,**kwargs: Any,) -> List[Tuple[Document, float]]:
    ```

<br>

## 리트리버(Retrievers)

리트리버(반환기)는 말 그대로 Document 들을 반환하는 기능을 하는 컴포넌트이다. 주어진 자연어(혹은 다른 타입의 비정형 데이터)로 구성된 쿼리로 부터 연관된(relevant) Document들을 미리 정해진 파라미터에 따라 내어주는 기능을 주로 수행한다.

벡터 스토어와 달리 Document들의 임베딩 벡터를 직접 저장해놓지 않으며 그저 반환만 하며 따라서 특정 벡터 스토어를 Backbone으로 사용하는 경우가 많다. 하지만 꼭 특정 벡터 스토어를 기반으로 구축/작동될 필요는 없다. 예를 들어 Ensemble Retriever는 특정 벡터 스토어를 기반으로 작동하는 것이 아닌 복수 개의 다른 Retriever 객체를 받아 혼합한 결과를 반환한다.

#### 기본 사용법

특정 벡터 스토어를 기반으로 구축된 보통의 리트리버는 벡터 스토어의 베이스 클래스인 [vectorStore](https://api.python.langchain.com/en/latest/_modules/langchain_core/vectorstores/base.html#VectorStore.from_texts)에서 as_retriever()로 만들어진다. 아래는 한 벡터스토어 인스턴스에서 현재 벡터스토어를 사용해 VectorStoreRetriever 객체를 만들어 반환하는 단순한 코드이다.

```python
def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
	tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
    return VectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)
```

코드를 보면 태그객체는 만들어 그것을 기반으로 `VectorStoreRetriever` 인스턴스를 생성, 반환한다. `VectorStoreRetriever`는 벡터스토어를 기반으로 한 리트리버의 한 종류로   `BaseRetriver`를 상속 받아 정의된다. `BaseRetriver` 부터 차례 대로 살펴보자.

### 기본 클래스 구조 분석

#### 1. [BaseRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain_core.retrievers.BaseRetriever.html#langchain_core.retrievers.BaseRetriever)

문서 반환시스템(Document reterival system)에 대한 추상 클래스이다. 이 `BaseRetriever`를 상속받아 커스텀 리트리버를 만들 수 있다. 문서 반환 시스템이란 앞서 설명한 것과 같이 **스트링 형식의 쿼리를 받아 가장 연관된 Document들을 반환하는 행위**로 정의될 수 있다.

Retriever는 [러너블 인터페이스](https://python.langchain.com/v0.1/docs/expression_language/interface/)를 따르기에  `invoke`, `batch`, `stream` 등의 러너블 메서드들을 통해 사용된다. 또한 커스텀 Retriever를 정의할 땐 반드시`_get_relevant_documents` 메서드를 오버라이딩해서 정의해 사용해야 한다.

소스 코드상에 딱히 딥하게 알아볼 코드는 없으며 추상 메서드로 정의된`_get_relevant_documents`를 하위 클래스에서 재정의 한다는 것만 알고 가면 될 것 같다. 스트링으로 된 query를 받아 Document의 리스트를 반환한다.

```python
@abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
```

`BaseRetriver`는 추상 클래스인 `ABC`([*참고](https://bluese05.tistory.com/61)) 를 상속받아 정의됐으며 `_get_relevant_document` 메서드는 `@abstractmethod` 데코레이터가 있기에 하위 클래스에서 **반드시** 재정의를 해야 한다.

(`get_releavant_documents`는 0.1.46버전 부터 deprecated 상태이며 0.3 버전 이후부터 삭제될 예정이라고 하니 사용 안하는 게 좋을 것 같다.)



#### 2. [VectorStoreRetriever](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.base.VectorStoreRetriever.html#langchain_core.vectorstores.base.VectorStoreRetriever)

벡터 스토어를 위한 Retriever이다. 가장 많이 사용하는 형식의 리트리버일 것이다. 기본 인스턴스 변수로 `VectorStore` 객체가 있으며 이 객체의 유사도 검색 메서드를 사용해서 `_get_relevant_documents` 메서드를 실행한다. 

다른 인스턴스 변수로 `search_type` 이 있는데  "similarity", "similarity_score_threshold", "mmr" 중 하나여야 한다. `allowed_search_types`은 클래스 변수로 list("similarity", "similarity_score_threshold", "mmr")이다. 

[코드](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores/base.py#L970) 내에서 살펴볼 것은 BaseRetriever와 마찬가지로 `_get_relevant_documents`이며 위의 세가지의 search_type 별로 분기 실행하는 if-elif-else 구문만으로 이루어진 간단한 구조의 메서드이다.

```python
def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
    if self.search_type == "similarity":
        docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
    elif self.search_type == "similarity_score_threshold":
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )
        docs = [doc for doc, _ in docs_and_similarities]
    elif self.search_type == "mmr":
        docs = self.vectorstore.max_marginal_relevance_search(
            query, **self.search_kwargs
        )
    else:
        raise ValueError(f"search_type of {self.search_type} not allowed.")
    return docs
```

즉 현재 리트리버에서 사용 중인 vectorStore에서 정의된 `similarity_search`, `similarity_search_with_relevance_scores`, `max_marginal_relevance_search`을 `search_type` 변수에 따라 다르게 쓰는 것일 뿐이다.