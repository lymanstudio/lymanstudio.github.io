## FAISS에서 Document더미로 임베딩 벡터를 생성, 저장하는 순서

loader를 통해 생성해놓은 List[Document] 자료를 이용해 `FAISS`벡터스토어를 만드는 과정을 설명

- `from_documents`, `from_texts`, `__from`이렇게 세개의 클래스 메서드를 거친 후 실제 `FAISS` 객체를 `__from`에서 생성`
- `__add()` 에서 실제로 메모리에 저장, docstore도 만들어 줌

0. 코드 내 호출 블록

   ```py
   embedding = OpenAIEmbeddings()
   vectorStore = FAISS.from_documents(docs, embedding = embedding)
   ```

   from_documents()

1. ```py
   def from_documents(
           cls: Type[VST], # 벡터 스토어 타입
           documents: List[Document],
           embedding: Embeddings, # 임베딩 함수
           **kwargs: Any,
       ) -> VST: # 벡터 스토어 타입 (VST = TypeVar("VST", bound="VectorStore"))
       texts = [d.page_content for d in documents]
       metadatas = [d.metadata for d in documents]
       return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)
   ```

   텍스트와 메타 데이터를 나눠 from_texts 호출

2. from_texts()
   ```py
   @classmethod # 클래스 메서드
   def from_texts(
           cls,
           texts: List[str],
           embedding: Embeddings, # 임베딩 함수
           metadatas: Optional[List[dict]] = None,
           ids: Optional[List[str]] = None,
           **kwargs: Any,
       ) -> FAISS:
       embeddings = embedding.embed_documents(texts)
       return cls.__from(
           texts,
           embeddings,
           embedding,
           metadatas=metadatas,
           ids=ids,
           **kwargs,
       )
   ```

   임베딩 함수로 텍스트들에 대해 임베딩 먼저 실행 뒤(`embed_documents()`) `__from()` 메서드에 텍스트, 임베딩들(`embeddings`), 임베딩함수(`embedding`), 메타데이터 전달 & 호출

   - embed_documents() : 처음에 인자로 넣어준 임베딩 함수에서 실행, 해당 함수가 `langchain_core.embedding.Embeddings` 를 상속받아 만든 함수라면 됨

3. __from()

   ```py
   @classmethod
   def __from(
           cls,
           texts: Iterable[str],
           embeddings: List[List[float]],
           embedding: Embeddings,
           metadatas: Optional[Iterable[dict]] = None,
           ids: Optional[List[str]] = None,
           normalize_L2: bool = False,
           distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
           **kwargs: Any,
       ) -> FAISS:
       faiss = dependable_faiss_import() ## FAISS 임포트 후 클래스 자체를 객체로 반환
       if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
           index = faiss.IndexFlatIP(len(embeddings[0]))
       else:
           # Default to L2, currently other metric types not initialized.
           index = faiss.IndexFlatL2(len(embeddings[0]))
       docstore = kwargs.pop("docstore", InMemoryDocstore())
       index_to_docstore_id = kwargs.pop("index_to_docstore_id", {})
       vecstore = cls(
           embedding,
           index,
           docstore,
           index_to_docstore_id,
           normalize_L2=normalize_L2,
           distance_strategy=distance_strategy,
           **kwargs,
       )
       vecstore.__add(texts, embeddings, metadatas=metadatas, ids=ids)
       return vecstore
   ```

   실제 사용될 다양한 소스들 생성 단계

   - `faiss` : faiss 클래스 자체로 엔진에 해당
   - `docstore`: 현재 메모리에 올릴 문서 스토어, kwargs에 있으면 있던거 사용, 없으면 생성
     - `docstore`의 구조는 {`docs_store_id`: `Document`} 의 구조이다.
   - `index_to_docstore_id`:  {`index`: `docs_store_id`}로 구성된 딕셔너리,  `docstore` 와 마찬가지로 kwargs에 있으면 있던거 사용, 없으면 생성
   - `vecstore`: 이제서야 나오는 `FAISS` 벡터스토어 객체, 인자로 받고 위에서 만든 객체들을 이용해 초기화 된다. `FAISS` 클래스의 `__init__`호출로 생성

   마지막에 `__add`함수로 실제 메모리에 넣은 뒤 `vecstore`를 반환 후 종료

4. __add()
   ```py
   def __add(
           self,
           texts: Iterable[str],
           embeddings: Iterable[List[float]],
           metadatas: Optional[Iterable[dict]] = None,
           ids: Optional[List[str]] = None,
       ) -> List[str]:
       faiss = dependable_faiss_import()
   
       if not isinstance(self.docstore, AddableMixin):
           raise ValueError(
               "If trying to add texts, the underlying docstore should support "
               f"adding items, which {self.docstore} does not"
           )
   
       # texts와 metadatas의 사이즈(개수)가 같은지 체크
       _len_check_if_sized(texts, metadatas, "texts", "metadatas") 
       _metadatas = metadatas or ({} for _ in texts)
       
       # 다시 Document들을 만들어줌
       documents = [
           Document(page_content=t, metadata=m) for t, m in zip(texts, _metadatas)
       ]
   
       # documents와 embeddings, ids의 사이즈(개수)가 같은지 체크
       _len_check_if_sized(documents, embeddings, "documents", "embeddings") 
       _len_check_if_sized(documents, ids, "documents", "ids")
   	
       # ids 들이 중복이 있는지 체크
       if ids and len(ids) != len(set(ids)):
           raise ValueError("Duplicate ids found in the ids list.")
   
       # Add to the index. ## 실제 임베딩 벡터들을 index라는 곳에 저장하는 단계
       vector = np.array(embeddings, dtype=np.float32)
       if self._normalize_L2:
           faiss.normalize_L2(vector)
       self.index.add(vector)
   
       # Add information to docstore and index. ## Document와 추가 정보들을 저장하는 단계
       ids = ids or [str(uuid.uuid4()) for _ in texts]
       self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
       starting_len = len(self.index_to_docstore_id)
       index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
       self.index_to_docstore_id.update(index_to_id)
       return ids
   ```

   

5. 