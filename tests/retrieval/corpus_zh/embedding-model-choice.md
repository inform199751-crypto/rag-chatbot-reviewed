# 多語言 embedding 模型選型

`all-MiniLM-L6-v2` 是純英文模型，384 維。中文查詢在它的向量空間裡跟英文文件幾乎沒有交集，
相關度分數普遍落在 0.1 到 0.2 之間，跟雜訊分不出來。要支援中文一定要換模型，不是調參數能解決的。

## 候選

| 模型 | 維度 | 大小 | 中文表現 |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 90 MB | 不可用 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 470 MB | 可用，速度快 |
| BAAI/bge-m3 | 1024 | 2.2 GB | 好，但 CPU 上較慢 |
| text-embedding-3-small | 1536 | 需要 API | 好，但每次查詢都要連外 |

## 換模型的代價

Chroma 的 collection 在建立時就把維度寫死了，所以換成不同維度的模型不能只跑增量更新，
必須把 `vector_store/docs_index` 整個目錄刪掉重建。只跑 `--full-rebuild` 會報
`Collection expecting embedding with dimension of 384, got 1024`。

同維度但不同模型更危險：不會報錯，但索引裡的向量與查詢向量來自不同的語意空間，
檢索結果全是雜訊，而且沒有任何警告。

## 相關度門檻要跟著模型走

分數尺度是模型的性質，不是語料的性質。0.2 對 MiniLM 是合理的地板，
對多語言模型偏低——不相關的文字也常拿到 0.3 以上。換模型之後門檻要重新量，不能沿用。
