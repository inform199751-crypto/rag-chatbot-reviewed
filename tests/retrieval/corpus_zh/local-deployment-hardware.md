# 地端部署的硬體需求

## VRAM 怎麼估

模型權重加上 KV cache 再加約 0.5 GB 的 CUDA 開銷，總和要小於顯卡的 VRAM。
以 4 GB 的 GTX 1050 Ti 為例：3B 的 Q4_K_M 量化檔約 2.0 GB，加上 4096 context 的 KV cache 之後還有餘裕；
8B 的 Q4_K_M 是 4.9 GB，直接塞不下。

## 舊顯卡的兩個陷阱

**驅動版本決定可用的 CUDA。** llama.cpp 的 CUDA 容器映像會宣告最低 CUDA 版本，
`nvidia-container-cli` 在容器啟動前就會擋下不符的驅動，錯誤訊息是
`requirement error: unsatisfied condition: cuda>=12.8`。這一關跟 GPU 型號無關，純粹看驅動。

**Pascal 架構（sm_61）不一定被編進映像。** 過了驅動這關，模型載入時才可能出現
`no kernel image is available for execution on the device`。這代表映像沒有為該架構編譯核心，
只能退回 CPU 版映像。

## CPU 推論的實際速度

3B Q4_K_M 在六執行緒的 i7 上大約每秒 7 個 token，prompt 處理階段約每秒 25 到 40 個 token。
1B 明顯更快。純 CPU 可以用來驗證流程與做示範，但互動體驗跟 GPU 差距很大。

Embedding 反而適合放 CPU：`all-MiniLM-L6-v2` 只有 23M 參數，
把 VRAM 完整留給推論比較划算。
