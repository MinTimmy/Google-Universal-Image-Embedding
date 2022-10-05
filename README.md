# Google Universal Image Embedding
這是我大學的獨立研究，如果有哪些地方沒有註明來源出處的，且務必寄信通知我，我會立即補上


## 比賽介紹
由Google 主辦，總獎金為 50,000 美金，比賽時間： 2022/07/11 ~ 2022/10/11
[Competiition Link](https://www.kaggle.com/competitions/google-universal-image-embedding)
### 比賽目標
設計一個 Embedding Model 來把 raw image 弄成 dimension 小於 64 的 Feature Embedding，並把這些 Feature Embedding投影到 Hyperplane 做Clustering，相同 object type 的Feature Embedding要互相聚集，不同的要互相分離。而 Kaggle 所提供的 object type 有很多達11種類。
### 評分方式
為 mean Precision @ 5 metric(mP@5)，這方法與 KNN Algorithm 相似，把這些 Feature Embedding 分布在hyperplane上，選定一張照片為Query ，以它為中心向外搜尋前 5 張最相鄰的 image ， 如果有 n 個與 Query  的 object type 相同，其分數就是 n/5 ，並且每一張 image 都會當一次 Query，最終在把所得到 mP@5取平均就是為最終成績。


## 研究歷程
### Paired-based Loss Function
這個 Kaggle 的比賽並沒有提供任何訓練資料集，所以參考一名 Kaggle Grandmaster 所提供的 Dataset 11  其 object type 有 11 個，共132 K 張照片。我們閱讀CVPR 所發布有關Metric Learning 的論文，其中選擇了[Roth et al. (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Integrating_Language_Guidance_Into_Vision-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf)，最終評分的結果 mP@5 (表格一) 為 0.221，以下是對於此技術的說明：
* Pair Mining. 先在一個 batch 中隨機找出一個 anchor ，如果anchor與相同object type 組成 positive pair，與之不同object type 組成negative pair。
* Loss Function.為Multi-Similarity Loss [( Wang at al.(CVPR 2019) )](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)，其分群方式是以positive pair要互相拉進對方，而negative pair是要互相推離對方。
* Other. Backbone為ResNet50，並用 Knowledge Distillation 技術，把 Language Model當成輔助訓練。

程式碼參考自: [LanguageGuidance_for_DML](https://github.com/ExplainableML/LanguageGuidance_for_DML)



### ArcFace Loss with ResNet50
之後有在實作其他論文 ，但其效果都不好，所以我們參考了 [Google Landmark Recognition 2020](https://www.kaggle.com/competitions/landmark-recognition-2020/overview/description) 的比賽，設計新的架構，其 Backbone 為 ResNet50，並使用ArcFace Loss [( Deng et al. (CVPR 2019) )](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf)為 Loss Function， Deng et al. 嘗試把  Embedding Model 所產生的 Feature Embedding 都投影到一弧面，在這弧面上做 Softmax 的分類。 mP@5 (表格一)從 0.221到 0.182，退步幅度 17%

### ArcFace Loss with CLIP
之後繼續嘗試使用不同pre-trained 的 CNN model，如SE-ResNeXt、Inception、GoogLeNet，但效果都不如預期，最終嘗試了 Vision Transformer 的 CLIP [(Radford et.al. OpenAI)](https://openai.com/blog/clip/)，mP@5 (表格一)從  0.230 到 0.485，進步幅度 110%。

### Transition Layer
#### Batch Normalization Layer
由 Model(1) 和 Model(2)，在 CLIP 之後增加 Batch Normalization Layer，讓Linear Layer 的輸入可以限制成Normal Distribution，訓練的收斂速度也會比較快， 其mP@5進步幅度 9%。

#### Linear Layer
例圖一為 Model(1) 的 Learning Curve，可以發現其 Validation Loss 比 Training Loss 來的更低，代表發生 underfitting 的問題，所以增加Linear Layer 的輸出維度越高，來提高整體Trainable Parameter的數量，其mP@5進步幅度 18%。


### Dataset
[Dataset 11](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/340489)
[Dataset 17K](https://www.kaggle.com/code/motono0223/guie-clip-tensorflow-train-example/notebook)


## 結論
* ArcFace Loss的優勢。Paired-Based Loss Function 必須搭配 pair mining ，但我們的 Dataset 17K的object type 數量遠大於 batch size，所以每個 batch 幾乎只會有 negative pair；而ArcFace Loss 不需要做 pair mining，所以更能處理大量 object type 的資料集。
* CLIP 的優勢。Vision Transformer 是利用 image 和text 之間的關係做訓練，相較於CNN 只有用到 image 的資訊，更能產生更佳的Feature Embedding。
