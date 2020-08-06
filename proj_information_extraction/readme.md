### MSRA NER

1. 标签体系: BIOE
2. 三种实体: 人物, 地点, 机构 

| 模型  |备注  | Precision |Recall   |F1  |Time/Epoch  | 参数|
|---|---|---|---|---|---|---|
| bilstm_crf  |  普通训练 | 82%  | 80%  | 81%  |25min/15   | lr=1e-3,batch_size=200,max_len=150,embedding_dim=100,hidden_dim=200 |
| bert  | Fine-tune BertForTokenClassification |95%  |95%   | 95%  | 1.5h/11  | lr=1e-5,batch_size=32,max_len=150,embedding_dim=768  |
| bert  | Feature-based BertForTokenClassification| 51%  | 38%  | 44%  | 50min/10 |lr=1e-3,batch_size=32,max_len=150,embedding_dim=768  |
| bert_bilstm_crf  | Fine-tune bert | 91%  | 91%  | 91%  | 30min/16  |lr=1e-5,batch_size=200,max_len=150,embedding_dim=768,hidden_dim=500(lstm)  |
| bert_bilstm_crf  | Fine-tune bert | 92%  | 91%  | 92%  | 30min/15  |bert_lr=1e-5,bilstm_crf_lr=1e-3,batch_size=200,max_len=150,embedding_dim=768,hidden_dim=500(lstm)|
| bert_bilstm_crf  | Feature-based bert| 86%  | 86%  | 86%  | 30min/43 |lr=1e-3,batch_size=200,max_len=150,embedding_dim=768,hidden_dim=500(lstm)  |


注：

* 相应调整lr和hidden_dim明显提升f1，随着hidden_dim的增大，f1的提升**变得缓慢**
* bert_bilstm_crf中，bert和bilstm_crf使用不同的学习率 (第五行)相比于使用相同的学习率 (第四行)，f1只有接近1%的提升
* **fine-tune bert使用小的lr** (1e-5)才可以获得f1的巨大提升(相比于使用1e-3的f1一直为0)
* feature-based bert (即使用bert输出作为embedding输入下游模型)，使用稍大的lr (1e-3)可以保证较好效果，训练速度也会迅速提高
* **fine-tune bert原模型 (第二行，外加一个Linear层+Softmax)获得最好效果，f1达95%**
