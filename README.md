案例背景：

在进行每一笔股票交易时，交易者（股民）都要给其账户所在的
证券公司支付一些手续费，虽然单笔交易的手续费不高，但是股票市
场每日都有巨额的成交量，每一笔交易的手续费汇总起来，数量便相
当可观。这部分收入对于一些证券公司来说很重要，甚至可以占到营
业总收入的50%以上，因此，证券公司对于客户（即交易者）的忠诚度
和活跃度是很看重的。
如果一个客户不再通过某个证券公司交易，即该客户流失了，那
么该证券公司便损失了一个收入来源，因此，证券公司会搭建一套客
户流失预警模型来预测客户是否会流失，并对流失概率较大的客户采
取相应的挽回措施，因为通常情况下，获得新客户的成本比保留现有
客户的成本要高得多

数据的结构：

账户资金（元）	
最后一次交易距今时间（天）	
上月交易佣金（元）	
计交易佣金（元）	
本券商使用时长（年）	
是否流失

这里我们采用逻辑回归、随机森林、XGBoost、支持向量机、朴素贝叶斯，五种算法来进行预测，再横向比较；

<img width="1545" height="948" alt="image" src="https://github.com/user-attachments/assets/f6952e47-8cd9-4cbe-b238-79b63c24bb92" />


逻辑回归

混淆矩阵：

<img width="650" height="500" alt="image" src="https://github.com/user-attachments/assets/76501c91-25bb-4fe7-8d46-486fcb7283ef" />

ROC曲线：

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/c5f20ee2-ca5e-48ea-9f5b-c20250856095" />

KS曲线：

<img width="873" height="719" alt="image" src="https://github.com/user-attachments/assets/d8f7ad9e-7c0d-401a-823c-148867fff352" />


随机森林

混淆矩阵：

<img width="650" height="500" alt="image" src="https://github.com/user-attachments/assets/fe97050a-b0b3-4ce0-95be-4c5ad3f3a314" />

ROC曲线：

<img width="894" height="747" alt="image" src="https://github.com/user-attachments/assets/fca972f7-a806-4d5a-8e4f-7391a5fe40e6" />

KS曲线：

<img width="873" height="726" alt="image" src="https://github.com/user-attachments/assets/59127379-345d-420b-bc17-cb6b27715e40" />


XGBoost

混淆矩阵：

<img width="650" height="500" alt="image" src="https://github.com/user-attachments/assets/b1519980-1812-4c5e-b4d4-174ccd5e2236" />

ROC曲线：

<img width="867" height="726" alt="image" src="https://github.com/user-attachments/assets/da89b01d-5ffa-4140-9945-2af5d3ef662a" />

KS曲线：

<img width="878" height="738" alt="image" src="https://github.com/user-attachments/assets/d393d53b-dcaf-41c5-9c49-e64d738f5936" />


支持向量机

混淆矩阵：

<img width="650" height="500" alt="image" src="https://github.com/user-attachments/assets/2f1a6a7e-aaf3-4e2f-953a-463ce825e269" />

ROC曲线：

<img width="861" height="723" alt="image" src="https://github.com/user-attachments/assets/18ebee52-07bd-435d-95a7-6ba5319f85dd" />

KS曲线：

<img width="873" height="726" alt="image" src="https://github.com/user-attachments/assets/cabd92c6-5790-45b6-8334-1e7dec3bf7b3" />


朴素贝叶斯

混淆矩阵：

<img width="650" height="500" alt="image" src="https://github.com/user-attachments/assets/ce270f30-fa4c-4f71-aa3c-9e549abe940c" />

ROC曲线：

<img width="884" height="737" alt="image" src="https://github.com/user-attachments/assets/09085080-10dc-4eb5-a7f5-488b7fb59888" />

KS曲线：

<img width="878" height="729" alt="image" src="https://github.com/user-attachments/assets/da840f5a-c30d-45c6-9f22-901538193aa5" />

分析总结：

准确率最高：SVM ≈ 0.798
流失用户召回率最高：朴素贝叶斯 ≈ 0.669
KS值最高（区分能力最强）：SVM ≈ 0.477 > 逻辑回归 ≈ 0.474 > XGBoost ≈ 0.430
数据不平衡问题明显：大部分模型准确率高是因为负类占比多，而流失用户识别能力受影响。

改进方向：

处理类别不平衡（SMOTE、欠采样、调整 class_weight）
特征工程（增加有区分能力的新特征）
模型融合（Ensemble）或阈值调整提高召回率
对 SVM/XGBoost 调参提高 KS 值和召回率
