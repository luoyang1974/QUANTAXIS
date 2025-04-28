# QUANTAXIS 系统架构

## 核心模块
1. QAData - 数据处理模块
   - 数据获取(QAFetch)
   - 数据存储
   - 数据清洗

2. QAEngine - 策略引擎
   - 回测框架
   - 事件驱动
   - 多线程支持

3. QAMarket - 市场模块
   - 行情处理
   - 订单管理
   - 账户系统

4. QAARP - 风险与绩效
   - 风险指标计算
   - 绩效分析
   - 报告生成

## 架构特点
- 模块化设计
- 事件驱动架构
- 支持分布式部署
- 容器化支持(Docker)

## 关键设计决策
1. 使用MongoDB作为主要数据存储
2. 采用异步IO处理高频数据
3. 抽象统一的交易接口
4. 基于Jupyter的可视化分析

## 数据流
```mermaid
graph TD
    A[市场数据源] --> B(QAFetch)
    B --> C[QAData存储]
    C --> D[QAEngine回测]
    D --> E[QAMarket执行]
    E --> F[QAARP分析]

## 数据处理架构
```mermaid
graph LR
    A[数据源] --> B[基础数据结构]
    B --> C[数据操作]
    C --> D[分析应用]
    
    B --> B1[股票日线 QA_DataStruct_Stock_day]
    B --> B2[股票分钟线 QA_DataStruct_Stock_min]  
    B --> B3[期货数据结构 QA_DataStruct_Future_*]
    B --> B4[Tick数据 QA_DataStruct_Stock_transaction]
    
    C --> C1[重采样 data_resample.py]
    C --> C2[复权处理 to_qfq/to_hfq]
    C --> C3[数据清洗]
    
    D --> D1[回测引擎]
    D --> D2[因子分析]
    D --> D3[可视化]

    C1 -->|支持| E[多种时间粒度]
    C1 -->|处理| F[不同市场交易时间]
```

关键特点：
1. 统一数据结构：所有市场数据继承自_quotation_base基类
2. 灵活重采样：支持从tick到日线的多级采样
3. 市场适配：特殊处理股票/期货/加密货币的交易时间
4. 高性能：使用lru_cache优化常用操作
5. 完整链条：从数据获取到分析应用的全流程支持
