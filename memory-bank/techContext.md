# QUANTAXIS 技术上下文

## 核心技术栈
- 编程语言: 
  - 当前支持: Python 3.7-3.8
  - 计划升级: Python 3.12+ (首要开发任务)
- 数据库: 
  - MongoDB (主存储)
  - ClickHouse (分析)
  - TDengine (时序数据)
- 消息队列: Redis
- 容器化: Docker

## 开发环境
- 依赖管理: pip + requirements.txt
- 构建工具: Makefile
- 测试框架: pytest
- 文档生成: Sphinx

## 关键依赖
- 数据处理: pandas, numpy
- 可视化: matplotlib, plotly
- 交易接口: 各券商API封装
- 异步处理: asyncio, threading

## 数据处理技术实现
1. 数据重采样核心实现：
```python
def QA_data_min_resample(min_data, type_='5min'):
    CONVERSION = {
        'code': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'amount': 'sum'
    }
    # 处理不同市场交易时间分段
    part_1 = min_data.iloc[idx.indexer_between_time('9:30', '11:30')]
    part_2 = min_data.iloc[idx.indexer_between_time('13:00', '15:00')]
    # 使用pandas resample进行高效重采样
    return pd.concat([part_1_res, part_2_res])
```

2. 核心数据结构设计：
- 统一基类 `_quotation_base` 提供通用接口
- 市场特定数据结构继承基类：
  - `QA_DataStruct_Stock_day`
  - `QA_DataStruct_Future_min`
  - `QA_DataStruct_Stock_transaction` (Tick数据)

3. 性能优化技术：
- LRU缓存常用计算结果
- 使用numpy向量化操作
- 多线程处理IO密集型任务
- 异步获取市场数据

4. 特殊市场时间处理：
- 股票市场：处理开盘集合竞价(9:15-9:25)
- 期货市场：处理夜盘交易时段(21:00-次日2:30)
- 加密货币：7×24小时交易处理

## 开发工具链
1. 本地开发:
   - Docker Compose
   - Jupyter Notebook
   - VS Code

2. CI/CD:
   - Travis CI
   - Google Cloud Build

3. 部署:
   - Docker容器
   - Kubernetes (可选)

## 编码规范
- PEP8标准
- YAPF自动格式化
- Pylint静态检查
- MyPy类型检查
